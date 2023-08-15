"""CDK stack for similarity search backend version 2"""
import os
from typing import Any
import requests
from constructs import Construct
from aws_cdk import Stack, App, Tags, Environment, CfnOutput
from aws_cdk import aws_rds
from aws_cdk import aws_ec2
from aws_cdk import aws_ecs
from aws_cdk import aws_s3
from aws_cdk import aws_logs
from aws_cdk import aws_iam

BUCKET_NAME = "similarity-search-base-v2-dev-bucket43879c71-31yyhrd3ipdf"

env = Environment(
    account=os.environ["CDK_DEFAULT_ACCOUNT"], region=os.environ["CDK_DEFAULT_REGION"]
)

app = App()

class pgVectorsTest(Stack):
    """VPC Stack"""

    def __init__(
        self,
        scope: Construct,
        id: str,
        **kwargs: Any,
    ) -> None:
        """Define stack."""
        super().__init__(scope, id, **kwargs)

        imported_vpc_id = os.environ.get("VPC_ID")

        if imported_vpc_id:
            print(f"Deploying into existing VPC: {imported_vpc_id}")
            vpc = aws_ec2.Vpc.from_lookup(self, "vpc", vpc_id=imported_vpc_id)
        else:
            # Create a VPC with public and private subnets
            # No need for `subnet_configuration` parameter, VPC creates a public and private
            # subnet per availability zone by default
            vpc = aws_ec2.Vpc(self, "vpc", max_azs=2)

        bucket = aws_s3.Bucket.from_bucket_name(self, 'bucket', BUCKET_NAME)

        engine = aws_rds.DatabaseInstanceEngine.postgres(
            version=aws_rds.PostgresEngineVersion.VER_15_3
        )

        rds = aws_rds.DatabaseInstance(
            self, 
            id="rds-database", 
            instance_type=aws_ec2.InstanceType("t3.large"),
            instance_identifier=id,
            engine=engine,
            database_name="pgVectors", 
            vpc_subnets=aws_ec2.SubnetSelection(
                subnet_type= aws_ec2.SubnetType.PUBLIC
            ), 
            credentials=aws_rds.Credentials.from_generated_secret(
                username="postgres",
           ),
           vpc=vpc
        )

        task_definition = aws_ecs.FargateTaskDefinition(
            self,
            "pgvectors-ingest-task",
            cpu=8192, # 8 Gb
            memory_limit_mib=61440, # 64 Gb
        )

        task_definition.add_to_task_role_policy(
            aws_iam.PolicyStatement(
                actions=["s3:ListBucket"], resources=[bucket.bucket_arn]
            )
        )
        task_definition.add_to_task_role_policy(
            aws_iam.PolicyStatement(
                actions=["s3:*Object"], resources=[f"{bucket.bucket_arn}/*"]
            )
        )
        task_definition.add_to_task_role_policy(
            aws_iam.PolicyStatement(
                actions=["secretsmanager:GetSecretValue"], resources=[rds.secret.secret_arn]
            )
        )

        task_definition.add_to_execution_role_policy(
            aws_iam.PolicyStatement(
                actions=["secretsmanager:GetSecretValue"], resources=[rds.secret.secret_arn]
            )
        )

        # Add a container to the task definition
        container = task_definition.add_container(
            'pgvectors-ingest-container',
            container_name="pg-vectors-ingest-container", 
            image=aws_ecs.ContainerImage.from_asset(
                directory=".",
                file="Dockerfile"
            ),
            secrets = {
                "host": aws_ecs.Secret.from_secrets_manager(rds.secret, "host"),
                "dbname": aws_ecs.Secret.from_secrets_manager(rds.secret, "dbname"),
                "user": aws_ecs.Secret.from_secrets_manager(rds.secret, "username"),
                "password": aws_ecs.Secret.from_secrets_manager(rds.secret, "password"),
                "port": aws_ecs.Secret.from_secrets_manager(rds.secret, "port"),
            }, 
            command=["python", "ingest.py"],
            environment={"BUCKET_NAME": bucket.bucket_name},
            logging=aws_ecs.LogDriver.aws_logs(
                stream_prefix=f"{id}-container-logs",
                log_retention=aws_logs.RetentionDays.ONE_MONTH,
            ),
        )


        # Create an ECS service with the task definition
        ecs_service = aws_ecs.FargateService(
            self, "fargate-service",
            cluster=aws_ecs.Cluster(
                self, 
                "ecs-cluster",
                vpc=vpc  
            ),
            task_definition=task_definition,
        )

        # enable ingress from 
        rds.connections.security_groups[0].add_ingress_rule(ecs_service.connections.security_groups[0], aws_ec2.Port.tcp(5432))
        
        crt_ip = requests.get('https://api.ipify.org').content.decode('utf8')
        rds.connections.security_groups[0].add_ingress_rule(aws_ec2.Peer.ipv4(f"{crt_ip}/32"), aws_ec2.Port.tcp(5432))

        db_secrets_output = CfnOutput(
            self, 
            "database-secrets-arn", 
            value=rds.secret.secret_arn
        )

        task_definition_output = CfnOutput(
            self, 
            "task-definition-arn", 
            value=task_definition.task_definition_arn
        )



pg_vectors_test_stack = pgVectorsTest(
    app, 
    "pg-vectors-similarity-search-test", 
    env=env
)
       

# TODO: add tags with github branch + commit hash + timestamp + author
# TODO: add env var for STAGE
# Tag infrastructure
for key, value in {
    "Project": "similarity-search-pg-vectors-test",
    "Stack": f"Test",
    "Owner": "leo@developmentseed.org",
}.items():
    if value:
        Tags.of(app).add(key, value)
        
app.synth()
