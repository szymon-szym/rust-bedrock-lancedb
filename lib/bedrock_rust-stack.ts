import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as iam from "aws-cdk-lib/aws-iam";
import { RustFunction } from "cargo-lambda-cdk";

export class BedrockRustStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // create s3 bucket for vector db
    const vectorDbBucket = new s3.Bucket(this, "lancedb-vector-bucket", {
      versioned: true,
    });

    new cdk.CfnOutput(this, "vector-bucket-name", {
      value: vectorDbBucket.bucketName,
    });

    const textGeneratorLambda = new RustFunction(this, "text-generator", {
      manifestPath: "lambdas/text_generator/Cargo.toml",
      environment: {
        BUCKET_NAME: vectorDbBucket.bucketName,
        PREFIX: "lance_db",
        TABLE_NAME: "embeddings",
      },
      memorySize: 512,
      timeout: cdk.Duration.seconds(30),
    });

    vectorDbBucket.grantRead(textGeneratorLambda);

    // add policy to allow calling bedrock
    textGeneratorLambda.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["bedrock:InvokeModel"],
        resources: [
          "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1",
          "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
        ],
        effect: iam.Effect.ALLOW,
      })
    );
  }
}
