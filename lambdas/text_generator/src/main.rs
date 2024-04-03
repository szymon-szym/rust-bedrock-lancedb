pub mod config;

use arrow::array::as_string_array;
use arrow_array::RecordBatch;
use aws_config::meta::region::RegionProviderChain;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::primitives::Blob;
use futures::TryStreamExt;
use std::{fs, io::Write, sync::Arc, time::Duration};

use lambda_runtime::{
    run, service_fn,
    tracing::{self, info, instrument},
    Error, LambdaEvent,
};

use lancedb::{connect, query::{ExecutableQuery, QueryBase}, Table};

use serde::{Deserialize, Serialize};

use config::Config;
use clap::Parser;

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct TitanResponse {
    embedding: Vec<f32>,
    input_text_token_count: i128,
}

#[derive(Deserialize, Serialize)]
struct EmbeddingsLanceDb {
    text: String,
    item: f32,
}

#[derive(Debug, serde::Deserialize)]
struct CloudeResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    model: String,
    stop_reason: String,
    stop_sequence: Option<String>,
    usage: Usage,
    content: Vec<Content>,
}

#[derive(Debug, serde::Deserialize)]
struct Content {
    #[serde(rename = "type")]
    response_type: String,
    text: String,
}

#[derive(Debug, serde::Deserialize)]
struct Usage {
    input_tokens: i128,
    output_tokens: i128,
}

#[derive(Deserialize)]
struct Request {
    prompt: String,
}

#[derive(Serialize)]
struct Response {
    req_id: String,
    msg: String,
}

#[instrument(skip_all)]
async fn function_handler(
    table: &Table,
    client: &aws_sdk_bedrockruntime::Client,
    event: LambdaEvent<Request>,
) -> Result<Response, Error> {

    let start_time_embeddings = std::time::Instant::now();

    // Extract some useful info from the request
    let prompt = event.payload.prompt;

    info!("got prompt from request: {}", prompt);

    // transform prompt to embeddings
    let embeddings_prompt = format!(
        r#"{{
        "inputText": "{}"
    }}"#,
        prompt
    );

    info!("invoking embeddings model with: {}", embeddings_prompt);

    let invocation = client
        .invoke_model()
        .content_type("application/json")
        .model_id("amazon.titan-embed-text-v1")
        .body(Blob::new(embeddings_prompt.as_bytes().to_vec()))
        .send()
        .await
        .unwrap();

    let titan_response =
        serde_json::from_slice::<TitanResponse>(&invocation.body().clone().into_inner()).unwrap();

    let embeddings = titan_response.embedding;

    info!("got embeddings for prompt from model in {} seconds", Duration::from(start_time_embeddings.elapsed()).as_secs_f32());

    let start_time_vector_db = std::time::Instant::now();

    let result: Vec<RecordBatch> = table
        .query()
        .limit(2)
        .nearest_to(embeddings)?
        .execute()
        .await?
        // .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let items = result
        .iter()
        .map(|batch| {
            let text_batch = batch.column(1);
            let texts = as_string_array(text_batch);
            texts
        })
        .flatten()
        .collect::<Vec<_>>();

    // info!("items {:?}", &items);

    let context = items
        .first()
        .unwrap()
        .unwrap_or("")
        .replace("\u{a0}", "")
        .replace("\n", " ")
        .replace("\t", " ");

    println!("context: {:?}", context);
    info!("got context in {}", Duration::from(start_time_vector_db.elapsed()).as_secs_f32());

    let start_time_llm = std::time::Instant::now();

    let prompt_for_llm = format!(
        r#"{{
        "system": "Respond only in Polish. Informative style. Information focused on health and safety for kids during vacations. Keep it short and use max 500 words. Please use examples from the following document in Polish: {}",
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [
            {{
                "role": "user",
                "content": [{{
                    "type": "text",
                    "text": "{}"
                }}]
            }}
        ]
    }}"#,
        context, prompt
    );

    // println!("prompt {:?}", prompt_for_llm);

    let generate_invocation = client
        .invoke_model()
        .content_type("application/json")
        .model_id("anthropic.claude-3-sonnet-20240229-v1:0")
        .body(Blob::new(prompt_for_llm.as_bytes().to_vec()))
        .send()
        .await
        .unwrap();

    let raw_response = generate_invocation.body().clone().into_inner();

    let generated_response = serde_json::from_slice::<CloudeResponse>(&raw_response).unwrap();

    println!("{:?}", generated_response.content);
    info!("got llm answer in {}", Duration::from(start_time_llm.elapsed()).as_secs_f32());

    // Prepare the response
    let resp = Response {
        req_id: event.context.request_id,
        msg: format!("Response {:?}.", generated_response),
    };

    Ok(resp)
}

#[instrument(skip_all)]
#[tokio::main]
async fn main() -> Result<(), Error> {

    let start_time_sdk = std::time::Instant::now();

    tracing::init_default_subscriber();

    info!("starting lambda");

    dotenv::dotenv().ok();
    let env_config = Config::parse();

    // set up aws sdk config
    let region_provider = RegionProviderChain::default_provider().or_else("us-east-1");
    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(region_provider)
        .load()
        .await;

    // initialize sdk client
    let bedrock_client = aws_sdk_bedrockruntime::Client::new(&config);

    info!("sdk clients initialized in {}", Duration::from(start_time_sdk.elapsed()).as_secs_f32());

    let bucket_name = env_config.bucket_name;
    let prefix = env_config.prefix;
    let table_name = env_config.table_name;

    let start_time_lance = std::time::Instant::now();
    
    let s3_db = format!("s3://{}/{}/", bucket_name, prefix);

    info!("bucket string {}", s3_db);

    // set AWS_DEFAULT_REGION env 

    std::env::set_var("AWS_DEFAULT_REGION", "us-east-1");

    let db = connect(&s3_db).execute().await?;

    info!("connected to db {:?}", db.table_names().execute().await);
    
    let table = db.open_table(&table_name).execute().await?;

    info!("connected to db in {}", Duration::from(start_time_lance.elapsed()).as_secs_f32());

    run(service_fn(|event: LambdaEvent<Request>| {
        function_handler(&table, &bedrock_client, event)
    }))
    .await
}
