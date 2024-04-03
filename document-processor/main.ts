import { BedrockEmbeddings } from "langchain/embeddings/bedrock";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { LanceDB } from "langchain/vectorstores/lancedb";

import { connect } from "vectordb"; // LanceDB

import dotenv from "dotenv";

dotenv.config();

(async () => {
  const dir = process.env.LANCEDB_BUCKET || "missing_s3_folder";
  const lanceDbTable = process.env.LANCEDB_TABLE || "missing_table_name";
  const awsRegion = process.env.AWS_REGION;

  console.log("lanceDbSrc", dir);
  console.log("lanceDbTable", lanceDbTable);
  console.log("awsRegion", awsRegion);

  const path = `documents/poradnik_bezpiecznego_wypoczynku.pdf`;

  const splitter = new CharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const embeddings = new BedrockEmbeddings({
    region: awsRegion,
    model: "amazon.titan-embed-text-v1",
  });

  const loader = new PDFLoader(path, {
    splitPages: false,
  });

  const documents = await loader.loadAndSplit(splitter);

  const db = await connect(dir);

  console.log("connected")

  const table = await db.openTable(lanceDbTable).catch((_) => {
    console.log("creating new table", lanceDbTable)
    return db.createTable(lanceDbTable, [
        { 
          vector: Array(1536), 
          text: 'sample',
        },
      ])
  })

  const preparedDocs = documents.map(doc => ({
    pageContent: doc.pageContent,
    metadata: {}
  }))

  await LanceDB.fromDocuments(preparedDocs, embeddings, { table })
  
})();
