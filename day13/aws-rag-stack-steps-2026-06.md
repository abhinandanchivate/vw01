# AWS RAG stack — full build (Console + CLI/code)

*The AWS equivalent of the Azure §6.3–6.7 RAG stack. Console (GUI) steps **and** runnable AWS CLI / boto3, current to June 2026. Region used throughout: `us-east-1`.*

## Service mapping (Azure → AWS)

| Azure piece | AWS equivalent used here |
|---|---|
| Azure AI Search (vector + hybrid + semantic rerank) | **Amazon OpenSearch Serverless** (vector search collection) + **Bedrock Rerank** for semantic reranking. *(Managed alt: Bedrock Knowledge Bases.)* |
| Azure OpenAI / Foundry chat + embedding deployments | **Amazon Bedrock** — Claude / Nova for chat, **Titan Text Embeddings V2** for embeddings |
| "Azure OpenAI on your data" / Foundry IQ | **Amazon Bedrock Knowledge Bases** (managed RAG) |
| Key Vault | **AWS Secrets Manager** (+ Parameter Store / KMS) |
| Application Insights | **Amazon CloudWatch** + **AWS X-Ray** / **CloudWatch Application Signals** |
| App Service / Container Apps (FastAPI host) | **Amazon ECS Express Mode** (App Runner's successor) |
| Managed identity + RBAC | **IAM roles** (task role / execution role) + IAM policies |

## Read first — three June-2026 gotchas

1. **App Runner is closed to new customers (since Apr 30, 2026).** It still runs for existing customers, but for a net-new build AWS now points you at **Amazon ECS Express Mode**, which gives the same one-form experience (Fargate + ALB + HTTPS + autoscaling) without the black box. Steps below use Express Mode.
2. **Newer Bedrock models require an inference profile** (cross-region), not the bare model ID. If you call `anthropic.claude-...` on-demand and get *"isn't supported, retry with an inference profile,"* prefix the ID with your region group (`us.`, `eu.`, `apac.`, …) or use the profile ARN. Get exact IDs from **Bedrock → Inference profiles** or `aws bedrock list-inference-profiles`.
3. **OpenSearch Serverless has a ~4-OCU floor** (~$300+/mo even when idle). For dev or cost-sensitive RAG, **Aurora PostgreSQL Serverless (pgvector)** is a supported, cheaper vector store for Knowledge Bases. Pick deliberately.

---

## A.3 ≈ Vector store + hybrid search (Azure AI Search)

For the **explicit pipeline** (full control over chunking/prompts/citations), the vector DB is an **OpenSearch Serverless vector search collection**. Semantic reranking — Azure's "semantic ranker" — is done with **hybrid search (k-NN + BM25, fused with RRF/normalization)** and optionally the **Bedrock Rerank API** (Cohere Rerank 3.5 or Amazon Rerank 1.0) as a second-stage reranker.

### Console

1. **OpenSearch Service** console → left nav **Serverless → Collections → Create collection**.
2. Use the **NextGen / Express create** flow (auto-configures encryption, network, and data-access policies):
   - **Name**: `rag-vectors`
   - **Collection type**: **Vector search**
   - **Deployment**: enable redundancy for prod (disable for cheapest dev).
   - **Security/Network**: Easy create for dev; for prod choose **Private** + a VPC (PrivateLink) endpoint and add `bedrock.amazonaws.com` if a Knowledge Base will write to it.
3. **Create**, then note the **Collection ARN** (`arn:aws:aoss:us-east-1:<acct>:collection/<id>`).
4. Create the **vector index** (in the collection → **Indexes**, or via Dev Tools / OpenSearch API). It needs a `knn_vector` field whose **dimension matches your embedding model** (1024 for Titan V2), plus a `text` field and a `metadata` field. *(If you instead let Bedrock "Quick create" the store in A.2, it builds this index for you.)*

### CLI

The Express console flow bundles the three policies; explicitly it's:

```bash
# 1) Encryption policy (AWS-owned key shown; use a KMS key for CMK)
aws opensearchserverless create-security-policy --name rag-enc --type encryption \
  --policy '{"Rules":[{"ResourceType":"collection","Resource":["collection/rag-vectors"]}],"AWSOwnedKey":true}'

# 2) Network policy (public for dev; switch to VPC for prod)
aws opensearchserverless create-security-policy --name rag-net --type network \
  --policy '[{"Rules":[{"ResourceType":"collection","Resource":["collection/rag-vectors"]},{"ResourceType":"dashboard","Resource":["collection/rag-vectors"]}],"AllowFromPublic":true}]'

# 3) Data-access policy — grant your app/task role and (if used) the Bedrock KB role
aws opensearchserverless create-access-policy --name rag-data --type data \
  --policy '[{"Rules":[{"ResourceType":"index","Resource":["index/rag-vectors/*"],"Permission":["aoss:*"]},{"ResourceType":"collection","Resource":["collection/rag-vectors"],"Permission":["aoss:*"]}],"Principal":["arn:aws:iam::<acct>:role/rag-task-role"]}]'

# 4) The collection itself
aws opensearchserverless create-collection --name rag-vectors --type VECTORSEARCH
```

> **Reranking (the "semantic ranker" analog).** After hybrid retrieval, optionally rerank with Bedrock:
> ```python
> import boto3
> rr = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
> out = rr.rerank(
>     queries=[{"type":"TEXT","textQuery":{"text":"your user query"}}],
>     sources=[{"type":"INLINE","inlineDocumentSource":{"type":"TEXT","textDocument":{"text": d}}} for d in candidate_chunks],
>     rerankingConfiguration={"type":"BEDROCK_RERANKING_MODEL","bedrockRerankingConfiguration":{
>         "numberOfResults":5,
>         "modelConfiguration":{"modelArn":"arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"}}})
> ```

---

## A.2 ≈ Bedrock models: chat + embeddings (Azure OpenAI / Foundry)

Unlike Azure, you don't "deploy" base models — you **enable model access once per account/region**, then invoke by model ID (or inference-profile ID). Per-deployment capacity is replaced by **per-account quotas** (TPM for chat, **RPM** for embeddings).

### Console — enable model access

1. **Amazon Bedrock** console → left nav **Model access** → **Modify model access** / **Enable specific models**.
2. Select the models you need and **Submit** (Anthropic models may ask for brief use-case details; approval is usually immediate):
   - **Chat**: an Anthropic Claude model and/or Amazon Nova.
   - **Embeddings**: **Titan Text Embeddings V2**.
3. *(Optional)* **Inference profiles** → confirm the cross-region profile for your chat model exists in your region.

**Choosing models** (maps to the Azure chat/embedding/deployment-type table):

| Need | 2026 pick on Bedrock |
|---|---|
| Balanced chat (most RAG) | **Claude Sonnet 4.6** (`anthropic.claude-sonnet-4-6`) |
| Cheapest chat | **Claude Haiku 4.5** or **Amazon Nova 2 Lite** |
| Hardest reasoning | **Claude Opus 4.8 / 4.7** *(note: no `temperature`/`top_p`/`top_k`; steer via prompt; thinking is "adaptive")* |
| Auto-pick per request | **Intelligent Prompt Routing** (≈ Azure Model Router) |
| Embeddings (default) | **Titan Text Embeddings V2** — 1024-dim (configurable 256/512), RAG-optimized, English-leaning |
| Embeddings (multilingual) | **Cohere Embed v4** or **Nova 2 Multimodal Embeddings** |

> The embedding **dimension must match your vector index** (1024 if you keep Titan V2's default). Titan V2 retains ~99% accuracy at 512 and ~97% at 256 if you want cheaper storage.

### Code — chat (Converse API)

```python
import boto3
brt = boto3.client("bedrock-runtime", region_name="us-east-1")

resp = brt.converse(
    # Use the cross-region inference-profile ID; confirm via `aws bedrock list-inference-profiles`
    modelId="us.anthropic.claude-sonnet-4-6-v1:0",
    system=[{"text": "Answer only from the provided context. Cite sources. If unsure, say so."}],
    messages=[{"role": "user", "content": [{"text": f"Context:\n{retrieved}\n\nQuestion: {question}"}]}],
    inferenceConfig={"maxTokens": 1024, "temperature": 0.2},  # omit temperature for Opus 4.7+
)
print(resp["output"]["message"]["content"][0]["text"])
```

### Code — embeddings (InvokeModel)

```python
import boto3, json
brt = boto3.client("bedrock-runtime", region_name="us-east-1")

def embed(text: str) -> list[float]:
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    r = brt.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=body)
    return json.loads(r["body"].read())["embedding"]
```

---

## A.3 (managed alternative) ≈ "Azure OpenAI on your data" → Bedrock Knowledge Bases

This is the managed-RAG path (parallel to the Azure doc's built-in-retrieval alternative). Bedrock handles S3 ingestion, chunking, embedding (Titan), writing to OpenSearch Serverless, and `RetrieveAndGenerate`.

### Console

1. **Bedrock** → **Knowledge Bases** → **Create** → **Knowledge Base with vector store**.
2. **IAM**: let Bedrock **create a service role** (simplest) or pick your own.
3. **Data source**: **Amazon S3** → give the bucket/prefix holding your docs (up to 5 sources).
4. **Embeddings model**: **Titan Text Embeddings V2**.
5. **Vector store**: **Quick create a new vector store → OpenSearch Serverless** (Bedrock builds the collection + index with matching dimensions), **or** point to the `rag-vectors` collection from A.3 (then field names/dimension must match exactly — the #1 source of errors).
6. **Create**, then open the **Data source** tab → **Sync** to ingest.

### CLI

```bash
# kb.json
cat > kb.json <<'JSON'
{
  "name": "rag-kb",
  "roleArn": "arn:aws:iam::<acct>:role/service-role/AmazonBedrockExecutionRoleForKnowledgeBase_rag",
  "knowledgeBaseConfiguration": {
    "type": "VECTOR",
    "vectorKnowledgeBaseConfiguration": {
      "embeddingModelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
      "embeddingModelConfiguration": {"bedrockEmbeddingModelConfiguration": {"dimensions": 1024}}
    }
  },
  "storageConfiguration": {
    "type": "OPENSEARCH_SERVERLESS",
    "opensearchServerlessConfiguration": {
      "collectionArn": "arn:aws:aoss:us-east-1:<acct>:collection/<id>",
      "vectorIndexName": "rag-index",
      "fieldMapping": {"vectorField": "vector", "textField": "text", "metadataField": "metadata"}
    }
  }
}
JSON

aws bedrock-agent create-knowledge-base --cli-input-json file://kb.json
aws bedrock-agent create-data-source --cli-input-json file://ds.json     # points the KB at your S3 bucket
aws bedrock-agent start-ingestion-job --knowledge-base-id <KB_ID> --data-source-id <DS_ID>
```

### Query

```python
import boto3
agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
resp = agent.retrieve_and_generate(
    input={"text": "What are the termination clauses in our supplier agreements?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "<KB_ID>",
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6",
        },
    },
)
print(resp["output"]["text"])   # citations are in resp["citations"]
```

---

## A.4 ≈ Secrets (Key Vault)

With IAM task roles you need very few stored secrets (e.g., a SharePoint app secret). Use **Secrets Manager** for genuine secrets, **Parameter Store** for plain config, **KMS** for keys.

### Console
**Secrets Manager → Store a new secret → Other type of secret →** enter key/value(s) → name it `rag/app` → (optionally enable rotation) → **Store**.

### CLI
```bash
aws secretsmanager create-secret --name rag/app \
  --secret-string '{"SHAREPOINT_CLIENT_SECRET":"xxxx"}'
```

### Read at runtime
```python
import boto3, json
sm = boto3.client("secretsmanager", region_name="us-east-1")
secret = json.loads(sm.get_secret_value(SecretId="rag/app")["SecretString"])
```
Grant the task role `secretsmanager:GetSecretValue` on this secret (see "Wire it together").

---

## A.5 ≈ Observability (Application Insights)

There's no single "Application Insights" resource — you assemble it:

- **CloudWatch Logs** — your app + ECS task logs (Express Mode auto-creates a `/ecs/<service>` log group).
- **CloudWatch metrics / Container Insights** — request, CPU, memory, scaling metrics (Express Mode enables Container Insights).
- **AWS X-Ray** + **CloudWatch Application Signals** — the APM/tracing analog: dependency calls to Bedrock and OpenSearch, latency, faults.

### Setup
1. Instrument the FastAPI app with the **AWS Distro for OpenTelemetry (ADOT)** SDK (or the X-Ray SDK) to emit traces; boto3/Bedrock/OpenSearch calls then show as dependencies.
2. In **CloudWatch → Application Signals**, enable the service to get an APM-style dashboard (latency, error rate, dependency map).
3. Logs and X-Ray traces appear automatically once the task role has `xray:PutTraceSegments` / `logs:*` (the ECS execution role covers logs).

---

## A.6 ≈ Serving the FastAPI app (App Service / Container Apps)

**Recommended: Amazon ECS Express Mode.** One form/command provisions a Fargate service, a shared ALB, HTTPS, autoscaling, and CloudWatch logs — and every resource stays editable later. (Alternatives at the end.)

### Step 1 — containerize (FastAPI)

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
# gunicorn + uvicorn workers, same pattern as the Azure startup command
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8080"]
```
`requirements.txt` minimally: `fastapi`, `uvicorn[standard]`, `gunicorn`, `boto3`, `opensearch-py`.
Add a health route so Express Mode's health check passes:
```python
@app.get("/health")
def health(): return {"ok": True}
```

### Step 2 — push to ECR

```bash
aws ecr create-repository --repository-name rag-api
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker build --platform=linux/amd64 -t rag-api .
docker tag rag-api:latest $ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/rag-api:latest
docker push $ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/rag-api:latest
```

### Step 3a — deploy via Console

1. **ECS** console → **Express Mode** → **Create**.
2. **Container image**: the ECR URI `…/rag-api:latest`.
3. **Container port**: `8080`. **Health check path**: `/health`.
4. **Environment variables**: e.g. `AWS_REGION`, `KB_ID`, `OPENSEARCH_ENDPOINT`.
5. **Task execution role**: create new (gets `AmazonECSTaskExecutionRolePolicy` — pulls image, writes logs).
6. **Infrastructure role**: create new (`AmazonECSInfrastructureRoleforExpressGatewayServices`).
7. **Task role** *(important — this is your "managed identity")*: attach the RAG policy from "Wire it together" so the app can call Bedrock / Secrets / OpenSearch.
8. Set autoscaling (default: CPU 60%, min 1, max 20) and networking (default VPC, or your own) → **Create**. The console shows the **Application URL** when ready.

### Step 3b — deploy via CLI

```bash
# One-time IAM roles for Express Mode
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

aws iam create-role --role-name ecsInfrastructureRoleForExpressServices \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name ecsInfrastructureRoleForExpressServices \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices

# Create the service
aws ecs create-express-gateway-service \
  --primary-container '{"image":"'$ACCOUNT'.dkr.ecr.us-east-1.amazonaws.com/rag-api:latest","containerPort":8080,"environment":[{"name":"AWS_REGION","value":"us-east-1"}]}' \
  --service-name "rag-api" \
  --execution-role-arn arn:aws:iam::$ACCOUNT:role/ecsTaskExecutionRole \
  --infrastructure-role-arn arn:aws:iam::$ACCOUNT:role/ecsInfrastructureRoleForExpressServices \
  --health-check-path "/health" \
  --cpu 1 --memory 2 \
  --scaling-target '{"minTaskCount":1,"maxTaskCount":20}' \
  --monitor-resources
```
Update later by pushing a new image and **Update** in the console (or re-running with the new tag).

### Alternatives

| Option | Best for |
|---|---|
| **Lambda + API Gateway** (FastAPI via Mangum) | Spiky/low traffic, true scale-to-zero, pay-per-request |
| **ECS on Fargate** (task def + service + ALB) | Full control over networking, sidecars, blue/green |
| **EKS** | You already run Kubernetes at scale (overkill otherwise) |
| **App Runner** | Only if you're an **existing** App Runner customer |

---

## Wire it together — IAM (the "managed identity + RBAC" equivalent)

Attach this policy to the **ECS task role** (`rag-task-role`) so the FastAPI app authenticates to everything by role, no keys in code:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Bedrock",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Converse", "bedrock:ConverseStream", "bedrock:Rerank"],
      "Resource": "*"
    },
    {
      "Sid": "BedrockKnowledgeBase",
      "Effect": "Allow",
      "Action": ["bedrock:Retrieve", "bedrock:RetrieveAndGenerate"],
      "Resource": "arn:aws:bedrock:us-east-1:<acct>:knowledge-base/<KB_ID>"
    },
    {
      "Sid": "OpenSearchServerlessData",
      "Effect": "Allow",
      "Action": ["aoss:APIAccessAll"],
      "Resource": "arn:aws:aoss:us-east-1:<acct>:collection/<id>"
    },
    {
      "Sid": "Secrets",
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": "arn:aws:secretsmanager:us-east-1:<acct>:secret:rag/app-*"
    }
  ]
}
```

Two extra notes that trip people up:
- **OpenSearch Serverless needs both** an IAM policy (`aoss:APIAccessAll`, above) **and** the collection's **data-access policy** (A.3) to list your task role as a principal. Miss either and you get 403s.
- If you use **inference-profile** model IDs, that's also where cross-region invoke permissions are scoped — keep `bedrock:InvokeModel` resource as `*` or include the profile ARNs.

That gives you the same end state as the Azure stack: documents in S3 → embedded with Titan → stored/retrieved in OpenSearch (or a managed Knowledge Base) → answered by Claude/Nova via Bedrock → served by a FastAPI app on ECS Express Mode, with secrets in Secrets Manager and telemetry in CloudWatch/X-Ray, all wired by IAM roles.
