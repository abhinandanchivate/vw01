# Azure RAG stack — Portal (GUI) steps

*Portal click-through equivalents of the Azure CLI in §6.3–6.7, current to June 2026.*

## What changed since the CLI doc was written (read first)

- **Azure OpenAI is now "Microsoft Foundry."** The model portal lives at **ai.azure.com** (also reachable as foundry.ai.azure.com) and has a **New Foundry / Foundry (classic)** toggle. Most flows below assume **New Foundry is ON**.
- **Two resource flavors.** You can still create a plain **Azure OpenAI** resource (classic), but the recommended path is a **Foundry resource**, which unlocks the full multi-provider catalog (OpenAI, Anthropic, Meta, Mistral, xAI, etc.) rather than only OpenAI deployments.
- **Models moved on.** `gpt-4o` still deploys and works, but it's several generations old. The current OpenAI line on Foundry runs through **gpt-5.5** (frontier, GA, 1M-token context) with **gpt-5 / 5.1 / 5.2** GA and **gpt-5.3** in preview, plus cheaper variants like **gpt-5-mini** and **gpt-5.4-mini**. There's also a new **Model Router** deployment that auto-picks a model per request.
- **Embeddings are unchanged.** `text-embedding-3-large` (3072-dim) is still the newest/most capable embedding model, so that part of the doc is still correct. The index vector dimension must match whatever you pick.
- **AI Search added a compute-type choice** (Default vs Confidential) at create time, and the semantic-ranker switch now lives under **Settings → Premium features**.

---

## 6.3 Azure AI Search

**Create the service**

1. Azure portal → **Create a resource** → search **Azure AI Search** → **Create**.
2. **Basics** tab:
   - **Subscription** / **Resource group** — match the rest of the stack.
   - **Service name** — becomes the URL `https://<name>.search.windows.net`.
   - **Region** — co-locate with your Foundry resource to avoid egress charges and meet integrated-vectorization region rules.
   - **Pricing tier** — click *Change Pricing Tier*. Pick **Basic** or **Standard (S1)** to start. **Semantic ranker requires Basic or higher — it is not available on Free.** (Tiers map to the CLI `--sku`: `free | basic | standard | standard2 | standard3 | storage_optimized_l1 | storage_optimized_l2`.)
   - **Compute type** *(2026 addition)* — **Default** (standard infra) unless you specifically need **Confidential** compute.
3. *(Optional)* **Networking** tab — leave Public for dev; choose Private endpoint for locked-down prod.
4. **Review + create** → **Create**.

**Scale (the CLI `--partition-count` / `--replica-count`)**

5. Open the service → **Settings → Scale**. **Partitions** = storage + indexing throughput; **Replicas** = query throughput (QPS) and availability (3+ replicas → 99.9% read SLA). Start at 1/1 and raise as load grows.

**Enable the semantic ranker (CLI's "semantic ranker On")**

6. In the service → **Settings → Premium features**. Under **Semantic ranker**, pick a plan:
   - **Free** — monthly request allowance, fine for dev/test.
   - **Standard** — per-query billing after the free allowance, for production.
   - *Note:* the portal toggle still also governs agentic-retrieval billing consent (the portal uses an older API version where the two are coupled), even though the newer management API separates them.

**Switch to Entra ID / RBAC auth (CLI `--auth-options aadOrApiKey`)**

7. In the service → **Settings → Keys**. Set **API access control** to **Role-based access control** (managed-identity only) or **Both** (keys + RBAC). "Both" is the equivalent of `aadOrApiKey`.
8. Then **Access control (IAM) → Add role assignment** and grant your app's identity the data-plane roles it needs: **Search Index Data Reader** (query), plus **Search Index Data Contributor** and **Search Service Contributor** if the app also builds/updates the index.

---

## 6.4 Microsoft Foundry resource + model deployments

This replaces `az cognitiveservices account create --kind OpenAI` and the two `deployment create` calls.

**Create the resource**

Either path works; the Foundry-portal path is recommended in 2026.

- **Via Azure portal (closest to the CLI):** **Create a resource** → search **Azure AI Foundry** (or **Azure OpenAI** for the classic resource) → **Create** → set Subscription, Resource group, Region, Name (this is your custom domain), pricing tier **S0** → **Review + create**.
- **Via Foundry portal:** go to **ai.azure.com**, sign in, and **Create a project** — this provisions the underlying Foundry resource for you.

**Deploy the chat model**

1. Sign in to **ai.azure.com**. Confirm the **New Foundry** toggle is **on**.
2. From the homepage, select **Discover** (upper-right) → **Models** (left pane). (Or open **Model catalog**.)
3. Search and select your chat model. Choices, mapped to the CLI:

   | Need | 2026 pick (deployment name is yours to choose) |
   |------|-----------------------------------------------|
   | Balanced, GA, safe for prod | **gpt-5.1** or **gpt-5.2** |
   | Cheapest | **gpt-5-mini** or **gpt-5.4-mini** |
   | Frontier / hardest reasoning | **gpt-5.5** (or **gpt-5.5 Pro**) |
   | Don't want to choose per request | **Model Router** (auto-selects) |
   | Legacy parity with the old doc | **gpt-4o** (still deployable) |

   *Partner/community models (e.g., Anthropic, Cohere, Llama) require an Azure Marketplace subscription step first; OpenAI models "sold by Azure" do not. Some gpt-5.x tiers need a quota request unless you're on Tier 5/6.*
4. Select **Use this model** / **Deploy**.
5. Set the **Deployment name** (your code references *this*, not the model name), **Deployment type** = **Global Standard** (broadest region availability; equals CLI `--sku-name GlobalStandard`), and **Capacity (TPM)** sized to expected load. Other deployment types available: Standard, DataZone Standard, Provisioned Managed, Global Batch.
6. **Deploy.** You land in the Playground to test.

**Deploy the embedding model**

7. Repeat steps 2–6 with **text-embedding-3-large**. **The index's vector dimension must equal the model's output (3072 for 3-large; 1536 for 3-small / ada-002).** Give it a sensible capacity (embeddings are cheap, so a higher TPM is fine).

*Alternative, as in the original doc:* instead of a custom pipeline you can use Foundry's built-in retrieval (now surfaced as **Foundry IQ / agentic retrieval / knowledge sources**) to wire Search to a chat model with managed chunking and citations. The explicit pipeline below keeps full control over chunking, prompts, and citations.

---

## 6.5 Key Vault

1. **Create a resource** → **Key Vault** → **Create**.
2. **Basics**: Resource group, **Vault name**, Region, **Pricing tier** = **Standard** (or **Premium** for HSM-backed keys).
3. **Access configuration** tab: set **Permission model** to **Azure role-based access control** (recommended — equals CLI `--enable-rbac-authorization true`) rather than **Vault access policy**.
4. *(Optional)* **Networking** → Private endpoint for prod.
5. **Review + create** → **Create**.
6. Grant access with RBAC: **Access control (IAM) → Add role assignment → Key Vault Secrets User** to your app's managed identity. With managed identity you should need very few stored secrets (e.g., a SharePoint app secret).

---

## 6.6 Application Insights

1. **Create a resource** → **Application Insights** → **Create**.
2. **Basics**: Resource group, **Name**, Region.
   - **Resource Mode** is **Workspace-based** only (classic/standalone was retired). Select an existing **Log Analytics workspace** or let it create one.
3. **Review + create** → **Create**.
4. Connect it to the API later (see App Service **Monitoring** tab, or **Settings → Application Insights** on the web app) to capture request latency, dependency calls to Search and OpenAI, exceptions, and custom events.

---

## 6.7 Serving compute (the FastAPI service)

Pick one. App Service is the straightforward choice; Container Apps suits containers / scale-to-zero.

### App Service (recommended)

1. **Create a resource** → **Web App** → **Create**.
2. **Basics**:
   - Resource group, **Name**.
   - **Publish** = **Code**.
   - **Runtime stack** = **Python 3.12**.
   - **Operating System** = **Linux**.
   - **Region**.
   - **App Service Plan** → create one and pick **SKU/size**: a **B-tier** (B1/B2/B3) for dev, **P-tier** (P0v3–P3v3) for prod / VNet / autoscale. (This is `az appservice plan create --sku`.)
3. **Monitoring** tab → enable **Application Insights** and point it at the component from §6.6.
4. *(Optional)* **Networking** tab → VNet integration (P-tier) if Search/OpenAI are private.
5. **Review + create** → **Create**.
6. **Set the startup command** (the gunicorn/uvicorn line): open the web app → **Settings → Configuration → General settings → Startup Command**:
   ```
   gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app
   ```
   Save (this restarts the app).
7. **Turn on managed identity**: **Settings → Identity → System assigned → Status: On → Save**. Then go assign this identity the roles listed in "Wire it together" below.

### Azure Container Apps (alternative)

1. **Create a resource** → **Container App** → **Create**.
2. **Basics**: Resource group, **Container app name**, Region, and create a **Container Apps Environment**.
3. **Container** tab: choose your image (or build from the `api/` source), set **Ingress** = **External** and **Target port** = **8000**. Provide a Dockerfile in `api/`, or let the cloud builder detect Python. (This is the GUI form of `az containerapp up`.)
4. **Review + create** → **Create**.
5. Enable a **managed identity** on the container app the same way (Identity blade) and assign the roles below. KEDA-based scale-to-zero is configured on the app's **Scale** settings.

---

## Wire it together (managed-identity RBAC)

The CLI doc leans on managed identity throughout. After the serving app has a system-assigned identity, grant it (via **Access control (IAM) → Add role assignment** on each target resource):

- **On the AI Search service:** *Search Index Data Reader* (+ *Search Index Data Contributor* / *Search Service Contributor* if it also writes the index).
- **On the Foundry / Azure OpenAI resource:** *Cognitive Services OpenAI User*.
- **On the Key Vault:** *Key Vault Secrets User*.

With those in place, the FastAPI app authenticates to Search, the chat/embedding deployments, and the vault using its identity — no keys in code.
