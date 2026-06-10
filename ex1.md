
```bash
RG_NAME=rg_sb_southeastasia_320672_3_178107956268
LOCATION=southeastasia

PG_SERVER=pgserver$RANDOM
PG_DB=appdb
PG_USER=pgadminuser
PG_PASSWORD='Postgres@12345'

STORAGE_NAME=funcpgstore$RANDOM
FUNC_APP=func-pg-crud-$RANDOM
```

Do **not** use this line:

```bash
RG_NAME=rg_func_postgres_crud
LOCATION=eastus
```

Because your resource group is in **southeastasia**, keep all resources in `southeastasia`.

---

## 1. Verify resource group

```bash
az group show \
  --name "$RG_NAME" \
  --query "{name:name, location:location}" \
  --output table
```

---

## 2. Create PostgreSQL Flexible Server

Use public access for now, because it is easier for Azure Function testing. Azure PostgreSQL Flexible Server supports public access with firewall rules, and Azure CLI supports testing/executing SQL using `az postgres flexible-server connect/execute`. ([Microsoft Learn][1])

```bash
az postgres flexible-server create \
  --resource-group "$RG_NAME" \
  --name "$PG_SERVER" \
  --location "$LOCATION" \
  --admin-user "$PG_USER" \
  --admin-password "$PG_PASSWORD" \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --storage-size 32 \
  --version 16 \
  --public-access 0.0.0.0 \
  --yes
```

Check server:

```bash
az postgres flexible-server show \
  --resource-group "$RG_NAME" \
  --name "$PG_SERVER" \
  --query "{name:name,fqdn:fullyQualifiedDomainName,state:state,publicAccess:network.publicNetworkAccess}" \
  --output table
```

Set host:

```bash
PG_HOST=$(az postgres flexible-server show \
  --resource-group "$RG_NAME" \
  --name "$PG_SERVER" \
  --query fullyQualifiedDomainName \
  --output tsv)

echo $PG_HOST
```

---

## 3. Add firewall rule for your current IP

```bash
MY_IP=$(curl -s https://ifconfig.me)

az postgres flexible-server firewall-rule create \
  --resource-group "$RG_NAME" \
  --name "$PG_SERVER" \
  --rule-name AllowMyIP \
  --start-ip-address "$MY_IP" \
  --end-ip-address "$MY_IP"
```

For Azure Function outbound testing, temporarily allow Azure services:

```bash
az postgres flexible-server firewall-rule create \
  --resource-group "$RG_NAME" \
  --name "$PG_SERVER" \
  --rule-name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0
```

---

## 4. Create database manually

Do **not** use `--database-name` with your earlier command. Create the DB separately:

```bash
az postgres flexible-server db create \
  --resource-group "$RG_NAME" \
  --server-name "$PG_SERVER" \
  --database-name "$PG_DB"
```

Check DB:

```bash
az postgres flexible-server db list \
  --resource-group "$RG_NAME" \
  --server-name "$PG_SERVER" \
  --output table
```

---

## 5. Test PostgreSQL connection from terminal

```bash
psql "host=$PG_HOST port=5432 dbname=$PG_DB user=$PG_USER password=$PG_PASSWORD sslmode=require"
```

Inside `psql`, run:

```sql
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL
);

INSERT INTO users(name, email)
VALUES ('Abhi', 'abhi@test.com');

SELECT * FROM users;
```

Exit:

```sql
\q
```

---

## 6. Create Azure Function project

Azure Functions Core Tools supports creating a Python function project with `func init`, and publishing with `func azure functionapp publish`. ([Microsoft Learn][2])

```bash
mkdir ServerlessAPI_Postgres
cd ServerlessAPI_Postgres

func init . --worker-runtime python
```

Create files:

```bash
touch function_app.py requirements.txt
```

---

## 7. Add dependencies

```bash
cat > requirements.txt <<'EOF'
azure-functions
flask
psycopg2-binary
EOF
```

---

## 8. Add Flask CRUD code inside Azure Function

Microsoft has an official Flask-on-Azure-Functions sample pattern, and Azure Functions Python supports WSGI/ASGI integration for Python web frameworks. ([Microsoft Learn][3])

```bash
cat > function_app.py <<'EOF'
import os
import json
import psycopg2
from flask import Flask, request, jsonify
import azure.functions as func

flask_app = Flask(__name__)

def get_conn():
    return psycopg2.connect(
        host=os.environ["PG_HOST"],
        database=os.environ["PG_DB"],
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
        port=5432,
        sslmode="require"
    )

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(150) UNIQUE NOT NULL
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

@flask_app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@flask_app.route("/api/init", methods=["POST"])
def create_table():
    init_db()
    return jsonify({"message": "users table created successfully"})

@flask_app.route("/api/users", methods=["POST"])
def create_user():
    data = request.get_json()

    name = data.get("name")
    email = data.get("email")

    if not name or not email:
        return jsonify({"error": "name and email are required"}), 400

    conn = get_conn()
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO users(name, email) VALUES (%s, %s) RETURNING id;",
            (name, email)
        )
        user_id = cur.fetchone()[0]
        conn.commit()
        return jsonify({"id": user_id, "name": name, "email": email}), 201
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

@flask_app.route("/api/users", methods=["GET"])
def get_users():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id, name, email FROM users ORDER BY id;")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    users = [
        {"id": row[0], "name": row[1], "email": row[2]}
        for row in rows
    ]

    return jsonify(users)

@flask_app.route("/api/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id, name, email FROM users WHERE id = %s;", (user_id,))
    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        return jsonify({"error": "user not found"}), 404

    return jsonify({"id": row[0], "name": row[1], "email": row[2]})

@flask_app.route("/api/users/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    data = request.get_json()

    name = data.get("name")
    email = data.get("email")

    if not name or not email:
        return jsonify({"error": "name and email are required"}), 400

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "UPDATE users SET name = %s, email = %s WHERE id = %s RETURNING id;",
        (name, email, user_id)
    )

    updated = cur.fetchone()
    conn.commit()

    cur.close()
    conn.close()

    if not updated:
        return jsonify({"error": "user not found"}), 404

    return jsonify({"id": user_id, "name": name, "email": email})

@flask_app.route("/api/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("DELETE FROM users WHERE id = %s RETURNING id;", (user_id,))
    deleted = cur.fetchone()

    conn.commit()

    cur.close()
    conn.close()

    if not deleted:
        return jsonify({"error": "user not found"}), 404

    return jsonify({"message": "user deleted successfully"})

app = func.WsgiFunctionApp(app=flask_app.wsgi_app, http_auth_level=func.AuthLevel.ANONYMOUS)
EOF
```

---

## 9. Run locally

Create local settings:

```bash
cat > local.settings.json <<EOF
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "PG_HOST": "$PG_HOST",
    "PG_DB": "$PG_DB",
    "PG_USER": "$PG_USER",
    "PG_PASSWORD": "$PG_PASSWORD"
  }
}
EOF
```

Start locally:

```bash
func start
```

Test:

```bash
curl http://localhost:7071/api/health
```

Create table:

```bash
curl -X POST http://localhost:7071/api/init
```

Create user:

```bash
curl -X POST http://localhost:7071/api/users \
  -H "Content-Type: application/json" \
  -d '{"name":"Abhi","email":"abhi@example.com"}'
```

Get users:

```bash
curl http://localhost:7071/api/users
```

---

## 10. Create Storage Account for Function App

```bash
az storage account create \
  --name "$STORAGE_NAME" \
  --resource-group "$RG_NAME" \
  --location "$LOCATION" \
  --sku Standard_LRS
```

---

## 11. Create Function App

```bash
az functionapp create \
  --resource-group "$RG_NAME" \
  --consumption-plan-location "$LOCATION" \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --name "$FUNC_APP" \
  --storage-account "$STORAGE_NAME" \
  --os-type Linux
```

---

## 12. Add PostgreSQL environment variables to Function App

```bash
az functionapp config appsettings set \
  --resource-group "$RG_NAME" \
  --name "$FUNC_APP" \
  --settings \
  PG_HOST="$PG_HOST" \
  PG_DB="$PG_DB" \
  PG_USER="$PG_USER" \
  PG_PASSWORD="$PG_PASSWORD"
```

---

## 13. Deploy Function

```bash
func azure functionapp publish "$FUNC_APP"
```

---

## 14. Test deployed API

Set URL:

```bash
APP_URL="https://$FUNC_APP.azurewebsites.net"
```

Health:

```bash
curl "$APP_URL/api/health"
```

Create table:

```bash
curl -X POST "$APP_URL/api/init"
```

Create user:

```bash
curl -X POST "$APP_URL/api/users" \
  -H "Content-Type: application/json" \
  -d '{"name":"Abhi","email":"abhi@azure.com"}'
```

Get users:

```bash
curl "$APP_URL/api/users"
```

Get one user:

```bash
curl "$APP_URL/api/users/1"
```

Update user:

```bash
curl -X PUT "$APP_URL/api/users/1" \
  -H "Content-Type: application/json" \
  -d '{"name":"Abhinandan","email":"abhinandan@azure.com"}'
```

Delete user:

```bash
curl -X DELETE "$APP_URL/api/users/1"
```

---

## Important correction for your variable block

Use this final block:

```bash
RG_NAME=rg_sb_southeastasia_320672_3_178107956268
LOCATION=southeastasia

PG_SERVER=pgserver$RANDOM
PG_DB=appdb
PG_USER=pgadminuser
PG_PASSWORD='Postgres@12345'

STORAGE_NAME=funcpgstore$RANDOM
FUNC_APP=func-pg-crud-$RANDOM
```

Do not mix:

```bash
rg_sb_southeastasia_320672_3_178107956268
```

with:

```bash
RG_NAME=rg_func_postgres_crud
LOCATION=eastus
```

