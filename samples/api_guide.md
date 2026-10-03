# Acme Search API Guide

## Authentication
All requests must include an API key in the `X-API-Key` header. Keys are created in the dashboard.

## Rate Limits
The API allows 100 requests per minute per key. When the limit is exceeded the server returns HTTP 429 and a `Retry-After` header.

## Endpoints
### POST /v1/search
Runs a search query.

```python
import requests

resp = requests.post(
    "https://api.acme.example/v1/search",
    headers={"X-API-Key": "YOUR_KEY"},
    json={"query": "red shoes", "limit": 10},
    timeout=10,
)
print(resp.json())
```

### GET /v1/documents/{id}
Returns a single document by its identifier.

## Configuration
Set `ACME_TIMEOUT` to change the default request timeout. The default is 30 seconds.
