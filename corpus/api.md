# API reference notes

The base URL is https://api.halden.example/v2. Every request needs an
Authorization header carrying a bearer token.

## Rate limits

Starter accounts are limited to 10 requests per second. Growth accounts are
limited to 60 requests per second. Scale accounts are negotiated. Exceeding
the limit returns HTTP 429 with a Retry-After header in seconds.

## Pagination

List endpoints return at most 100 items. Use the cursor field from the
response to request the next page. Cursors expire after 30 minutes.

## Idempotency

Write requests accept an Idempotency-Key header. A repeated key within 24
hours returns the original response rather than creating a duplicate record.
Keys are scoped per endpoint, not globally.

## Webhooks

Webhook payloads are signed with HMAC SHA-256. The signature is in the
X-Halden-Signature header. Reject any payload whose timestamp is more than
five minutes old, which is what stops a captured request being replayed.
