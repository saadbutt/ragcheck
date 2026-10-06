# Troubleshooting

## 401 responses

A 401 means the token was not accepted. Check that the header reads
"Authorization: Bearer" followed by the token, and that the token belongs to
the same region as the workspace. Tokens are not portable across regions.

## 429 responses

A 429 means the rate limit was exceeded. Back off for the number of seconds in
the Retry-After header. Retrying immediately extends the limit window rather
than resetting it.

## Duplicate records

Duplicates almost always mean the Idempotency-Key header was missing or was
regenerated on retry. Generate the key once, before the first attempt, and
reuse it for every retry of that same request.

## Missing webhooks

Check the delivery log before anything else. A webhook that shows as delivered
with a non-2xx response is a problem at your endpoint, not ours. Failed
deliveries are retried six times over 24 hours with exponential backoff.

## Slow queries

A list query that is slow is usually missing a filter and is scanning the full
collection. Add a date range. The default range when none is supplied is the
last 90 days.
