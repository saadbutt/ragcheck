# Security and access

## Authentication

Tokens are issued per service account and do not expire automatically. Rotate
them at least every 90 days. A revoked token stops working within 60 seconds.

## Roles

There are four roles. Owner can do anything including deleting the workspace.
Admin can manage members and billing but cannot delete the workspace. Member
can read and write project data. Viewer has read access only.

## Data residency

Workspaces are pinned to a region at creation and the region cannot be changed
afterwards. Available regions are eu-west-1, us-east-1 and ap-southeast-2.

## Encryption

Data is encrypted at rest with AES-256 and in transit with TLS 1.3. Customer
managed keys are available on the Scale plan only.

## Audit log

Every write action is recorded with the actor, the timestamp and the previous
value. Audit entries are immutable and are retained for 2 years.
