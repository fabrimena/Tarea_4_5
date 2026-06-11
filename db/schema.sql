CREATE TABLE IF NOT EXISTS customers (
    id          SERIAL PRIMARY KEY,
    full_name   VARCHAR(120) NOT NULL,
    email       VARCHAR(160) NOT NULL,
    country     VARCHAR(60)  NOT NULL,
    risk_level  VARCHAR(20)  NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    created_at  TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS accounts (
    id             SERIAL PRIMARY KEY,
    customer_id    INTEGER      NOT NULL REFERENCES customers(id),
    account_number VARCHAR(20)  NOT NULL UNIQUE,
    balance        NUMERIC(14,2) NOT NULL DEFAULT 0,
    status         VARCHAR(20)  NOT NULL DEFAULT 'active',
    created_at     TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transactions (
    id              SERIAL PRIMARY KEY,
    account_id      INTEGER       NOT NULL REFERENCES accounts(id),
    amount          NUMERIC(14,2) NOT NULL,
    currency        VARCHAR(3)    NOT NULL DEFAULT 'USD',
    country         VARCHAR(60)   NOT NULL,
    status          VARCHAR(20)   NOT NULL CHECK (status IN ('approved', 'failed', 'pending')),
    is_flagged      BOOLEAN       NOT NULL DEFAULT FALSE,
    failure_reason  VARCHAR(200),
    created_at      TIMESTAMP     NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fraud_cases (
    id              SERIAL PRIMARY KEY,
    transaction_id  INTEGER      NOT NULL REFERENCES transactions(id),
    reason          TEXT         NOT NULL,
    severity        VARCHAR(20)  NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status          VARCHAR(20)  NOT NULL DEFAULT 'open',
    created_at      TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tool_access_logs (
    id            SERIAL PRIMARY KEY,
    tool_name     VARCHAR(80)  NOT NULL,
    justification TEXT         NOT NULL,
    request_data  JSONB,
    created_at    TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_transactions_flagged ON transactions(is_flagged);
CREATE INDEX IF NOT EXISTS idx_transactions_account_id ON transactions(account_id);
