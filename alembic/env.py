"""
alembic/env.py  ─  Alembic migration environment for Phase 4
═══════════════════════════════════════════════════════════════
Configured to:
  1. Read DATABASE_URL from .env (not from alembic.ini — keeps credentials secure)
  2. Import src.db.models.Base for autogenerate support
  3. Work with pgvector's vector type
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool, text

from alembic import context

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so src.db.models can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")

# ── Alembic config ────────────────────────────────────────────────────────────
config = context.config

# Set the sqlalchemy.url from DATABASE_URL env var (not from alembic.ini)
# This keeps credentials out of alembic.ini which may be committed to git.
database_url = os.getenv("DATABASE_URL", "")
if not database_url:
    raise EnvironmentError(
        "DATABASE_URL is not set. Add it to .env before running migrations."
    )
config.set_main_option("sqlalchemy.url", database_url)

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Import models for autogenerate ────────────────────────────────────────────
# pgvector must be registered BEFORE metadata is used
import pgvector.sqlalchemy  # noqa: F401 — registers the 'vector' type

from src.db.models import Base  # noqa: E402

target_metadata = Base.metadata


# ── Migration functions ───────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode (generate SQL script without connecting).
    Useful for reviewing migrations before applying them.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode (apply directly to the database).
    This is the normal mode when running `alembic upgrade head`.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # NullPool: no connection reuse in migrations
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,          # detect column type changes
            compare_server_default=True, # detect server_default changes
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
