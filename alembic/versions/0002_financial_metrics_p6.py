"""Phase 6: financial_metrics_p6 and metric_aliases tables

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-07
"""
from alembic import op
import sqlalchemy as sa

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'financial_metrics_p6',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('company_id', sa.Integer(), sa.ForeignKey('companies.id'), nullable=False),
        sa.Column('fiscal_year', sa.Integer(), nullable=False),
        sa.Column('fiscal_period', sa.String(5), nullable=False),
        sa.Column('period_end_date', sa.Date(), nullable=True),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('value', sa.Numeric(20, 6), nullable=False),
        sa.Column('unit', sa.String(20), nullable=False),
        sa.Column('scale', sa.String(20), nullable=False, server_default='UNITS'),
        sa.Column('dimension_type', sa.String(20), nullable=True),
        sa.Column('dimension_value', sa.String(100), nullable=True),
        sa.Column('is_derived', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('document_id', sa.Integer(), sa.ForeignKey('documents.id'), nullable=True),
        sa.Column('document_version_id', sa.Integer(), sa.ForeignKey('document_versions.id'), nullable=True),
        sa.Column('section_id', sa.Integer(), sa.ForeignKey('sections.id'), nullable=True),
        sa.Column('chunk_id', sa.Integer(), sa.ForeignKey('chunks.id'), nullable=True),
        sa.Column('page', sa.Integer(), nullable=True),
        sa.Column('extraction_method', sa.String(20), nullable=False),
        sa.Column('source_text', sa.Text(), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_fin_metric_p6_company', 'financial_metrics_p6', ['company_id'])
    op.create_index('idx_fin_metric_p6_period', 'financial_metrics_p6',
                    ['company_id', 'fiscal_year', 'fiscal_period'])
    # Unique index for rows without dimension (NULL dimension_type/value)
    op.create_index(
        'uq_fin_metric_p6_nodim',
        'financial_metrics_p6',
        ['company_id', 'fiscal_year', 'fiscal_period', 'metric_name', 'document_version_id'],
        unique=True,
        postgresql_where=sa.text("dimension_type IS NULL AND dimension_value IS NULL"),
    )
    # Unique index for rows with dimension
    op.create_index(
        'uq_fin_metric_p6_dim',
        'financial_metrics_p6',
        ['company_id', 'fiscal_year', 'fiscal_period', 'metric_name',
         'dimension_type', 'dimension_value', 'document_version_id'],
        unique=True,
        postgresql_where=sa.text("dimension_type IS NOT NULL"),
    )

    op.create_table(
        'metric_aliases',
        sa.Column('alias', sa.String(200), primary_key=True),
        sa.Column('canonical_name', sa.String(100), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('metric_aliases')
    op.drop_index('uq_fin_metric_p6_dim', table_name='financial_metrics_p6')
    op.drop_index('uq_fin_metric_p6_nodim', table_name='financial_metrics_p6')
    op.drop_index('idx_fin_metric_p6_period', table_name='financial_metrics_p6')
    op.drop_index('idx_fin_metric_p6_company', table_name='financial_metrics_p6')
    op.drop_table('financial_metrics_p6')
