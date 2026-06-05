from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql://postgres:2001@localhost:5432/OlistRisk"
)

with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM v_base_unificada"))
    print("Conexión OK — filas:", result.fetchone()[0])