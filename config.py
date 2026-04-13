"""Variables de configuracion centralizadas (carga .env)."""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

# Cargar .env desde la raiz del proyecto
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path)

logger = logging.getLogger(__name__)


def _require(name: str) -> str:
    """Obtener variable de entorno requerida o lanzar error descriptivo.

    Args:
        name: Nombre de la variable de entorno.

    Returns:
        Valor de la variable de entorno.

    Raises:
        EnvironmentError: Si la variable no esta definida o esta vacia.
    """
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(
            f"Variable de entorno requerida no encontrada: '{name}'. "
            f"Revisa tu archivo .env (ejemplo en .env.example)."
        )
    return value


# ── Variables de conexion a PostgreSQL ────────────────────────────────────────
DB_HOST: str = _require("DB_HOST")
DB_PORT: str = os.getenv("DB_PORT", "5432").strip()
DB_NAME: str = _require("DB_NAME")
DB_USER: str = _require("DB_USER")
DB_PASSWORD: str = _require("DB_PASSWORD")

# String de conexion SQLAlchemy completo
DB_URL: str = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── Variables opcionales de APIs externas ─────────────────────────────────────
CEN_API_KEY: str = os.getenv("CEN_API_KEY", "")


if __name__ == "__main__":
    print("Conexion configurada correctamente")
    print(f"  Host     : {DB_HOST}:{DB_PORT}")
    print(f"  Base de datos : {DB_NAME}")
    print(f"  Usuario  : {DB_USER}")
    print(f"  DB_URL   : {DB_URL.replace(DB_PASSWORD, '***')}")
