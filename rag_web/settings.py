from pathlib import Path
import os

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

_database_url = os.getenv("DATABASE_URL", "").strip()
if not _database_url or _database_url.lower().startswith("sqlite"):
    raise ImproperlyConfigured(
        "Set DATABASE_URL to MySQL in .env, e.g. "
        "mysql+pymysql://USER:PASSWORD@127.0.0.1:3310/DATABASE_NAME "
        "(SQLite is not used)."
    )

_db = make_url(_database_url)
if "mysql" not in (_db.drivername or ""):
    raise ImproperlyConfigured(
        "DATABASE_URL must use a MySQL driver (e.g. mysql+pymysql://...)."
    )

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-secret-key-change-in-prod")
DEBUG = os.getenv("DJANGO_DEBUG", "true").lower() == "true"
ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "chat_ui",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "rag_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "rag_web.wsgi.application"
ASGI_APPLICATION = "rag_web.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": _db.database or "",
        "USER": _db.username or "",
        "PASSWORD": _db.password or "",
        "HOST": _db.host or "127.0.0.1",
        "PORT": str(_db.port or 3306),
        "OPTIONS": {
            "charset": "utf8mb4",
            "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
        },
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
