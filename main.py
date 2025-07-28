import os
import sys
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from app.models import get_tortoise_config
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from tortoise.contrib.fastapi import RegisterTortoise
from app.api import create_api
from app.services import (
    ContactFinderConfig,
    create_contact_finder,
    init_scheduler,
    start_scheduler,
    shutdown_scheduler,
)


# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage database and bounce service lifecycle"""
    db_url = os.getenv("DATABASE_URL", "sqlite://db.sqlite3")
    config = get_tortoise_config(db_url)

    # Initialize scheduler and register bounce callbacks
    scheduler = init_scheduler()
    from app.services.validators import EmailBounceValidator

    scheduler.register_callback("send_bounce_email", EmailBounceValidator.send_bounce_email)
    scheduler.register_callback("check_bounce_status", EmailBounceValidator.check_bounce_status)

    async with RegisterTortoise(
        app=app,
        config=config,
        generate_schemas=True,
        add_exception_handlers=True,
    ):
        # Start scheduler after DB is ready
        await start_scheduler()

        try:
            yield
        finally:
            # Graceful shutdown of scheduler
            await shutdown_scheduler()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""

    # Initialize contact finder
    config = ContactFinderConfig(
        serper_api_key=os.getenv("SERPER_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        email_verifier_key=os.getenv("EMAIL_VERIFIER_KEY"),
        enable_mx_validation=True,
        enable_domain_validation=True,
        enable_email_checker=True,
        enable_email_bounceback=True,
        max_emails=20,
    )

    finder = create_contact_finder(config)

    # Create API with finder
    app = create_api(finder)

    # Set lifespan
    app.router.lifespan_context = lifespan

    return app


def main():
    """Main application entry point"""

    # Check for environment file
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found!")
        print("   Copy .env.example to .env and add your API keys")
        print()

    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print("🚀 Starting ContactFinder API Server")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print()
    print("📚 API Documentation:")
    print(f"   http://localhost:{port}/docs")
    print(f"   http://localhost:{port}/redoc")
    print()

    try:
        uvicorn.run(
            "main:create_app",
            host=host,
            port=port,
            reload=debug,
            factory=True,
            log_level="info" if debug else "warning",
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        exit(1)


if __name__ == "__main__" and not sys.argv[0].endswith("spawn.py"):
    main()
