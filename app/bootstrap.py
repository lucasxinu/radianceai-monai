from app.settings import settings

def bootstrap():
    print(f"Starting {settings.app_name} in {settings.environment}")