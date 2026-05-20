# ----- central configuration management @ backend/utils/config.py -----

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "swiggy-dineout-voice-agent"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database
    database_url: str = "sqlite:///./dineout_voice.db"

    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # Groq (LLM)
    groq_api_key: str = ""

    # Sarvam (TTS)
    sarvam_api_key: str = ""

    # Swiggy OAuth
    swiggy_client_id: str = ""
    swiggy_client_secret: str = ""
    swiggy_oauth_callback_url: str = ""

    # ML Models
    classifier_model_path: str = "ml/models/classifier-int8.onnx"
    moonshine_model_base: str = "ml/models/moonshine-tiny-{lang}-int8.onnx"

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"


settings = Settings()
