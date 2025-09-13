#!/usr/bin/env python3

import asyncio
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.ui.gradio_interface import launch_ui
from config.settings import settings
import logging
from loguru import logger


def setup_logging():
    """Setup logging configuration."""
    # Remove default logger
    logger.remove()
    
    # Add console logging
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add file logging
    if settings.log_file:
        Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            settings.log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )


def check_environment():
    """Check if environment is properly configured."""
    errors = []
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        errors.append("❌ .env file not found. Please copy .env.template to .env and configure it.")
    
    # Check if API key is set
    try:
        if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
            errors.append("❌ GEMINI_API_KEY not set in .env file")
    except Exception as e:
        errors.append(f"❌ Error reading settings: {e}")
    
    # Check if data directories exist
    data_dir = Path("./data")
    if not data_dir.exists():
        print("📁 Creating data directory...")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path("./data/logs")
    if not logs_dir.exists():
        print("📁 Creating logs directory...")
        logs_dir.mkdir(parents=True, exist_ok=True)
    
    if errors:
        print("\n".join(errors))
        print("\nPlease fix the above issues and try again.")
        return False
    
    return True


def main():
    """Main entry point."""
    print("🚀 Starting TTT-Agent...")
    
    # Setup logging
    setup_logging()
    logger.info("TTT-Agent starting up")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    try:
        # Launch the UI
        logger.info(f"Launching interface on {settings.gradio_host}:{settings.gradio_port}")
        launch_ui()
        
    except KeyboardInterrupt:
        logger.info("Shutting down TTT-Agent...")
        print("\n👋 TTT-Agent shutdown complete")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
