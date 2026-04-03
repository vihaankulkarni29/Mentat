#!/usr/bin/env python
"""
Mentat Live Launcher — Sets up environment and starts the live app
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║              Mentat Live — Intelligence Dashboard             ║")
    print("║                     Starting in 3 seconds...                  ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    if groq_key:
        print("Groq API Key: configured from environment")
    else:
        print("Groq API Key: not set (news sentiment will be skipped)")
    print("📍 Opening: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the app.")
    print()
    
    import time
    time.sleep(1)
    
    # Start Streamlit
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "live.py", "--logger.level=warning"],
        env=os.environ.copy()
    )

if __name__ == "__main__":
    main()
