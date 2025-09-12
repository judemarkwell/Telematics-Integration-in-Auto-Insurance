"""
Database setup script for the Telematics Insurance System.

This script sets up the database with tables and sample data.
Run this script once to initialize your database.
"""

import sys
import os
import asyncio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.db.init_db import main

if __name__ == "__main__":
    asyncio.run(main())
