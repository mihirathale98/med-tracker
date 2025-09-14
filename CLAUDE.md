# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical lab tracking system called "med-tracker" that provides a Model Context Protocol (MCP) server for ingesting, normalizing, and analyzing medical lab reports. The system can process CSV files, PDFs, and images (via OCR) to extract lab data and store it in a local SQLite database.

## Key Components

- **labs_mcp.py**: Main MCP server implementing lab data ingestion, normalization, and analysis
- **main.py**: Simple entry point (currently just prints hello message)
- **exploration.ipynb**: Jupyter notebook for data exploration and analysis
- **labs_specimen.csv**: Sample lab data extracted from MIMIC-IV dataset
- **data/**: Directory for storing input files and SQLite database
- **exports/**: Directory for generated reports and plots

## Architecture

The system follows a clear data flow:
1. **Ingestion**: Parse lab reports from various formats (CSV/PDF/images)
2. **Normalization**: Map diverse column names to canonical schema using ALIASES mapping
3. **Storage**: Store in SQLite database with canonical columns (subject_id, charttime, valuenum, etc.)
4. **Analysis**: Generate trends, summaries, and longitudinal comparisons
5. **Export**: Create time-series plots and Markdown reports

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv add "mcp[cli]" pandas numpy matplotlib pdfplumber pytesseract pillow
```

### Running the MCP Server
```bash
# Test locally
uv run labs_mcp.py

# For Claude Desktop integration, add to claude_desktop_config.json:
# {
#   "mcpServers": {
#     "medlabs": {
#       "command": "uv",
#       "args": ["--directory", "/absolute/path/to/project", "run", "labs_mcp.py"]
#     }
#   }
# }
```

### Key Data Schema

The system normalizes all lab data to this canonical schema:
- `subject_id`: Patient identifier
- `charttime`: Collection timestamp (ISO8601)
- `label`: Test name (e.g., "Hemoglobin", "Glucose")
- `valuenum`: Numeric result value
- `valueuom`: Units of measurement
- `ref_low`/`ref_high`: Reference range bounds
- `flag`: Abnormal indicator

### Database Location
- Default: `./data/medlabs.sqlite`
- Override with `MEDLABS_DB` environment variable

### Important Notes
- MCP server uses stdio transport - avoid print statements, use stderr for logging
- OCR quality for images may vary - adjust parsers as needed
- System is for informational purposes only, not medical advice
- Reference range parsing handles formats like "13-17" or separate low/high columns