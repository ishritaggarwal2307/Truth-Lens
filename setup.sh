#!/bin/bash

echo "=========================================="
echo "  Truth-Lens Setup Script"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "Creating project directories..."
mkdir -p data/raw/real data/raw/fake data/processed data/models logs temp .cache

# Create sample data structure file
cat > data/README.md << 'DATAREADME'
# Data Directory Structure

## Organization

```
data/
├── raw/
│   ├── real/          # Place authentic human speech samples here
│   └── fake/          # Place AI-generated speech samples here
├── processed/          # Preprocessed features (auto-generated)
└── models/            # Trained model files (auto-generated)
```

## Dataset Preparation

1. Download audio files (WAV or MP3 format)
2. Organize into `real/` and `fake/` folders
3. Ensure balanced dataset (similar number of files in each)

### Recommended Datasets

- **ASVspoof 2019 LA**: https://www.asvspoof.org/
- **Fake-or-Real (FoR)**: Available on Kaggle
- **Generate your own**: Use ElevenLabs, Resemble.ai, or other TTS services

## Quick Start

Minimum dataset size for hackathon demo:
- 100 real audio files
- 100 fake audio files
- Each file: 3-10 seconds duration
DATAREADME

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Add audio files to data/raw/real and data/raw/fake"
echo "  3. Train model: python src/train.py"
echo "  4. Start API: python src/api/app.py"
echo "  5. Open frontend: http://localhost:3000"
echo ""
