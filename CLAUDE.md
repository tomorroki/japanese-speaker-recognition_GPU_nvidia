# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Running the Application
```bash
# Start the main Streamlit web application
streamlit run app.py

# Test the core recognition system
python enhanced_speaker_recognition.py

# Create background embeddings for improved accuracy
python create_embeddings.py --jvs-path [JVS_PATH] --cv-max-samples 5000
```

### GPU Setup (CUDA 12.6)
```bash
# Install GPU-accelerated PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Then install remaining dependencies
pip install -r requirements.txt
```

### Background Data Processing
```bash
# Create JVS embeddings from downloaded dataset
python prep_jvs_embeddings.py [JVS_PATH] --per-speaker 50

# Create Common Voice embeddings (requires Hugging Face auth)
python prep_common_voice_embeddings.py --max-samples 5000

# Create dummy embeddings for testing
python create_dummy_cv_embeddings.py
```

### Testing and Validation
```bash
# Test dataset manager functionality
python dataset_manager.py

# Test background loader
python background_loader.py
```

## Architecture Overview

This is a Japanese speaker recognition system built with SpeechBrain's ECAPA-TDNN model. The architecture follows a modular design with clear separation of concerns:

### Core Components

**Main Application (`app.py`)**
- Streamlit web interface with session state management
- Multi-tab UI: Speaker Recognition, Speaker Management, Statistics
- Real-time model initialization and caching controls
- Configuration management through UI checkboxes

**Recognition Engine (`enhanced_speaker_recognition.py`)**
- Primary `JapaneseSpeakerRecognizer` class
- Handles audio preprocessing, embedding extraction, and similarity scoring
- Implements AS-Norm (Adaptive Score Normalization) using background models
- Manages speaker database with caching system (`enrolled_speakers_embeddings.npz`)
- Supports both local speakers and dataset speakers (JVS/Common Voice)

**Dataset Management (`dataset_manager.py`)**
- `DatasetManager` class handles speaker classification and filtering
- Distinguishes between JVS speakers (jvs001-jvs100), Common Voice speakers, and custom speakers
- Configurable exclusion rules for background speakers
- Audio file discovery and validation

**Background Processing (`background_loader.py`)**
- `BackgroundEmbeddingLoader` manages pre-computed embeddings for score normalization
- Loads JVS and Common Voice embeddings from `.npz` files
- Provides combined background embeddings for AS-Norm
- Automatic fallback to audio file processing if embeddings unavailable

### Data Flow

1. **Initialization**: Load SpeechBrain model, build speaker database from `enroll/` folder
2. **Background Model**: Load or compute background embeddings for score normalization
3. **Recognition**: Extract embedding from input audio, compute cosine similarity with enrolled speakers
4. **Scoring**: Apply AS-Norm using background statistics, return top matches with confidence scores

### Configuration System

Central configuration in `config.json` controls:
- Model settings (device, SpeechBrain model)
- Audio processing (sample rate, duration limits)
- Recognition thresholds and normalization
- Dataset inclusion/exclusion rules
- UI display preferences
- Caching behavior

### File Structure

```
enroll/                          # Speaker enrollment directory
├── [speaker_name]/              # Individual speaker folders
│   └── *.wav                    # Audio samples for each speaker

background_datasets/             # Optional: raw audio for background model
├── jvs/                        # JVS corpus audio files
└── common_voice_ja/            # Common Voice audio files

background_jvs_ecapa.npz         # Pre-computed JVS embeddings (3.8MB)
background_common_voice_ja_ecapa.npz  # Pre-computed Common Voice embeddings (19MB)
enrolled_speakers_embeddings.npz # Cached speaker embeddings
```

### Key Design Patterns

**Caching Strategy**: The system uses aggressive caching for expensive operations:
- Speaker embeddings are cached in `.npz` files with timestamp validation
- Background embeddings are pre-computed to avoid dataset downloads
- Cache invalidation based on configuration changes and file modifications

**Modular Recognition Pipeline**: Each step is isolated:
- Audio preprocessing (resampling, normalization, duration checks)
- Embedding extraction (SpeechBrain ECAPA-TDNN)
- Similarity computation (cosine similarity)
- Score normalization (AS-Norm with background statistics)

**Flexible Speaker Management**: The system supports multiple speaker sources:
- Custom speakers in `enroll/` folder
- JVS corpus speakers (configurable inclusion)
- Common Voice speakers (configurable inclusion)
- Background speaker exclusion rules

### Development Notes

- The system automatically detects CUDA, MPS (Apple Silicon), or falls back to CPU
- Background embeddings are preferred over raw audio processing for performance
- Speaker IDs follow specific patterns: `jvs001-jvs100` for JVS, `cv_*`/`commonvoice_*` for Common Voice
- All audio is normalized to 16kHz mono with duration limits (2-30 seconds)
- The recognition threshold is configurable but defaults to 0.25
- Streamlit session state maintains model and speaker database across requests

### Performance Considerations

- GPU acceleration provides 3-5x speedup for embedding extraction
- Pre-computed background embeddings eliminate need for large dataset storage (2.7GB → 23MB)
- Speaker embedding caching reduces startup time from minutes to seconds
- Background model size is configurable (default: 100 samples) for memory/accuracy trade-off