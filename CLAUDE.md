# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is a Japanese speaker recognition and multi-speaker analysis system built with SpeechBrain's ECAPA-TDNN model and pyannote.audio. The system provides both single speaker identification and comprehensive multi-speaker diarization with temporal analysis.

## Common Commands

### Running the Application
```bash
# Start the main Streamlit web application
streamlit run app.py

# Test the core recognition system (single speaker)
python enhanced_speaker_recognition.py

# Test multi-speaker analysis system
python speaker_diarization.py

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

### Multi-Speaker Analysis Setup
```bash
# Set Hugging Face token for pyannote.audio (required)
echo "HF_TOKEN=your_huggingface_token_here" > .env

# Accept terms at: https://huggingface.co/pyannote/speaker-diarization
```

### Background Data Processing
```bash
# Create JVS embeddings from downloaded dataset
python prep_jvs_embeddings.py [JVS_PATH] --per-speaker 50

# Create Common Voice embeddings (requires Hugging Face auth)
python prep_common_voice_embeddings.py --max-samples 5000

# Create dummy embeddings for testing (quick start)
python create_dummy_cv_embeddings.py
```

### Testing and Validation
```bash
# Test dataset manager functionality
python dataset_manager.py

# Test background loader
python background_loader.py

# Test segment processor (multi-speaker)
python segment_processor.py

# Test speaker diarization
python speaker_diarization.py
```

## Architecture Overview

This system combines single speaker recognition with multi-speaker diarization analysis using a modular, two-stage approach:

### Core Components

**Main Application (`app.py`)**
- Streamlit web interface with 4-tab structure
- Session state management for both single and multi-speaker systems
- Real-time model initialization and caching controls
- Configuration management through UI controls
- Tabs: 🎤 Single Speaker, 🎭 Multi-Speaker Analysis, 👥 Speaker Management, 📊 Statistics

**Single Speaker Recognition (`enhanced_speaker_recognition.py`)**
- Primary `JapaneseSpeakerRecognizer` class
- ECAPA-TDNN embedding extraction and similarity scoring
- AS-Norm (Adaptive Score Normalization) using background models
- Speaker database management with caching (`enrolled_speakers_embeddings.npz`)
- Support for both local speakers and dataset speakers (JVS/Common Voice)
- New `recognize_segment()` method for multi-speaker integration

**Multi-Speaker Analysis System (`speaker_diarization.py`)**
- `SpeakerDiarizer` class using pyannote.audio for temporal speaker separation
- `MultiSpeakerRecognizer` class integrating diarization with recognition
- Two-stage processing: diarization → segment recognition
- Comprehensive result structure with timeline data and confidence scores
- Support for configurable speaker count and filtering options

**Segment Processing (`segment_processor.py`)**
- `SegmentProcessor` class handling audio segment extraction
- Audio preprocessing for recognition (16kHz conversion, normalization)
- Bridge between raw diarization results and recognition system
- Validation and quality control for audio segments

**Dataset Management (`dataset_manager.py`)**
- Enhanced `DatasetManager` class with speaker classification
- Distinguishes between JVS speakers (jvs001-jvs100), Common Voice, and custom speakers
- Configurable exclusion rules for background speakers
- Audio file discovery and validation

**Background Processing (`background_loader.py`)**
- `BackgroundEmbeddingLoader` manages pre-computed embeddings
- Loads JVS and Common Voice embeddings from `.npz` files
- Provides combined background embeddings for AS-Norm
- Automatic fallback to audio processing if embeddings unavailable

### Multi-Speaker Analysis Architecture

#### Two-Stage Processing Pipeline
1. **Diarization Stage**: pyannote.audio processes raw audio to detect "when who is speaking"
2. **Recognition Stage**: Each detected segment is processed through ECAPA-TDNN for speaker identification

#### Data Flow
1. **Audio Input**: Multi-speaker audio file uploaded via Streamlit
2. **Diarization**: pyannote.audio creates temporal speaker segments
3. **Segment Extraction**: Each segment is extracted and preprocessed to 16kHz
4. **Recognition**: Each segment is recognized against enrolled speakers
5. **Integration**: Results combined into comprehensive timeline analysis
6. **Visualization**: Multiple chart types (diarization timeline, speaker timeline, statistics)

### Configuration System

Central configuration in `config.json` controls all aspects:

```json
{
  "model": {
    "speechbrain_model": "speechbrain/spkrec-ecapa-voxceleb",
    "device": "auto"
  },
  "audio": {
    "sample_rate": 16000,
    "duration_min": 2.0,
    "duration_max": 30.0
  },
  "recognition": {
    "threshold": 0.25,
    "use_score_normalization": true,
    "background_speakers_count": 100
  },
  "diarization": {
    "model": "pyannote/speaker-diarization-3.1",
    "min_speakers": 1,
    "max_speakers": 10,
    "min_segment_duration": 0.5
  },
  "segment_processing": {
    "target_sample_rate": 16000,
    "normalize": true
  },
  "datasets": {
    "exclude_background_speakers": true,
    "allow_jvs_speakers": true,
    "allow_common_voice_speakers": false
  },
  "ui": {
    "show_jvs_in_results": false,
    "show_common_voice_in_results": false
  }
}
```

### File Structure

```
japanese-speaker-recognition_GPU_nvidia/
├── app.py                                    # Main Streamlit application (4 tabs)
├── enhanced_speaker_recognition.py           # Single speaker recognition system
├── speaker_diarization.py                   # Multi-speaker analysis system (NEW)
├── segment_processor.py                     # Audio segment processing (NEW)
├── dataset_manager.py                       # Dataset and speaker management
├── background_loader.py                     # Background embedding management
├── config.json                              # Comprehensive configuration
├── .env                                     # Environment variables (HF_TOKEN)
├── requirements.txt                         # Dependencies (including pyannote.audio)

# Embedding files
├── background_jvs_ecapa.npz                 # JVS background embeddings (3.8MB)
├── background_common_voice_ja_ecapa.npz     # Common Voice embeddings (19MB)
├── enrolled_speakers_embeddings.npz         # Speaker database cache

# Speaker enrollment
├── enroll/                                  # User speaker enrollment
│   ├── speaker1/
│   │   ├── sample1.wav
│   │   └── sample2.wav
│   └── speaker2/

# Processing scripts
├── prep_jvs_embeddings.py                   # JVS embedding creation
├── prep_common_voice_embeddings.py          # Common Voice embedding creation
├── create_embeddings.py                     # Unified embedding creation
├── create_dummy_cv_embeddings.py            # Quick start dummy data

# Documentation
├── README.md                                # Complete system documentation
├── QUICK_START_GUIDE.md                     # Quick start for both modes
├── SPEAKER_MANAGEMENT_GUIDE.md              # Speaker management guide
├── FAQ.md                                   # Comprehensive FAQ
└── CLAUDE.md                                # This file
```

### Key Design Patterns

**Modular Two-Stage Architecture**: 
- Clear separation between diarization (pyannote.audio) and recognition (ECAPA-TDNN)
- Each stage optimized for its specific task with appropriate preprocessing
- Segment processor bridges the gap between stages

**Unified Speaker Management**: 
- Single enrollment system (`enroll/` folder) serves both single and multi-speaker modes
- Consistent speaker database across all functionalities
- Flexible filtering options for JVS/Common Voice speakers

**Comprehensive Caching Strategy**: 
- Speaker embeddings cached with timestamp validation
- Background embeddings pre-computed to avoid large dataset downloads
- Multi-level caching for optimal performance

**Interactive Visualization**:
- Multiple timeline views: diarization results and speaker recognition results
- Real-time filtering and configuration options
- Detailed per-segment analysis with top-5 candidate scores

### Processing Requirements

#### Single Speaker Recognition
- Input: 2-30 seconds, single speaker audio
- Processing: ECAPA-TDNN embedding + AS-Norm scoring
- Output: Speaker identification with confidence scores

#### Multi-Speaker Analysis
- Input: Multi-speaker audio (any length, 2-10 speakers)
- Processing: pyannote.audio diarization → segment recognition
- Output: Timeline analysis with speaker switching patterns

### Environment Variables

```bash
# Required for multi-speaker analysis
HF_TOKEN=your_huggingface_token_here
```

### Dependencies

Key dependencies for multi-speaker functionality:
```
pyannote.audio>=3.1.0
python-dotenv>=0.19.0
speechbrain>=0.5.12
torch>=1.13.0
streamlit>=1.20.0
plotly>=5.0.0
```

## Development Notes

### Multi-Speaker Analysis Implementation

- **pyannote.audio**: Handles diarization with automatic speaker detection
- **Segment Recognition**: Each diarized segment processed individually through existing recognition pipeline
- **Result Integration**: Comprehensive data structure combining temporal and recognition information
- **Visualization**: Plotly-based interactive charts for timeline analysis

### Speaker Database Integration

- **Unified Enrollment**: Same `enroll/` folder structure serves both modes
- **Automatic Detection**: System automatically detects CUDA/MPS/CPU
- **Background Embeddings**: AS-Norm works for both single and multi-speaker recognition
- **Filtering Options**: JVS/Common Voice speakers can be shown/hidden independently

### Configuration Management

- **Single Config File**: `config.json` controls all system aspects
- **UI Integration**: Many settings controllable through Streamlit interface
- **Environment Variables**: Sensitive data (HF_TOKEN) in `.env` file
- **Runtime Configuration**: Settings can be modified without restart

### Performance Considerations

- **GPU Acceleration**: 3-5x speedup for both diarization and recognition
- **Background Embeddings**: Eliminate need for 2.7GB+ dataset storage
- **Segment Caching**: Reduces repeated processing of similar audio segments
- **Memory Management**: Configurable background speaker count and max speakers

### Authentication Requirements

Multi-speaker analysis requires:
1. Hugging Face account and token
2. Agreement to pyannote/speaker-diarization terms
3. Token configuration in `.env` file

This ensures compliance with model licensing and usage terms.

## Important Implementation Details

### Segment Processing Flow

1. **Diarization**: Raw audio → pyannote.audio → temporal segments
2. **Extraction**: Segments extracted from original audio
3. **Preprocessing**: Segments converted to 16kHz, normalized for recognition
4. **Recognition**: Each segment processed through ECAPA-TDNN pipeline
5. **Integration**: Results combined with temporal information

### Speaker ID Patterns

- **Custom speakers**: Any name in `enroll/` folder
- **JVS speakers**: `jvs001` through `jvs100`
- **Common Voice speakers**: `cv_*` or `commonvoice_*` prefixes

### Visualization Features

- **Diarization Timeline**: Raw pyannote.audio results with SPEAKER_XX labels
- **Speaker Timeline**: Recognition-based timeline with actual speaker names
- **Statistics Table**: Per-speaker analysis with time, segments, confidence
- **Top-5 Display**: Candidate ranking for each segment

## Troubleshooting Common Issues

### Multi-Speaker Analysis Issues

1. **HF_TOKEN Error**: Ensure token is set in `.env` and terms are accepted
2. **No Segments Detected**: Check audio quality, speaker separation, duration
3. **Poor Recognition**: Verify speaker enrollment, consider background embeddings
4. **Memory Issues**: Reduce `max_speakers` and `background_speakers_count` in config

### General Issues

1. **Model Loading**: Ensure internet connection for initial download
2. **Audio Format**: Use supported formats (WAV, MP3, FLAC, M4A, OGG)
3. **Duration Limits**: 2-30 seconds for single speaker, no limit for multi-speaker
4. **Speaker Enrollment**: Ensure 2-5 audio samples per speaker in `enroll/` folders

The system is designed to be robust and user-friendly while providing professional-grade speaker analysis capabilities.