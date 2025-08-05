# 🏥 Medical Imaging Diagnosis Agent (LangGraph)

An AI-powered medical imaging analysis tool that combines advanced image analysis with research-backed diagnostic insights using LangGraph workflows, Google Gemini AI, and medical literature search.

## 🌟 Features

- **Advanced AI Image Analysis**: Powered by Google Gemini 2.0 Flash for comprehensive medical image interpretation
- **LangGraph Workflow**: Structured multi-step analysis pipeline with memory and state management
- **Research Integration**: Automatic medical literature search via PubMed and ArXiv
- **Multiple Image Formats**: Support for X-ray, MRI, CT, Ultrasound, and DICOM files
- **Interactive Web Interface**: User-friendly Streamlit application
- **Evidence-Based Analysis**: Combines AI analysis with current medical research findings
- **Patient-Friendly Explanations**: Clear, accessible interpretations alongside technical analysis

## 🎯 How It Works

The agent uses a sophisticated LangGraph workflow with three main stages:

1. **Image Analysis Node**: Analyzes uploaded medical images using Google Gemini AI
2. **Literature Search Node**: Automatically searches medical databases for relevant research
3. **Synthesis Node**: Combines image findings with research context for comprehensive diagnosis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google AI Studio API Key ([Get one here](https://aistudio.google.com/apikey))

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "AI Medical Imaging Agent"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional):
   ```bash
   # Create a .env file
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

4. **Run the application**:
   ```bash
   streamlit run medical_imaging_agent_langgraph.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📋 Usage

1. **Configure API Key**: Enter your Google API key in the sidebar (or set it in the .env file)
2. **Upload Image**: Select a medical image file (X-ray, MRI, CT scan, etc.)
3. **Analyze**: Click "Analyze Image with LangGraph" to start the workflow
4. **Review Results**: Get comprehensive analysis with research context

## 🖼️ Supported Image Formats

| Format | Extensions | Typical Use Cases |
|--------|------------|-------------------|
| **X-ray** | JPG, JPEG, PNG | Bone fractures, chest imaging |
| **MRI** | DICOM, DCM, PNG, JPG | Soft tissue, brain imaging |
| **CT Scan** | DICOM, DCM, PNG, JPG | Cross-sectional imaging |
| **Ultrasound** | PNG, JPG, JPEG | Real-time imaging |

### File Limitations
- **Maximum size**: 10 MB
- **Maximum resolution**: 10,000,000 pixels
- **Supported formats**: JPG, JPEG, PNG, DCM, DICOM

## 🏗️ Technical Architecture

### LangGraph Workflow

```
START → Image Analysis → Literature Search → Diagnosis Synthesis → END
```

#### Nodes:
- **`analyze_image_node`**: Processes medical images using Gemini AI
- **`search_literature_node`**: Queries PubMed and ArXiv for relevant research
- **`synthesize_diagnosis_node`**: Combines findings into comprehensive diagnosis

#### State Management:
- **Memory**: Persistent conversation history
- **Image Data**: Base64 encoded image storage
- **Search Results**: Cached literature findings
- **Analysis Chain**: Maintains workflow state

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **AI Model** | Google Gemini 2.0 Flash | Image analysis and text generation |
| **Workflow** | LangGraph | Multi-step analysis pipeline |
| **UI** | Streamlit | Web interface |
| **Search** | PubMed + ArXiv APIs | Medical literature research |
| **Image Processing** | PIL (Pillow) | Image validation and processing |

## 🔧 Configuration

### Environment Variables

```bash
GOOGLE_API_KEY=your_google_ai_api_key
```

### Configuration Constants

```python
class Config:
    MAX_IMAGE_SIZE_MB = 10          # Maximum file size
    MAX_IMAGE_PIXELS = 10_000_000   # Maximum image resolution
    SEARCH_DELAY_SECONDS = 1        # Delay between API calls
    PUBMED_MAX_RESULTS = 5          # Number of search results
    GEMINI_MODEL = "gemini-2.0-flash"  # AI model version
    TEMPERATURE = 0.1               # Model creativity (0-1)
```

## 📊 Analysis Output Structure

### 1. Image Type & Region
- Imaging modality identification
- Anatomical region analysis
- Technical quality assessment

### 2. Key Findings
- Systematic observation listing
- Abnormality descriptions
- Measurements and characteristics
- Severity ratings

### 3. Diagnostic Assessment
- Primary diagnosis with confidence
- Differential diagnoses
- Evidence-based reasoning
- Critical findings flagging

### 4. Patient-Friendly Explanation
- Plain language interpretation
- Visual analogies
- Common concern addressing

### 5. Research Context & References
- Medical literature findings
- Direct links to research papers
- Evidence-based recommendations
- Treatment protocols

## 🛡️ Safety & Disclaimers

> ⚠️ **IMPORTANT MEDICAL DISCLAIMER**
> 
> This tool is designed for **educational and research purposes only**. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. 
> 
> - Always consult qualified healthcare professionals
> - Do not make medical decisions based solely on AI analysis
> - This tool cannot replace clinical expertise and patient examination
> - Emergency medical situations require immediate professional care

## 🔍 Research Integration

The agent automatically searches multiple medical databases:

- **PubMed**: Primary medical literature database
- **ArXiv**: Recent biomedical research papers
- **Fallback**: Manual search links when APIs are unavailable

### Search Strategy
1. Extract key terms from image analysis
2. Generate focused medical queries
3. Perform multi-source searches with rate limiting
4. Filter and rank results by relevance

## 🐛 Troubleshooting

### Common Issues

**API Key Problems**:
- Ensure your Google AI Studio API key is valid
- Check API quotas and billing status

**Image Upload Issues**:
- Verify file format is supported
- Check file size is under 10MB
- Ensure image is not corrupted

**Analysis Failures**:
- Check internet connectivity for literature search
- Verify API key permissions
- Review logs for specific error messages

### Logging

The application uses Python's logging module. Check console output for detailed error messages and debugging information.

---

**Remember**: This is an AI tool for educational purposes. Always consult healthcare professionals for medical decisions.
