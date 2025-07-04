import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

# Set API key in session state from environment variable if available
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", None)
with st.sidebar:
    st.title("ℹ️ Configuration")

    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input(
            "Enter your Google API Key:",
            type="password",
        )
        st.caption(
            "Get your API key from [Google AI Studio]"
            "(https://aistudio.google.com/apikey) 🔑"
        )
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("API Key set successfully!")
            st.rerun()
    else:
        st.success("API Key is configured")
        if st.button("Reset API Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.success("API Key reset successfully!")
            st.rerun()
    
    st.info(
        "This tool provides AI-powered analysis of medical imaging data using "
        "advanced computer vision and radiological expertise."
    )

medical_agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=st.session_state.GOOGLE_API_KEY,
    ),
    tools=[DuckDuckGoTools()],
    markdown=True
) if st.session_state.GOOGLE_API_KEY else None

if not medical_agent:
    st.warning("Please configure your API key in the sidebar to continue")

# Medical Analysis Query
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links of them too
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""

st.title("🏥 Medical Imaging Diagnosis Agent")
st.write("Upload a medical image for professional analysis")

# Creating containers for better organization
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload a medical image (X-ray, MRI, CT, etc.)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        help="Supported formats: JPG, JPEG, PNG, DICOM"
    )

if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = PILImage.open(uploaded_file)
            width, height = image.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_image = image.resize((new_width, new_height))

            st.image(
                resized_image,
                caption="Uploaded Medical Image",
                use_container_width=True
            )

            analyze_button = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True
            )

    with analysis_container:
        if analyze_button:
            if not medical_agent:
                st.error("Please configure your API key in the sidebar to analyze images.")
            else:
                with st.spinner("🔄 Analyzing image... Please wait."):
                    try:
                        temp_path = "temp_resized_image.png"
                        resized_image.save(temp_path)

                        # Create AgnoImage object
                        agno_image = AgnoImage(filepath=temp_path)

                        # Run Analysis
                        response = medical_agent.run(
                            query,
                            images=[agno_image]
                        )
                        st.markdown("### 📋 Analysis Results")
                        st.markdown("---")
                        st.markdown(response.content)
                        st.markdown("---")
                        st.caption(
                            "Note: This analysis is generated by AI and should be reviewed by "
                            "a qualified healthcare professional."
                        )
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

else:
    st.info("👆 Please upload a medical image to begin analysis")