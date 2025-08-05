import os
import base64
import tempfile
import logging
from typing import TypedDict, Annotated, List, Dict, Any
from PIL import Image as PILImage
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
import requests
from urllib.parse import quote
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
class Config:
    MAX_IMAGE_SIZE_MB = 10
    MAX_IMAGE_PIXELS = 10_000_000  # 10M pixels
    SEARCH_DELAY_SECONDS = 1
    PUBMED_MAX_RESULTS = 5
    TEMP_FILE_PREFIX = "medical_img_"
    
    SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "dcm", "dicom"]
    GEMINI_MODEL = "gemini-2.0-flash"
    TEMPERATURE = 0.1

# Define the state structure for our LangGraph agent
class MedicalAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    image_analysis: str
    search_results: List[str]
    final_diagnosis: str
    image_data: str  # base64 encoded image
    image_type: str
    image_format: str  # Added to track actual image format


def validate_image(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded image file."""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > Config.MAX_IMAGE_SIZE_MB:
        return False, f"File too large: {file_size_mb:.1f}MB (max: {Config.MAX_IMAGE_SIZE_MB}MB)"
    
    # Check file type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in Config.SUPPORTED_IMAGE_FORMATS:
        return False, f"Unsupported format: {file_extension}"
    
    try:
        # Validate image can be opened
        image = PILImage.open(uploaded_file)
        width, height = image.size
        
        # Check image dimensions
        if width * height > Config.MAX_IMAGE_PIXELS:
            return False, f"Image too large: {width}x{height} pixels (max: {Config.MAX_IMAGE_PIXELS} pixels)"
        
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def get_image_mime_type(file_extension: str) -> str:
    """Get proper MIME type for image format."""
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'dcm': 'image/dicom',
        'dicom': 'image/dicom'
    }
    return mime_types.get(file_extension.lower(), 'image/png')


# Multi-source medical literature search tools
@tool
def search_pubmed(query: str) -> str:
    """Search PubMed for medical literature and research papers.
    
    Args:
        query: The search query for medical literature
        
    Returns:
        PubMed search results as a string
    """
    logger.info(f"Searching PubMed for: {query}")
    
    try:
        # Try LangChain PubMed first
        pubmed = PubmedQueryRun()
        results = pubmed.invoke(query)
        if results and len(results) > 50:
            logger.info("PubMed LangChain search successful")
            return f"PubMed results for '{query}':\n{results}"
    except Exception as e:
        logger.warning(f"PubMed LangChain search failed: {str(e)}")
    
    # Fallback to direct PubMed API
    try:
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={quote(query)}&retmax={Config.PUBMED_MAX_RESULTS}&retmode=xml"
        response = requests.get(search_url, timeout=10)
        
        if response.status_code == 200:
            # Simple extraction of PMIDs for demo
            import re
            pmids = re.findall(r'<Id>(\d+)</Id>', response.text)
            if pmids:
                links = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids[:3]]
                result = f"PubMed results for '{query}':\nFound {len(pmids)} articles:\n"
                for i, (pmid, link) in enumerate(zip(pmids[:3], links), 1):
                    result += f"{i}. PMID: {pmid} - {link}\n"
                logger.info("PubMed API search successful")
                return result
    except Exception as e:
        logger.error(f"PubMed API search failed: {str(e)}")
    
    # Always provide a manual search link as fallback
    search_link = f"https://pubmed.ncbi.nlm.nih.gov/?term={quote(query)}"
    logger.warning("All PubMed searches failed, providing manual link")
    return f"PubMed search for '{query}':\nAPI temporarily unavailable, but you can search manually:\n🔗 {search_link}"


@tool
def search_arxiv_medical(query: str) -> str:
    """Search ArXiv for recent medical and biomedical research papers.
    
    Args:
        query: The search query for research papers
        
    Returns:
        ArXiv search results as a string with clickable links
    """
    logger.info(f"Searching ArXiv for: {query}")
    
    try:
        arxiv = ArxivQueryRun()
        results = arxiv.invoke(query)
        
        # Extract and add ArXiv links
        if results:
            import re
            enhanced_results = results
            
            # Try multiple patterns for arXiv IDs
            patterns = [
                r'(\d{4}\.\d{4,5})',  # 2023.12345 format
                r'arXiv:(\d{4}\.\d{4,5})',  # arXiv:2023.12345 format
            ]
            
            # Also add a generic link to search for the title on arXiv
            title_match = re.search(r'Title:\s*([^\n]+)', results)
            if title_match:
                title = title_match.group(1).strip()
                search_link = f"https://arxiv.org/search/?query={quote(title[:50])}&searchtype=title"
                enhanced_results += f"\n🔗 Search this paper on ArXiv: {search_link}"
            
            # Look for arXiv IDs and add direct links
            for pattern in patterns:
                arxiv_ids = re.findall(pattern, results)
                for arxiv_id in arxiv_ids:
                    link = f"https://arxiv.org/abs/{arxiv_id}"
                    enhanced_results = enhanced_results.replace(
                        arxiv_id, 
                        f"{arxiv_id} (🔗 {link})"
                    )
            
            logger.info("ArXiv search successful")
            return f"ArXiv results for '{query}':\n{enhanced_results}"
        
        return f"ArXiv results for '{query}':\n{results}"
    except Exception as e:
        logger.error(f"ArXiv search failed: {str(e)}")
        return f"ArXiv search failed: {str(e)}"


@tool
def search_medical_literature(query: str) -> str:
    """Comprehensive medical literature search using multiple sources.
    
    Args:
        query: The search query for medical information
        
    Returns:
        Combined search results from multiple sources
    """
    all_results = []
    
    # Try PubMed first (best for medical literature)
    pubmed_result = search_pubmed.invoke(query)
    if pubmed_result and "search failed" not in pubmed_result.lower():
        all_results.append(pubmed_result)
    
    # Try ArXiv for recent research papers
    arxiv_result = search_arxiv_medical.invoke(query)
    if arxiv_result and "search failed" not in arxiv_result.lower():
        all_results.append(arxiv_result)
    
    if all_results:
        combined_results = "\n\n" + "="*50 + "\n\n".join(all_results)
        return f"Multi-source search results for '{query}':{combined_results}"
    else:
        return f"All search sources failed for query: '{query}'. Proceeding with general medical knowledge."


# Tool for image encoding with better error handling
def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string for processing."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image: {str(e)}")
        raise


class MedicalImagingAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            self.model = ChatGoogleGenerativeAI(
                model=Config.GEMINI_MODEL,
                google_api_key=api_key,
                temperature=Config.TEMPERATURE
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise
            
        self.tools = [search_medical_literature]
        self.tool_node = ToolNode(self.tools)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(MedicalAgentState)
        
        # Add nodes
        workflow.add_node("analyze_image", self.analyze_image_node)
        workflow.add_node("search_literature", self.search_literature_node)
        workflow.add_node("synthesize_diagnosis", self.synthesize_diagnosis_node)
        
        # Add edges
        workflow.add_edge(START, "analyze_image")
        workflow.add_edge("analyze_image", "search_literature")
        workflow.add_edge("search_literature", "synthesize_diagnosis")
        workflow.add_edge("synthesize_diagnosis", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def analyze_image_node(self, state: MedicalAgentState) -> Dict[str, Any]:
        """Node to analyze the medical image."""
        # Get the image data from state
        image_data = state.get("image_data", "")
        image_format = state.get("image_format", "png")
        
        if not image_data:
            return {"image_analysis": "No image data provided"}
        
        # Get proper MIME type based on format
        mime_type = get_image_mime_type(image_format)
        
        # Create a message with the image
        message_content = [
            {
                "type": "text",
                "text": """Analyze this medical image and provide:

            1. **Image Type & Region**:
            - Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
            - Identify anatomical region and positioning
            - Comment on image quality and technical adequacy

            2. **Key Findings**:
            - List primary observations systematically
            - Note any abnormalities with precise descriptions
            - Include measurements and densities where relevant
            - Describe location, size, shape, and characteristics
            - Rate severity: Normal/Mild/Moderate/Severe

            3. **Initial Assessment**:
            - Provide preliminary observations
            - Note any areas requiring further research
            - Identify key terms for literature search

            Please be thorough but concise in your analysis."""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }
            }
        ]
        
        try:
            response = self.model.invoke([HumanMessage(content=message_content)])
            analysis = response.content
            logger.info("Image analysis completed successfully")
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            analysis = f"Error analyzing image: {str(e)}"
        
        return {"image_analysis": analysis}
    
    def search_literature_node(self, state: MedicalAgentState) -> Dict[str, Any]:
        """Node to generate search queries and perform searches."""
        image_analysis = state.get("image_analysis", "")
        
        # Generate search queries based on the image analysis
        search_prompt = f"""Based on this medical image analysis:

        {image_analysis}

        Generate 2 focused search queries for medical research:
        1. Main condition/diagnosis
        2. Treatment or imaging technique

        Format each query as a separate line starting with "SEARCH:" using simple medical terms.
        Example: "SEARCH: radius ulna fracture"
        """
        
        try:
            response = self.model.invoke([HumanMessage(content=search_prompt)])
            search_queries = []
            
            # Extract search queries from response
            content = response.content if isinstance(response.content, str) else str(response.content)
            lines = content.split('\n')
            for line in lines:
                if line.strip().startswith("SEARCH:"):
                    query = line.replace("SEARCH:", "").strip()
                    if query:
                        search_queries.append(query)
            
            # If no queries found, create default ones
            if not search_queries:
                search_queries = [
                    "medical imaging diagnosis",
                    "radiology protocols"
                ]
            
        except Exception as e:
            logger.error(f"Failed to generate search queries: {str(e)}")
            search_queries = [f"Error generating search queries: {str(e)}"]
        
        # Perform searches directly and collect results
        import time
        search_results = []
        successful_searches = 0
        
        for i, query in enumerate(search_queries):
            try:
                # Add delay between searches  
                if i > 0:
                    time.sleep(Config.SEARCH_DELAY_SECONDS)
                    
                logger.info(f"Attempting search {i+1}/{len(search_queries)} for: {query}")
                result = search_medical_literature.invoke(query)
                if result and len(result) > 50 and "Search failed" not in result:
                    search_results.append(result)
                    successful_searches += 1
                    logger.info(f"Search successful, result length: {len(result)}")
                else:
                    search_results.append(f"Search returned minimal results for: {query}")
                    logger.warning(f"Search returned minimal/failed results for: {query}")
            except Exception as e:
                error_msg = f"Search failed for '{query}': {str(e)}"
                search_results.append(error_msg)
                logger.error(error_msg)
        
        # Store results in state
        return {
            "search_results": search_results,
            "messages": [AIMessage(content=f"Completed {successful_searches}/{len(search_queries)} successful literature searches")]
        }
    
    def synthesize_diagnosis_node(self, state: MedicalAgentState) -> Dict[str, Any]:
        """Node to synthesize final diagnosis with research context."""
        image_analysis = state.get("image_analysis", "")
        search_results = state.get("search_results", [])
        
        # Format search results properly
        research_context = "Search completed. Analysis includes available research findings and standard medical knowledge."
        successful_results = [result for result in search_results if result and "search failed" not in result.lower() and "unavailable" not in result.lower() and len(result) > 100]
        
        if successful_results:
            research_context = f"Found {len(successful_results)} research sources:\n\n" + "\n\n".join([f"Search Result {i+1}:\n{result}" for i, result in enumerate(successful_results)])
        else:
            research_context = "Research analysis based on standard evidence-based medical knowledge and clinical guidelines."
        
        synthesis_prompt = f"""Based on the medical image analysis and any available research findings, provide a comprehensive diagnosis:

        ## Image Analysis:
        {image_analysis}

        ## Research Context:
        {research_context}

        Please structure your final response as follows:

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

        ### 5. Research Context & References
        - If research results are available, summarize relevant findings from medical literature
        - **Include direct links to research papers and sources when available**
        - When search results are not available, provide standard evidence-based medical knowledge
        - Include general treatment protocols and diagnostic criteria based on established medical practice
        - Clearly indicate whether recommendations are based on internet research or standard medical knowledge
        - Format links as clickable URLs for easy access

        **Important**: This analysis is for educational purposes and should be reviewed by a qualified healthcare professional.

        Format your response using clear markdown headers and bullet points. Be thorough yet accessible.
        """
        
        try:
            response = self.model.invoke([HumanMessage(content=synthesis_prompt)])
            final_diagnosis = response.content
            logger.info("Diagnosis synthesis completed successfully")
        except Exception as e:
            logger.error(f"Diagnosis synthesis failed: {str(e)}")
            final_diagnosis = f"Error synthesizing diagnosis: {str(e)}"
        
        return {
            "final_diagnosis": final_diagnosis,
            "messages": [AIMessage(content=final_diagnosis)]
        }
    
    def analyze_medical_image(self, image_path: str, image_format: str = "png") -> str:
        """Main method to analyze a medical image."""
        try:
            # Encode image to base64
            image_data = encode_image_to_base64(image_path)
            
            # Run the graph
            initial_state: MedicalAgentState = {
                "messages": [HumanMessage(content="Analyze the uploaded medical image")],
                "image_data": image_data,
                "image_analysis": "",
                "search_results": [],
                "final_diagnosis": "",
                "image_type": "medical_image",
                "image_format": image_format
            }
            
            result = self.graph.invoke(initial_state, config={"configurable": {"thread_id": "medical_analysis"}})
            return result.get("final_diagnosis", "Analysis failed")
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return f"Error during analysis: {str(e)}"


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Medical Imaging Agent",
        page_icon="🏥",
        layout="wide"
    )
    
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
            "LangGraph, Google Gemini, and internet search for research context."
        )
        
        # Display configuration info
        with st.expander("📋 Configuration Details"):
            st.write(f"**Max Image Size:** {Config.MAX_IMAGE_SIZE_MB} MB")
            st.write(f"**Max Image Pixels:** {Config.MAX_IMAGE_PIXELS:,}")
            st.write(f"**Supported Formats:** {', '.join(Config.SUPPORTED_IMAGE_FORMATS).upper()}")

    # Initialize agent
    medical_agent = None
    if st.session_state.GOOGLE_API_KEY:
        try:
            medical_agent = MedicalImagingAgent(st.session_state.GOOGLE_API_KEY)
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            logger.error(f"Agent initialization failed: {str(e)}")

    if not medical_agent:
        st.warning("Please configure your API key in the sidebar to continue")
        return

    st.title("🏥 Medical Imaging Diagnosis Agent (LangGraph)")
    st.write("Upload a medical image for professional analysis using advanced AI workflows")

    # Creating containers for better organization
    upload_container = st.container()
    image_container = st.container()
    analysis_container = st.container()

    with upload_container:
        uploaded_file = st.file_uploader(
            "Upload a medical image (X-ray, MRI, CT, etc.)",
            type=Config.SUPPORTED_IMAGE_FORMATS,
            help=f"Supported formats: {', '.join(Config.SUPPORTED_IMAGE_FORMATS).upper()}\nMax size: {Config.MAX_IMAGE_SIZE_MB} MB"
        )

    if uploaded_file is not None:
        # Validate the uploaded file
        is_valid, validation_message = validate_image(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ Invalid file: {validation_message}")
            return
        
        # Get file extension for proper MIME type handling
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with image_container:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                try:
                    image = PILImage.open(uploaded_file)
                    width, height = image.size
                    aspect_ratio = width / height
                    new_width = 500
                    new_height = int(new_width / aspect_ratio)
                    resized_image = image.resize((new_width, new_height))

                    st.image(
                        resized_image,
                        caption=f"Uploaded Medical Image ({width}x{height}, {file_extension.upper()})",
                        use_container_width=True
                    )

                    analyze_button = st.button(
                        "🔍 Analyze Image with LangGraph",
                        type="primary",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    return

        with analysis_container:
            if analyze_button:
                if not medical_agent:
                    st.error("Please configure your API key in the sidebar to analyze images.")
                else:
                    with st.spinner("🔄 Analyzing image with LangGraph workflow... Please wait."):
                        # Use secure temporary file handling
                        temp_file = None
                        try:
                            # Create secure temporary file
                            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False, prefix=Config.TEMP_FILE_PREFIX) as temp_file:
                                resized_image.save(temp_file.name)
                                temp_path = temp_file.name

                            # Run Analysis using LangGraph agent
                            result = medical_agent.analyze_medical_image(temp_path, file_extension)
                            
                            st.markdown("### 📋 Analysis Results (LangGraph)")
                            st.markdown("---")
                            st.markdown(result)
                            st.markdown("---")
                            st.caption(
                                "⚠️ **Disclaimer**: This analysis is generated by AI using LangGraph workflow and should be reviewed by "
                                "a qualified healthcare professional. This tool is for educational purposes only."
                            )
                            
                        except Exception as e:
                            logger.error(f"Analysis error: {str(e)}")
                            st.error(f"Error during analysis: {str(e)}")
                        finally:
                            # Clean up temporary file securely
                            if temp_file and os.path.exists(temp_path):
                                try:
                                    os.unlink(temp_path)
                                except Exception as e:
                                    logger.warning(f"Failed to clean up temp file: {str(e)}")

    else:
        st.info("👆 Please upload a medical image to begin analysis")
        
        # Show example of supported formats
        with st.expander("ℹ️ Supported Medical Image Formats"):
            st.write("**X-ray**: JPG, JPEG, PNG")
            st.write("**MRI**: DICOM, DCM, PNG, JPG")
            st.write("**CT Scan**: DICOM, DCM, PNG, JPG")
            st.write("**Ultrasound**: PNG, JPG, JPEG")


if __name__ == "__main__":
    main()