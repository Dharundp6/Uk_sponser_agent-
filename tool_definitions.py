"""
Tool definitions for Gemini function calling
Defines the interface for all available tools
"""
import google.generativeai as genai


def get_tools():
    """
    Create tool definitions using Gemini's proper types

    Returns:
        Tool object containing all function declarations
    """

    search_visa_sponsors = genai.protos.FunctionDeclaration(
        name="search_visa_sponsors",
        description="Search the UK Home Office database for companies that can sponsor Skilled Worker visas. Returns up to 20 companies matching the query.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "query": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Search query (company name, industry, or location). Examples: 'Google', 'tech companies London', 'financial services'"
                ),
                "max_results": genai.protos.Schema(
                    type=genai.protos.Type.INTEGER,
                    description="Maximum number of results to return (1-20)"
                ),
                "filter_a_rated_only": genai.protos.Schema(
                    type=genai.protos.Type.BOOLEAN,
                    description="If true, only return companies with A-rating (actively sponsoring)"
                )
            },
            required=["query"]
        )
    )

    search_web_jobs = genai.protos.FunctionDeclaration(
        name="search_web_jobs",
        description="Search the web for current job openings at specific companies. Returns real job listings with application links.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "company_name": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Name of the company to search jobs for"
                ),
                "job_role": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Job role/title to search for. Examples: 'graduate', 'software engineer', 'data analyst', 'consultant'"
                ),
                "location": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Location filter (e.g., 'London', 'UK', 'Manchester')"
                ),
                "max_results": genai.protos.Schema(
                    type=genai.protos.Type.INTEGER,
                    description="Maximum results to return"
                )
            },
            required=["company_name"]
        )
    )

    search_latest_company_news = genai.protos.FunctionDeclaration(
        name="search_latest_company_news",
        description="Search for the LATEST news, hiring announcements, and real-time information about a company using Google Search. Use this when user asks for current/recent/latest information.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "company_name": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Name of the company"
                ),
                "search_query": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Specific search query about the company (e.g., 'recent hiring', 'layoffs 2024', 'expansion plans')"
                )
            },
            required=["company_name", "search_query"]
        )
    )

    match_jobs_to_resume = genai.protos.FunctionDeclaration(
        name="match_jobs_to_resume",
        description="Match available jobs to the user's resume/profile. Returns jobs ranked by relevance to user's skills and experience. ONLY use if resume has been uploaded.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "companies": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.STRING),
                    description="List of company names to search jobs for"
                ),
                "max_jobs": genai.protos.Schema(
                    type=genai.protos.Type.INTEGER,
                    description="Maximum jobs to return"
                )
            },
            required=["companies"]
        )
    )

    check_visa_eligibility = genai.protos.FunctionDeclaration(
        name="check_visa_eligibility",
        description="Check if user meets basic requirements for UK Skilled Worker visa based on their profile. Returns eligibility assessment and guidance.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "job_title": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="The job title/role being considered"
                ),
                "salary": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description="Annual salary in GBP (optional)"
                ),
                "has_uk_degree": genai.protos.Schema(
                    type=genai.protos.Type.BOOLEAN,
                    description="Whether applicant has UK degree"
                )
            },
            required=["job_title"]
        )
    )

    analyze_resume = genai.protos.FunctionDeclaration(
        name="analyze_resume",
        description="Parse and analyze a resume file (PDF, DOCX, TXT) to extract skills, experience, education, and create user profile. Must be called before match_jobs_to_resume.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "file_path": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Full path to resume file"
                )
            },
            required=["file_path"]
        )
    )

    get_database_stats = genai.protos.FunctionDeclaration(
        name="get_database_stats",
        description="Get statistics about the sponsor database and current session state. Useful for answering questions like 'how many companies sponsor visas?'",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={},
            required=[]
        )
    )

    # Return as a Tool containing all function declarations
    return genai.protos.Tool(
        function_declarations=[
            search_visa_sponsors,
            search_web_jobs,
            search_latest_company_news,
            match_jobs_to_resume,
            check_visa_eligibility,
            analyze_resume,
            get_database_stats
        ]
    )
