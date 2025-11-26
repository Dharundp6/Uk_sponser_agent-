"""
Tool implementations for the UK Visa Assistant
Contains all the actual logic for each tool
"""
import json
import re
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import PyPDF2
import docx
import google.generativeai as genai

from models import UserProfile
from vector_db import EnhancedVectorDatabase
from config import MIN_SALARY, NEW_ENTRANT_SALARY


class ToolImplementations:
    """All tool implementations for the agent"""

    def __init__(self, gemini_model: genai.GenerativeModel, vector_db: EnhancedVectorDatabase):
        """
        Initialize tool implementations

        Args:
            gemini_model: Gemini model instance for LLM-based operations
            vector_db: Vector database for sponsor search
        """
        self.gemini_model = gemini_model
        self.vector_db = vector_db

    def search_visa_sponsors(self, query: str, max_results: int = 15,
                            filter_a_rated_only: bool = False) -> Dict[str, Any]:
        """Search sponsor database using FAISS"""
        return self.vector_db.search(query, max_results, filter_a_rated_only)

    def search_web_jobs(self, company_name: str, job_role: str = "graduate",
                       location: str = "UK", max_results: int = 10) -> Dict[str, Any]:
        """Search web for jobs"""
        try:
            # Robust type conversion
            company_name = str(company_name) if company_name else ""
            job_role = str(job_role) if job_role else "graduate"
            location = str(location) if location else "UK"

            try:
                max_results = int(max_results) if max_results is not None else 10
            except (ValueError, TypeError):
                max_results = 10
            max_results = max(1, min(max_results, 20))

            if not company_name.strip():
                return {
                    "success": False,
                    "error": "Company name is required",
                    "jobs": []
                }

            query = f"{company_name} {job_role} jobs {location} visa sponsorship site:linkedin.com OR site:indeed.co.uk"
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={max_results}"

            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            jobs = []
            for result in soup.find_all('div', class_='g')[:max_results]:
                title_elem = result.find('h3')
                link_elem = result.find('a')
                snippet_elem = result.find('div', class_=['VwiC3b', 'yXK7lf'])

                if title_elem and link_elem:
                    jobs.append({
                        "title": title_elem.get_text(),
                        "company": company_name,
                        "link": link_elem.get('href'),
                        "snippet": snippet_elem.get_text() if snippet_elem else ""
                    })

            return {
                "success": True,
                "count": len(jobs),
                "jobs": jobs,
                "search_query": query
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "jobs": []
            }

    def search_latest_company_news(self, company_name: str, search_query: str) -> Dict[str, Any]:
        """Search for latest company information using Gemini with Google Search grounding"""
        try:
            # Robust type conversion
            company_name = str(company_name).strip() if company_name else ""
            search_query = str(search_query).strip() if search_query else "hiring news"

            if not company_name:
                return {
                    "success": False,
                    "error": "Company name is required"
                }

            full_query = f"{company_name} {search_query} UK 2024 2025"

            print(f"      🔍 Searching with Gemini+Google: '{full_query}'")

            prompt = f"""Search for the latest information about {company_name} regarding: {search_query}

Focus on:
- Recent news (2024-2025)
- Hiring announcements
- Company expansions or changes
- Graduate programs
- Visa sponsorship information

Provide a structured summary with:
1. Key recent developments
2. Hiring status
3. Relevant links if available

Search query: {full_query}"""

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1500
                )
            )

            result_text = response.text

            grounding_metadata = []
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    metadata = candidate.grounding_metadata
                    if hasattr(metadata, 'grounding_chunks'):
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, 'web'):
                                grounding_metadata.append({
                                    'uri': getattr(chunk.web, 'uri', ''),
                                    'title': getattr(chunk.web, 'title', '')
                                })

            return {
                "success": True,
                "company": company_name,
                "search_query": search_query,
                "summary": result_text,
                "sources": grounding_metadata,
                "search_enabled": True
            }

        except Exception as e:
            print(f"      ⚠️ Gemini search failed: {e}")
            return self._fallback_web_search(company_name, search_query)

    def _fallback_web_search(self, company_name: str, search_query: str) -> Dict[str, Any]:
        """Fallback web scraping if Gemini search fails"""
        try:
            query = f"{company_name} {search_query} UK 2024 2025"
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&num=5"

            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            results = []
            for result in soup.find_all('div', class_='g')[:5]:
                title_elem = result.find('h3')
                link_elem = result.find('a')
                snippet_elem = result.find('div', class_=['VwiC3b', 'yXK7lf'])

                if title_elem and link_elem:
                    results.append({
                        "title": title_elem.get_text(),
                        "link": link_elem.get('href'),
                        "snippet": snippet_elem.get_text() if snippet_elem else ""
                    })

            summary = f"Found {len(results)} recent articles about {company_name} regarding {search_query}."

            return {
                "success": True,
                "company": company_name,
                "search_query": search_query,
                "summary": summary,
                "sources": results,
                "search_enabled": False
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def match_jobs_to_resume(self, companies: List[str], max_jobs: int = 5,
                            user_profile: Optional[UserProfile] = None) -> Dict[str, Any]:
        """Match jobs to user profile"""
        if not user_profile or not user_profile.skills:
            return {
                "success": False,
                "error": "No resume/profile available. Please upload resume first."
            }

        try:
            # Robust type conversion
            try:
                max_jobs = int(max_jobs) if max_jobs is not None else 5
            except (ValueError, TypeError):
                max_jobs = 5
            max_jobs = max(1, min(max_jobs, 20))

            # Handle companies list properly
            if isinstance(companies, str):
                companies = [companies]
            elif not isinstance(companies, list):
                companies = list(companies) if companies else []

            companies = [str(c).strip() for c in companies if c][:5]

            if not companies:
                return {
                    "success": False,
                    "error": "No valid company names provided"
                }

            matched_jobs = []

            for company in companies:
                job_role = user_profile.job_titles[0] if user_profile.job_titles else "graduate"
                job_results = self.search_web_jobs(company, job_role)

                if job_results['success']:
                    for job in job_results['jobs'][:2]:
                        job_text = f"{job['title']} {job['snippet']}".lower()
                        relevance = sum(5 for skill in user_profile.skills if skill.lower() in job_text)

                        matched_skills = [s for s in user_profile.skills if s.lower() in job_text]

                        job['relevance_score'] = relevance
                        job['matched_skills'] = matched_skills
                        matched_jobs.append(job)

            matched_jobs.sort(key=lambda x: x['relevance_score'], reverse=True)

            return {
                "success": True,
                "count": len(matched_jobs[:max_jobs]),
                "matched_jobs": matched_jobs[:max_jobs]
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def check_visa_eligibility(self, job_title: str, salary: Optional[float] = None,
                              has_uk_degree: bool = False) -> Dict[str, Any]:
        """Check visa eligibility"""
        # Robust type conversion
        job_title = str(job_title).strip() if job_title else "Unknown"

        try:
            salary = float(salary) if salary is not None else None
        except (ValueError, TypeError):
            salary = None

        if isinstance(has_uk_degree, str):
            has_uk_degree = has_uk_degree.lower() in ('true', '1', 'yes')
        else:
            has_uk_degree = bool(has_uk_degree) if has_uk_degree is not None else False

        eligibility = {
            "eligible": False,
            "requirements_met": [],
            "requirements_needed": [],
            "guidance": []
        }

        if salary:
            if salary >= MIN_SALARY:
                eligibility['requirements_met'].append(f"Salary £{salary:,.0f} meets standard threshold")
                eligibility['eligible'] = True
            elif salary >= NEW_ENTRANT_SALARY:
                eligibility['requirements_met'].append(f"Salary £{salary:,.0f} meets new entrant threshold")
                eligibility['guidance'].append("You qualify under 'new entrant' route (first 3 years in UK)")
                eligibility['eligible'] = True
            else:
                eligibility['requirements_needed'].append(f"Salary below minimum (£{MIN_SALARY:,.0f} standard, £{NEW_ENTRANT_SALARY:,.0f} new entrant)")
        else:
            eligibility['guidance'].append(f"Salary not specified - ensure job offers at least £{MIN_SALARY:,.0f} (or £{NEW_ENTRANT_SALARY:,.0f} for new entrants)")

        if has_uk_degree:
            eligibility['requirements_met'].append("UK degree (provides points advantage)")
            eligibility['guidance'].append("Your UK degree gives you extra points in the visa application")

        eligibility['guidance'].extend([
            "Must have job offer from licensed sponsor",
            "Job must be at RQF Level 3 or above (A-level equivalent)",
            "English language requirement: CEFR Level B1",
            "Healthcare surcharge: £1,035 per year"
        ])

        return {
            "success": True,
            "eligibility": eligibility,
            "job_title": job_title
        }

    def analyze_resume(self, file_path: str) -> Dict[str, Any]:
        """Analyze resume file"""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            ext = path.suffix.lower()
            if ext == '.pdf':
                text = self._extract_pdf(file_path)
            elif ext in ['.docx', '.doc']:
                text = self._extract_docx(file_path)
            elif ext == '.txt':
                text = path.read_text()
            else:
                return {"success": False, "error": f"Unsupported format: {ext}"}

            if not text.strip():
                return {"success": False, "error": "Could not extract text from file"}

            profile = self._parse_with_llm(text)

            return {
                "success": True,
                "profile": profile,
                "text_length": len(text)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    def _parse_with_llm(self, text: str) -> UserProfile:
        """Parse resume with LLM"""
        prompt = f"""Extract information from this resume in JSON format:

{text[:3000]}

Return ONLY JSON:
{{
    "name": "string or null",
    "skills": ["skill1", "skill2"],
    "experience_years": number or null,
    "education": ["degree1"],
    "job_titles": ["title1"],
    "industries": ["industry1"]
}}"""

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=800
                )
            )

            result_text = response.text.strip()
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*', '', result_text)

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)

            if json_match:
                data = json.loads(json_match.group())
                return UserProfile(
                    name=data.get('name'),
                    skills=data.get('skills', []),
                    experience_years=data.get('experience_years'),
                    education=data.get('education', []),
                    job_titles=data.get('job_titles', []),
                    industries=data.get('industries', []),
                    resume_text=text[:2000]
                )
        except Exception as e:
            print(f"      ⚠️ Resume parsing error: {e}")

        return UserProfile(resume_text=text[:2000])

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.vector_db.get_stats()
