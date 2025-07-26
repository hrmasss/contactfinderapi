import re
import time
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from email_validator import validate_email, EmailNotValidError
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field
from .core import (
    ContactFinder,
    CompanyInfo,
    Email,
    CompanyResearchResult,
    ContactFinderConfig,
    EmailValidator,
)
from .validators import (
    MXValidator,
    DomainValidator,
    EmailCheckerValidator,
    EmailBounceValidator,
)


# ======================================
# STRUCTURED OUTPUT MODELS
# ======================================


class EmployeeEmailSearchResult(BaseModel):
    """Structured result for employee email search"""

    found_emails: List[str] = Field(
        description="Publicly found email addresses for the employee"
    )
    generated_emails: List[str] = Field(
        description="Generated email addresses based on patterns"
    )
    confidence_scores: List[float] = Field(
        description="Confidence scores for each email (0.0-1.0)"
    )
    reasoning: str = Field(
        description="Brief explanation of name parsing and pattern application"
    )


# ======================================
# UTILITIES
# ======================================


class EmailUtils:
    """Clean email utilities using email-validator package"""

    EXCLUDED_DOMAINS = {
        "gmail.com",
        "outlook.com",
        "yahoo.com",
        "hotmail.com",
        "protonmail.com",
        "aol.com",
        "icloud.com",
        "live.com",
        "hunter.io",
        "leadiq.com",
        "clearbit.com",
        "linkedin.com",
        "facebook.com",
        "twitter.com",
    }

    @staticmethod
    def clean_email(email: str) -> str:
        """Clean and validate email using email-validator"""
        try:
            email = email.strip().lower()
            if not email or "@" not in email:
                return ""
            validated = validate_email(email, check_deliverability=False)
            return validated.email
        except EmailNotValidError:
            return ""

    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract valid business emails from text"""
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        emails = re.findall(email_pattern, text)

        clean_emails = []
        for email in emails:
            cleaned = EmailUtils.clean_email(email)
            if cleaned and EmailUtils.is_business_email(cleaned):
                clean_emails.append(cleaned)

        return list(set(clean_emails))

    @staticmethod
    def is_business_email(email: str) -> bool:
        """Check if email is from a business domain"""
        if "@" not in email:
            return False
        domain = email.split("@")[1]
        return domain not in EmailUtils.EXCLUDED_DOMAINS

    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract clean domain from URL"""
        if not url:
            return ""

        domain = re.sub(r"^https?://", "", url.strip().lower())
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0].split("?")[0].split("#")[0]

        if not domain or "." not in domain or len(domain) < 4:
            return ""

        return domain


# ======================================
# SMART EMAIL GENERATION ENGINE
# ======================================


class SmartEmailGenerator:
    """LLM-powered email generation with intelligent domain/pattern cycling"""

    def __init__(
        self, llm: ChatOpenAI, search_api: Optional[GoogleSerperAPIWrapper] = None
    ):
        self.llm = llm
        self.search_api = search_api

    def search_for_employee_email(
        self, employee_name: str, company_info: CompanyInfo
    ) -> List[str]:
        """Search for publicly available employee emails"""
        if not self.search_api:
            return []

        found_emails = []
        search_queries = [
            f'"{employee_name}" {company_info.name} email contact',
            f'"{employee_name}" @{company_info.domains[0] if company_info.domains else ""}',
            f"{employee_name} {company_info.name} directory contact information",
        ]

        for query in search_queries:
            try:
                result = self.search_api.run(query)
                emails = EmailUtils.extract_emails(result)
                # Filter emails that match company domains
                for email in emails:
                    if "@" in email:
                        email_domain = email.split("@")[1]
                        if email_domain in company_info.domains:
                            found_emails.append(email)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Search failed for query '{query}': {e}")
                continue

        return list(set(found_emails))

    def generate_emails_with_llm(
        self,
        employee_name: str,
        company_info: CompanyInfo,
        essential_patterns: List[str],
        max_emails: int,
    ) -> EmployeeEmailSearchResult:
        """Generate emails using LLM with smart domain/pattern cycling"""

        # First search for publicly available emails
        found_emails = self.search_for_employee_email(employee_name, company_info)

        # Create smart cycling of domains and patterns
        domain_pattern_combinations = self._create_smart_combinations(
            company_info.domains, essential_patterns, max_emails - len(found_emails)
        )

        prompt = f"""
        You are an expert at analyzing names and generating professional email addresses.
        
        TASK: Generate email addresses for "{employee_name}" at {company_info.name}
        
        COMPANY DOMAINS (in order of preference): {company_info.domains}
        
        ESSENTIAL EMAIL PATTERNS to follow:
        {chr(10).join([f"- {pattern}" for pattern in essential_patterns])}
        
        DOMAIN-PATTERN COMBINATIONS to generate (in priority order):
        {chr(10).join([f"- {domain} with {pattern}" for domain, pattern in domain_pattern_combinations])}
        
        INSTRUCTIONS:
        1. Parse the name "{employee_name}" intelligently - it may not be simple "first last"
        2. Handle complex names (multiple first names, compound last names, titles, etc.)
        3. Generate emails following the exact domain-pattern combinations provided above
        4. For each pattern, apply it correctly:
           - firstname.lastname@domain.com: john.smith@company.com
           - firstname_lastname@domain.com: john_smith@company.com  
           - firstlast@domain.com: johnsmith@company.com
           - f.lastname@domain.com: j.smith@company.com
           - flast@domain.com: jsmith@company.com
           - firstname@domain.com: john@company.com
           - firstl@domain.com: johns@company.com
        5. Assign confidence scores (0.1-1.0) based on:
           - Domain preference (first domain = higher confidence)
           - Pattern commonality (firstname.lastname typically most common)
           - Name complexity handling
        
        ALREADY FOUND EMAILS (give these confidence 0.95): {found_emails}
        
        Return EXACTLY the number of generated emails requested: {len(domain_pattern_combinations)}
        """

        try:
            structured_llm = self.llm.with_structured_output(EmployeeEmailSearchResult)
            result = structured_llm.invoke(prompt)

            # Validate and clean the results
            all_emails = found_emails + result.generated_emails
            all_confidences = [0.95] * len(found_emails) + result.confidence_scores

            # Ensure we don't exceed max_emails
            if len(all_emails) > max_emails:
                all_emails = all_emails[:max_emails]
                all_confidences = all_confidences[:max_emails]

            return EmployeeEmailSearchResult(
                found_emails=found_emails,
                generated_emails=result.generated_emails,
                confidence_scores=all_confidences,
                reasoning=result.reasoning,
            )

        except Exception as e:
            print(f"LLM email generation failed: {e}")
            # Fallback to basic generation
            return self._fallback_generation(
                employee_name, company_info, essential_patterns, max_emails
            )

    def _create_smart_combinations(
        self, domains: List[str], patterns: List[str], target_count: int
    ) -> List[tuple]:
        """Create smart domain-pattern combinations with cycling logic"""
        if not domains or not patterns:
            return []

        combinations = []
        pattern_chunks = [patterns[i : i + 5] for i in range(0, len(patterns), 5)]

        # Cycle through chunks, then domains
        for chunk_idx, pattern_chunk in enumerate(pattern_chunks):
            for domain_idx, domain in enumerate(domains):
                for pattern in pattern_chunk:
                    if len(combinations) >= target_count:
                        return combinations
                    combinations.append((domain, pattern))

        return combinations[:target_count]

    def _fallback_generation(
        self,
        employee_name: str,
        company_info: CompanyInfo,
        essential_patterns: List[str],
        max_emails: int,
    ) -> EmployeeEmailSearchResult:
        """Fallback email generation without LLM"""
        name_parts = employee_name.lower().strip().split()
        if len(name_parts) < 2:
            return EmployeeEmailSearchResult(
                found_emails=[],
                generated_emails=[],
                confidence_scores=[],
                reasoning="Insufficient name parts",
            )

        first_name = name_parts[0]
        last_name = name_parts[-1]

        combinations = self._create_smart_combinations(
            company_info.domains, essential_patterns, max_emails
        )
        generated_emails = []
        confidences = []

        for i, (domain, pattern) in enumerate(combinations):
            email = self._apply_pattern_fallback(pattern, first_name, last_name, domain)
            generated_emails.append(email)

            # Calculate confidence based on position
            base_confidence = 0.7
            position_penalty = i * 0.05
            confidences.append(max(0.1, base_confidence - position_penalty))

        return EmployeeEmailSearchResult(
            found_emails=[],
            generated_emails=generated_emails,
            confidence_scores=confidences,
            reasoning="Fallback generation using basic name parsing",
        )

    def _apply_pattern_fallback(
        self, pattern: str, first_name: str, last_name: str, domain: str
    ) -> str:
        """Apply pattern to generate email (fallback version)"""
        template = pattern.split("@")[0] if "@" in pattern else pattern

        if "firstname.lastname" in template:
            return f"{first_name}.{last_name}@{domain}"
        elif "firstname_lastname" in template:
            return f"{first_name}_{last_name}@{domain}"
        elif "firstlast" in template:
            return f"{first_name}{last_name}@{domain}"
        elif "f.lastname" in template:
            return f"{first_name[0]}.{last_name}@{domain}"
        elif "flast" in template:
            return f"{first_name[0]}{last_name}@{domain}"
        elif "firstl" in template:
            return f"{first_name}{last_name[0]}@{domain}"
        elif "firstname" in template:
            return f"{first_name}@{domain}"

        return f"{first_name}.{last_name}@{domain}"


# ======================================
# IMPROVED CONTACT FINDER
# ======================================


class ContactFinder(ContactFinder):
    """Improved contact finder with LLM-powered email generation"""

    def __init__(self, config: ContactFinderConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, api_key=config.openai_api_key
        )

        # Initialize search API
        if config.serper_api_key:
            self.search_api = GoogleSerperAPIWrapper()
        else:
            self.search_api = None

        # Initialize smart email generator
        self.email_generator = SmartEmailGenerator(self.llm, self.search_api)

        # Initialize validators
        self.validators = self._create_validators()

    def _create_validators(self) -> List[EmailValidator]:
        """Create validation pipeline"""
        validators = []
        if self.config.enable_mx_validation:
            validators.append(MXValidator())
        if self.config.enable_domain_validation:
            validators.append(DomainValidator())
        if self.config.enable_email_checker:
            validators.append(EmailCheckerValidator())
        if self.config.enable_email_bounceback:
            validators.append(EmailBounceValidator(config=self.config))
        return validators

    def find_company_info(
        self, company_name: str, context: Optional[Dict] = None
    ) -> CompanyInfo:
        """Find company information with structured LLM output"""

        if not self.search_api:
            return CompanyInfo(
                name=company_name,
                domains=[f"{company_name.lower().replace(' ', '')}.com"],
                patterns=self.config.essential_patterns.copy(),
            )

        context_str = ""
        if context:
            context_items = [
                f"{k}: {v}"
                for k, v in context.items()
                if v and k in {"location", "industry", "linkedin", "website"}
            ]
            if context_items:
                context_str = ", ".join(context_items)

        search_queries = [
            f'"{company_name}" {context_str} official website company information'.strip(),
            f'"{company_name}" {context_str} email contact directory employees'.strip(),
        ]

        all_results = ""
        found_emails = []

        for query in search_queries:
            try:
                result = self.search_api.run(query)
                all_results += f" {result}"
                emails = EmailUtils.extract_emails(result)
                found_emails.extend(emails)
                time.sleep(1)
            except Exception as e:
                print(f"Search failed: {e}")

        prompt = f"""
        Analyze the following search results for the company \"{company_name}\"{f" with context: {context_str}" if context_str else ""} and extract structured information.

        IMPORTANT: Only extract domains and information that are directly and unambiguously associated with the intended company. Use the provided context to exclude domains belonging to other companies.

        Search Results:
        {all_results[:2000]}

        Extract:
        1. Official website URL (must match the target company)
        2. Brief company description (1-2 sentences)
        3. Likely email domains (from website and found emails, but only if clearly for the correct company)

        Return the information in the exact JSON format specified.
        """

        try:
            structured_llm = self.llm.with_structured_output(CompanyResearchResult)
            result = structured_llm.invoke(prompt)

            clean_domains = []
            for domain in result.likely_domains:
                clean_domain = EmailUtils.extract_domain(domain)
                if clean_domain:
                    clean_domains.append(clean_domain)

            # Add domains from found emails
            for email in found_emails:
                if "@" in email:
                    domain = email.split("@")[1]
                    if domain not in clean_domains and EmailUtils.is_business_email(
                        email
                    ):
                        clean_domains.append(domain)

            return CompanyInfo(
                name=company_name,
                website=EmailUtils.extract_domain(result.website)
                if result.website
                else "",
                domains=clean_domains[: self.config.max_domains],
                patterns=self.config.essential_patterns.copy(),
                description=result.description,
                found_emails=found_emails,
            )

        except Exception as e:
            print(f"LLM structured output failed: {e}")
            return CompanyInfo(
                name=company_name,
                domains=[f"{company_name.lower().replace(' ', '')}.com"],
                patterns=self.config.essential_patterns.copy(),
                found_emails=found_emails,
            )

    def find_employee_emails(
        self, employee_name: str, company_info: CompanyInfo
    ) -> List[Email]:
        """Generate employee emails using LLM with smart domain/pattern cycling"""

        if not company_info.domains:
            return []

        # Use LLM-powered email generation
        email_result = self.email_generator.generate_emails_with_llm(
            employee_name,
            company_info,
            self.config.essential_patterns,
            self.config.max_emails,
        )

        # Convert to Email objects
        emails = []
        all_emails = email_result.found_emails + email_result.generated_emails

        for i, (email_address, confidence) in enumerate(
            zip(all_emails, email_result.confidence_scores)
        ):
            if "@" in email_address:
                domain = email_address.split("@")[1]
                # Determine pattern (this is approximate for found emails)
                pattern = (
                    "found_publicly"
                    if i < len(email_result.found_emails)
                    else "generated"
                )

                emails.append(
                    Email(
                        address=email_address,
                        confidence=confidence,
                        pattern=pattern,
                        status="unknown",
                        domain=domain,
                    )
                )

        # Apply validation pipeline
        for email in emails:
            for validator in self.validators:
                if email.status in ("unknown", "risky"):
                    status = validator.validate(email.address, company_info)
                    if status is not None and status != "unknown":
                        email.status = status
                        if status == "invalid":
                            email.confidence *= 0.1
                        elif status == "valid":
                            email.confidence = min(1.0, email.confidence * 1.2)
                    if status in ("valid", "invalid"):
                        break

        # Sort by confidence (found emails will naturally be on top due to higher confidence)
        emails.sort(key=lambda x: x.confidence, reverse=True)

        return emails[: self.config.max_emails]


# ======================================
# FACTORY
# ======================================


def create_contact_finder(config: ContactFinderConfig) -> ContactFinder:
    """Factory function"""
    return ContactFinder(config)
