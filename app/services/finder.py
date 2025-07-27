import re
import time
from langchain_openai import ChatOpenAI
from typing import List, Optional, Dict, Tuple
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

    found_emails: List[str] = Field(description="Publicly found email addresses")
    found_email_patterns: List[str] = Field(
        description="Inferred patterns from found emails"
    )
    generated_emails: List[str] = Field(
        description="Generated email addresses based on patterns"
    )
    generated_email_patterns: List[str] = Field(
        description="Patterns used to generate each email"
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
    """Email utilities with validation and extraction"""

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

    @classmethod
    def clean_email(cls, email: str) -> str:
        """Clean and validate email using email-validator"""
        try:
            email = email.strip().lower()
            if not email or "@" not in email:
                return ""
            validated = validate_email(email, check_deliverability=False)
            return validated.email
        except EmailNotValidError:
            return ""

    @classmethod
    def extract_emails(cls, text: str) -> List[str]:
        """Extract valid business emails from text"""
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        emails = re.findall(email_pattern, text)

        clean_emails = [
            cleaned
            for email in emails
            if (cleaned := cls.clean_email(email)) and cls.is_business_email(cleaned)
        ]

        return list(set(clean_emails))

    @classmethod
    def is_business_email(cls, email: str) -> bool:
        """Check if email is from a business domain"""
        if "@" not in email:
            return False
        domain = email.split("@")[1]
        return domain not in cls.EXCLUDED_DOMAINS

    @classmethod
    def extract_domain(cls, url: str) -> str:
        """Extract clean domain from URL"""
        if not url:
            return ""

        domain = re.sub(r"^https?://", "", url.strip().lower())
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0].split("?")[0].split("#")[0]

        return domain if domain and "." in domain and len(domain) >= 4 else ""


# ======================================
# EMAIL GENERATION ENGINE
# ======================================


class EmailGenerator:
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
        if not self.search_api or not company_info.domains:
            return []

        search_queries = [
            f'"{employee_name}" {company_info.name} email contact',
            f'"{employee_name}" @{company_info.domains[0]}',
            f"{employee_name} {company_info.name} directory",
        ]

        found_emails = []
        for query in search_queries:
            try:
                result = self.search_api.run(query)
                emails = EmailUtils.extract_emails(result)
                # Filter for company domains only
                company_emails = [
                    email
                    for email in emails
                    if email.split("@")[1] in company_info.domains
                ]
                found_emails.extend(company_emails)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Search failed for '{query}': {e}")

        return list(set(found_emails))

    def generate_emails_with_llm(
        self,
        employee_name: str,
        company_info: CompanyInfo,
        essential_patterns: List[str],
        max_emails: int,
    ) -> EmployeeEmailSearchResult:
        """Generate emails using LLM with smart domain/pattern cycling"""

        found_emails = self.search_for_employee_email(employee_name, company_info)
        remaining_slots = max_emails - len(found_emails)

        if remaining_slots <= 0:
            # Still need to analyze found email patterns
            found_patterns = self._analyze_found_email_patterns(
                found_emails, essential_patterns
            )
            return EmployeeEmailSearchResult(
                found_emails=found_emails[:max_emails],
                found_email_patterns=found_patterns[:max_emails],
                generated_emails=[],
                generated_email_patterns=[],
                confidence_scores=[0.95] * min(len(found_emails), max_emails),
                reasoning="Sufficient emails found publicly",
            )

        domain_pattern_combinations = self._create_smart_combinations(
            company_info.domains, essential_patterns, remaining_slots
        )

        prompt = self._build_generation_prompt(
            employee_name,
            company_info,
            essential_patterns,
            domain_pattern_combinations,
            found_emails,
        )

        try:
            structured_llm = self.llm.with_structured_output(EmployeeEmailSearchResult)
            result = structured_llm.invoke(prompt)

            # Combine and validate results
            all_emails = found_emails + result.generated_emails
            all_patterns = result.found_email_patterns + result.generated_email_patterns
            all_confidences = [0.95] * len(found_emails) + result.confidence_scores

            # Ensure we don't exceed max_emails
            if len(all_emails) > max_emails:
                all_emails = all_emails[:max_emails]
                all_patterns = all_patterns[:max_emails]
                all_confidences = all_confidences[:max_emails]

            return EmployeeEmailSearchResult(
                found_emails=found_emails,
                found_email_patterns=result.found_email_patterns,
                generated_emails=result.generated_emails[:remaining_slots],
                generated_email_patterns=result.generated_email_patterns[
                    :remaining_slots
                ],
                confidence_scores=all_confidences,
                reasoning=result.reasoning,
            )

        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._fallback_generation(
                employee_name,
                company_info,
                essential_patterns,
                remaining_slots,
                found_emails,
            )

    def _build_generation_prompt(
        self,
        employee_name: str,
        company_info: CompanyInfo,
        essential_patterns: List[str],
        combinations: List[Tuple[str, str]],
        found_emails: List[str],
    ) -> str:
        """Build the LLM prompt for email generation"""
        return f"""
        Generate professional email addresses for "{employee_name}" at {company_info.name}.
        
        DOMAINS (priority order): {company_info.domains}
        ESSENTIAL_PATTERNS: {essential_patterns}
        
        TASKS:
        1. For FOUND EMAILS ({found_emails}):
           - Analyze each email to determine which pattern it follows
           - Return the actual pattern (e.g., "firstname.lastname@domain.com", "f.lastname@domain.com")
        
        2. GENERATE EXACTLY {len(combinations)} NEW EMAILS using these combinations:
           {chr(10).join([f"- {domain} with {pattern}" for domain, pattern in combinations])}
        
        INSTRUCTIONS:
        - Parse "{employee_name}" intelligently (handle complex names, titles, etc.)
        - Apply patterns correctly:
          * first.last@domain.com
          * firstlast@domain.com
          * first@domain.com
          * flast@domain.com
          * firstl@domain.com
          * f.last@domain.com
          * first_last@domain.com  
        - Assign confidence scores (0.1-1.0) based on domain preference and pattern commonality
        - For found emails, assign confidence 0.95
        - Return the EXACT pattern used for each email (both found and generated)
        
        IMPORTANT: 
        - found_email_patterns should contain the actual pattern each found email follows
        - generated_email_patterns should contain the exact pattern used to generate each email
        - Patterns should match the format "first.last@domain.com" etc., not arbitrary values
        """

    def _create_smart_combinations(
        self, domains: List[str], patterns: List[str], target_count: int
    ) -> List[Tuple[str, str]]:
        """Create smart domain-pattern combinations with cycling logic"""
        if not domains or not patterns or target_count <= 0:
            return []

        combinations = []
        pattern_chunks = [patterns[i : i + 5] for i in range(0, len(patterns), 5)]

        for chunk in pattern_chunks:
            for domain in domains:
                for pattern in chunk:
                    if len(combinations) >= target_count:
                        return combinations
                    combinations.append((domain, pattern))

        return combinations

    def _fallback_generation(
        self,
        employee_name: str,
        company_info: CompanyInfo,
        essential_patterns: List[str],
        count: int,
        found_emails: Optional[List[str]] = None,
    ) -> EmployeeEmailSearchResult:
        """Fallback email generation without LLM"""
        if found_emails is None:
            found_emails = []

        found_patterns = self._analyze_found_email_patterns(
            found_emails, essential_patterns
        )

        name_parts = employee_name.lower().strip().split()
        if len(name_parts) < 2:
            return EmployeeEmailSearchResult(
                found_emails=found_emails,
                found_email_patterns=found_patterns,
                generated_emails=[],
                generated_email_patterns=[],
                confidence_scores=[0.95] * len(found_emails),
                reasoning="Insufficient name parts",
            )

        first_name, last_name = name_parts[0], name_parts[-1]
        combinations = self._create_smart_combinations(
            company_info.domains, essential_patterns, count
        )

        generated_emails = []
        generated_patterns = []
        confidences = []

        for i, (domain, pattern) in enumerate(combinations):
            email = self._apply_pattern(pattern, first_name, last_name, domain)
            generated_emails.append(email)
            generated_patterns.append(pattern)
            confidences.append(max(0.1, 0.7 - i * 0.05))

        # Combine all confidences
        all_confidences = [0.95] * len(found_emails) + confidences

        return EmployeeEmailSearchResult(
            found_emails=found_emails,
            found_email_patterns=found_patterns,
            generated_emails=generated_emails,
            generated_email_patterns=generated_patterns,
            confidence_scores=all_confidences,
            reasoning="Fallback generation with basic name parsing",
        )

    def _apply_pattern(
        self, pattern: str, first_name: str, last_name: str, domain: str
    ) -> str:
        """Apply pattern to generate email"""
        pattern_map = {
            "firstname.lastname": f"{first_name}.{last_name}",
            "firstname_lastname": f"{first_name}_{last_name}",
            "firstlast": f"{first_name}{last_name}",
            "f.lastname": f"{first_name[0]}.{last_name}",
            "flast": f"{first_name[0]}{last_name}",
            "firstl": f"{first_name}{last_name[0]}",
            "firstname": first_name,
        }

        template = pattern.split("@")[0] if "@" in pattern else pattern
        local_part = pattern_map.get(template, f"{first_name}.{last_name}")

        return f"{local_part}@{domain}"


# ======================================
# IMPROVED CONTACT FINDER
# ======================================


class ContactFinder(ContactFinder):
    """Improved contact finder with LLM-powered email generation"""

    def __init__(self, config: ContactFinderConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini", temperature=0, api_key=config.openai_api_key
        )
        self.search_api = GoogleSerperAPIWrapper() if config.serper_api_key else None
        self.email_generator = EmailGenerator(self.llm, self.search_api)
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

        context_str = self._build_context_string(context)
        search_results, found_emails = self._perform_company_search(
            company_name, context_str
        )

        return self._extract_company_info(company_name, context_str, found_emails)

    def _build_context_string(self, context: Optional[Dict]) -> str:
        """Build context string from provided context"""
        if not context:
            return ""

        relevant_keys = {"location", "industry", "linkedin", "website"}
        context_items = [
            f"{k}: {v}" for k, v in context.items() if v and k in relevant_keys
        ]

        return ", ".join(context_items)

    def _perform_company_search(
        self, company_name: str, context_str: str
    ) -> Tuple[str, List[str]]:
        """Perform search queries for company information"""
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
                found_emails.extend(EmailUtils.extract_emails(result))
                time.sleep(1)
            except Exception as e:
                print(f"Search failed: {e}")

        return all_results, found_emails

    def _extract_company_info(
        self,
        company_name: str,
        context_str: str,
        found_emails: List[str],
    ) -> CompanyInfo:
        """Extract structured company information using LLM, allowing valid subdomains."""

        # Collect all unique domains (including subdomains) from found emails
        candidate_domains = list(
            {email.split("@")[1].lower() for email in found_emails if "@" in email}
        )

        prompt = f"""
        You are an expert at identifying the official email domains for a company, including subdomains only if there is clear evidence they are used for employee or personal (named) emails.

        Company: "{company_name}"{f" with context: {context_str}" if context_str else ""}

        Candidate email domains (from found emails): {candidate_domains}
        Found Emails: {found_emails}

        INSTRUCTIONS:
        - Only select domains (including subdomains) that are clearly and officially associated with the target company, based on the email addresses and context.
        - Include a subdomain (e.g., us.domain.com, marketing.domain.com) ONLY if at least one employee or personal (named) email uses it (e.g., jane.doe@us.domain.com). Do NOT include subdomains that are only used for generic, support, info, or marketing addresses unless there is evidence of employee usage.
        - Ignore domains that are not clearly for the target company, even if they appear in scraped emails.
        - If you are unsure, prefer to return fewer domains.
        - Also extract the official website URL and a brief company description (1-2 sentences).

        Return:
        1. Official website URL (must match target company)
        2. Brief company description
        3. List of likely email domains (including subdomains if appropriate)
        4. For each selected domain, a short justification for why it is associated with the target company.
        """

        try:
            structured_llm = self.llm.with_structured_output(CompanyResearchResult)
            result = structured_llm.invoke(prompt)

            # Use the domains as returned by the LLM
            clean_domains = [
                domain.strip().lower()
                for domain in result.likely_domains
                if domain and "." in domain
            ]

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
            print(f"LLM extraction failed: {e}")
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

        # Generate emails using LLM
        email_result = self.email_generator.generate_emails_with_llm(
            employee_name,
            company_info,
            self.config.essential_patterns,
            self.config.max_emails,
        )

        # Convert to Email objects
        emails = self._create_email_objects(email_result)

        # Apply validation pipeline
        self._validate_emails(emails, company_info)

        # Update company patterns with only successful ones
        self._update_company_patterns(emails, company_info)

        # Sort by confidence and return
        emails.sort(key=lambda x: x.confidence, reverse=True)
        return emails[: self.config.max_emails]

    def _update_company_patterns(
        self, emails: List[Email], company_info: CompanyInfo
    ) -> None:
        """Update company patterns with only those that resulted in valid/risky emails"""
        successful_patterns = {
            email.pattern
            for email in emails
            if email.status in ("valid", "risky")
            and email.pattern != "unknown@domain.com"
        }

        # Only keep patterns that were successful
        company_info.patterns = list(successful_patterns)

    def _create_email_objects(
        self, email_result: EmployeeEmailSearchResult
    ) -> List[Email]:
        """Convert email results to unique Email objects, keeping highest confidence per address"""
        all_emails = email_result.found_emails + email_result.generated_emails
        all_patterns = (
            email_result.found_email_patterns + email_result.generated_email_patterns
        )

        all_confidences = email_result.confidence_scores

        email_map = {}
        for email_address, pattern, confidence in zip(
            all_emails, all_patterns, all_confidences
        ):
            if "@" in email_address:
                domain = email_address.split("@")[1]
                # If duplicate, keep the one with higher confidence
                if (
                    email_address not in email_map
                    or confidence > email_map[email_address].confidence
                ):
                    email_map[email_address] = Email(
                        address=email_address,
                        confidence=confidence,
                        pattern=pattern,
                        status="unknown",
                        domain=domain,
                    )

        return list(email_map.values())

    def _validate_emails(self, emails: List[Email], company_info: CompanyInfo) -> None:
        """Apply validation pipeline to emails"""
        for email in emails:
            for validator in self.validators:
                if email.status in ("unknown", "risky"):
                    status = validator.validate(email.address, company_info)
                    if status and status != "unknown":
                        email.status = status
                        # Adjust confidence based on validation
                        if status == "invalid":
                            email.confidence *= 0.1
                        elif status == "valid":
                            email.confidence = min(1.0, email.confidence * 1.2)
                    if status in ("valid", "invalid"):
                        break


# ======================================
# FACTORY
# ======================================


def create_contact_finder(config: ContactFinderConfig) -> ContactFinder:
    """Factory function"""
    return ContactFinder(config)
