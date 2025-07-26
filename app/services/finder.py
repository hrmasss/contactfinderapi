import threading
import os
import re
import time
import requests
import dns.resolver
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from email_validator import validate_email, EmailNotValidError
from langchain_community.utilities import GoogleSerperAPIWrapper
from .core import (
    ContactFinder,
    CompanyInfo,
    Email,
    CompanyResearchResult,
    ContactFinderConfig,
    EmailValidator,
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
            # Basic cleaning first
            email = email.strip().lower()
            if not email or "@" not in email:
                return ""

            # Use email-validator
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

        # Remove protocol and www
        domain = re.sub(r"^https?://", "", url.strip().lower())
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0].split("?")[0].split("#")[0]

        # Validate domain format
        if not domain or "." not in domain or len(domain) < 4:
            return ""

        return domain


# ======================================
# PATTERN ENGINE
# ======================================


class PatternEngine:
    """Email pattern detection and application"""

    @staticmethod
    def extract_patterns_from_text(text: str) -> List[str]:
        """Extract email patterns from search results"""
        patterns = []
        text_lower = text.lower()

        # Look for explicit pattern mentions
        pattern_indicators = {
            "firstname.lastname@": "firstname.lastname@domain.com",
            "first.last@": "firstname.lastname@domain.com",
            "firstname_lastname@": "firstname_lastname@domain.com",
            "first_last@": "firstname_lastname@domain.com",
            "firstlast@": "firstlast@domain.com",
            "f.lastname@": "f.lastname@domain.com",
            "flast@": "flast@domain.com",
            "firstname@": "firstname@domain.com",
        }

        for indicator, pattern in pattern_indicators.items():
            if indicator in text_lower:
                patterns.append(pattern)

        return patterns

    @staticmethod
    def extract_patterns_from_emails(emails: List[str]) -> List[str]:
        """Infer patterns from actual email addresses"""
        patterns = []

        for email in emails:
            if "@" not in email:
                continue

            local = email.split("@")[0].lower()

            # Skip system emails
            if any(
                skip in local
                for skip in ["info", "contact", "support", "admin", "noreply"]
            ):
                continue

            # Analyze structure
            if "." in local and len(local.split(".")) == 2:
                parts = local.split(".")
                if len(parts[0]) == 1 and len(parts[1]) > 2:
                    patterns.append("f.lastname@domain.com")
                elif len(parts[0]) > 2 and len(parts[1]) > 2:
                    patterns.append("firstname.lastname@domain.com")
            elif "_" in local and len(local.split("_")) == 2:
                patterns.append("firstname_lastname@domain.com")
            elif local.isalpha() and 4 <= len(local) <= 8:
                patterns.append("firstname@domain.com")
            elif local.isalpha() and len(local) > 8:
                patterns.append("firstlast@domain.com")

        return list(set(patterns))

    @staticmethod
    def apply_pattern(
        pattern: str, first_name: str, last_name: str, domain: str
    ) -> str:
        """Apply pattern to generate email"""
        template = pattern.split("@")[0]

        if not last_name or len(last_name) <= 1:
            return f"{first_name}@{domain}"

        # Pattern mapping
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
# VALIDATORS
# ======================================


class MXValidator(EmailValidator):
    """MX record validation"""

    @property
    def name(self) -> str:
        return "mx_record"

    def validate(self, email: str, company_info: CompanyInfo = None) -> str:
        if "@" not in email:
            return "invalid"

        domain = email.split("@")[1]
        try:
            dns.resolver.resolve(domain, "MX")
            return "unknown"
        except dns.resolver.NoAnswer:
            try:
                dns.resolver.resolve(domain, "A")
                return "unknown"
            except Exception:
                return "invalid"
        except Exception:
            return "invalid"


class DomainValidator(EmailValidator):
    """Domain ownership validation"""

    @property
    def name(self) -> str:
        return "domain"

    def validate(self, email: str, company_info: CompanyInfo = None) -> str:
        if not company_info or "@" not in email:
            return "unknown"

        domain = email.split("@")[1]

        # Check against known company domains
        if domain in company_info.domains:
            return "unknown"

        # Check against company website
        if company_info.website and domain in company_info.website:
            return "unknown"

        # Simple company name matching
        company_clean = company_info.name.lower().replace(" ", "").replace("-", "")
        if company_clean in domain:
            return "unknown"

        return "invalid"


class EmailCheckerValidator(EmailValidator):
    """Validator using emails-checker.net API"""

    @property
    def name(self) -> str:
        return "emails_checker"

    def validate(self, email: str, company_info: CompanyInfo = None) -> str:
        api_key = os.getenv("EMAIL_CHECKER_API_KEY")
        if not api_key or not email or "@" not in email:
            return "unknown"
        try:
            resp = requests.get(
                "https://api.emails-checker.net/check",
                params={"access_key": api_key, "email": email},
                timeout=5,
            )
            if resp.status_code != 200:
                return "unknown"
            data = resp.json()
            if not data.get("success") or "response" not in data:
                return "unknown"
            result = data["response"].get("result", "unknown")
            # Map API result to our status
            if result == "deliverable":
                return "valid"
            if result == "undeliverable":
                return "invalid"
            if result == "risky":
                return "risky"
            return "unknown"
        except Exception:
            return "unknown"


class EmailBounceValidator(EmailValidator):
    """Validator that triggers a bounce check email, but does not block or change status immediately."""

    @property
    def name(self) -> str:
        return "email_bounce"

    def __init__(self, config=None):
        import logging

        self.callback = getattr(config, "bounce_callback", None) if config else None
        self.logger = logging.getLogger("EmailBounceValidator")

    def validate(self, email: str, company_info: CompanyInfo = None) -> str:
        from os import getenv
        import requests
        import time

        def send_and_check(email, company_info):
            api_key = getenv("EMAIL_BOUNCE_API_KEY")
            if not api_key or not email or "@" not in email:
                return
            headers = {"Email-API-Key": api_key}
            data = {"recipient_email": email}
            try:
                resp = requests.post(
                    "http://194.113.195.63:8000/send-email/",
                    headers=headers,
                    json=data,
                    timeout=5,
                )
                if resp.status_code == 200 and resp.json().get("mail_sent"):
                    time.sleep(300)
                    status_resp = requests.get(
                        "http://194.113.195.63:8000/check-status/",
                        headers=headers,
                        params={"recipient_email": email},
                        timeout=5,
                    )
                    if status_resp.status_code == 200:
                        result = status_resp.json()
                        status = result.get("status")
                        if status in ("valid", "invalid"):
                            if self.callback:
                                self.callback(email, status, company_info)
                            else:
                                self.logger.warning(
                                    "No bounce_callback provided in config; skipping additional steps."
                                )
            except Exception:
                pass

        threading.Thread(target=send_and_check, args=(email, company_info)).start()
        return None


# ======================================
# CONTACT FINDER
# ======================================


class ContactFinder(ContactFinder):
    """Contact finder"""

    def __init__(self, config: ContactFinderConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, api_key=config.openai_api_key
        )

        # Add search agent capability
        if config.serper_api_key:
            self.search_api = GoogleSerperAPIWrapper()
            try:
                from langchain.agents import create_react_agent
                from langchain.tools import Tool

                class SearchTools:
                    def __init__(self, search_api):
                        self.search_api = search_api

                    def get_tools(self):
                        return [
                            Tool(
                                name="Google Search",
                                func=self.search_api.run,
                                description="Useful for searching the web for company information, domains, and contact details.",
                            )
                        ]

                self.search_tools = SearchTools(self.search_api)
                self.agent = create_react_agent(self.llm, self.search_tools.get_tools())
            except Exception:
                self.agent = None
        else:
            self.search_api = None
            self.agent = None

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
        """Find company information with structured LLM output and agent search ability"""

        if not (self.search_api or self.agent):
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
            f'"{company_name}" {context_str} email pattern format structure'.strip(),
        ]

        all_results = ""
        found_emails = []

        # Use agent if available, otherwise fallback to direct search
        for query in search_queries:
            if self.agent:
                try:
                    result = self.agent.invoke(query)
                except Exception:
                    result = self.search_api.run(query) if self.search_api else ""
            elif self.search_api:
                result = self.search_api.run(query)
            else:
                result = ""
            all_results += f" {result}"
            emails = EmailUtils.extract_emails(result)
            found_emails.extend(emails)
            time.sleep(1)  # Rate limiting

        # Use structured output from LLM with improved prompt
        prompt = f"""
        Analyze the following search results for the company \"{company_name}\"{f" with context: {context_str}" if context_str else ""} and extract structured information.

        IMPORTANT: Only extract domains and information that are directly and unambiguously associated with the intended company. If there are multiple companies with similar names, use the provided context (such as location, industry, website, or LinkedIn) to exclude domains belonging to other companies. Do NOT include domains that are not clearly linked to the target company.

        Search Results:
        {all_results[:2000]}

        Extract:
        1. Official website URL (must match the target company)
        2. Brief company description (1-2 sentences)
        3. Likely email domains (from website and found emails, but only if they are clearly for the correct company)

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
        """Generate employee emails"""

        if not company_info.domains:
            return []

        name_parts = employee_name.lower().strip().split()
        if len(name_parts) < 2:
            return []

        first_name = name_parts[0]
        last_name = name_parts[-1]

        all_patterns = self.config.essential_patterns

        emails = []
        seen = set()

        for i, domain in enumerate(company_info.domains):
            for j, pattern in enumerate(all_patterns):
                email_address = PatternEngine.apply_pattern(
                    pattern, first_name, last_name, domain
                )
                if email_address in seen:
                    continue
                seen.add(email_address)

                base_confidence = 0.5
                domain_penalty = i * 0.1
                pattern_penalty = j * 0.02
                confidence = max(
                    0.1, base_confidence - domain_penalty - pattern_penalty
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
                            email.confidence = min(1.0, email.confidence * 1.1)
                    if status in ("valid", "invalid"):
                        break

        # After validation, update company_info.patterns
        company_info.patterns = list(
            {
                pattern
                for email, pattern in zip(
                    emails, all_patterns * len(company_info.domains)
                )
                if email.status in ("valid", "risky", "unknown")
            }
        )

        emails.sort(key=lambda x: x.confidence, reverse=True)
        return emails[: self.config.max_emails]


# ======================================
# FACTORY
# ======================================


def create_contact_finder(config: ContactFinderConfig) -> ContactFinder:
    """Factory function"""
    return ContactFinder(config)
