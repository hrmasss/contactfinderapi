from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
from abc import ABC, abstractmethod


# ======================================
# CORE MODELS
# ======================================


class Email(BaseModel):
    address: str
    confidence: float = Field(ge=0.0, le=1.0)
    pattern_type: str = "generated"
    status: str = "unknown"  # unknown, valid, invalid, risky
    domain: str = ""


class CompanyInfo(BaseModel):
    name: str
    website: str = ""
    domains: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    description: str = ""
    found_emails: List[str] = Field(default_factory=list)


class ContactResult(BaseModel):
    company: CompanyInfo
    employee_name: Optional[str] = None
    emails: List[Email] = Field(default_factory=list)
    cache_used: bool = False


# ======================================
# LLM STRUCTURED OUTPUTS
# ======================================


class CompanyResearchResult(BaseModel):
    """Structured output for company research"""

    company_name: str
    website: str = ""
    description: str = ""
    likely_domains: List[str] = Field(default_factory=list)
    found_patterns: List[str] = Field(default_factory=list)


class EmailGenerationResult(BaseModel):
    """Structured output for email generation"""

    emails: List[Email]
    total_generated: int
    patterns_used: List[str]


# ======================================
# VALIDATION INTERFACE (Pluggable)
# ======================================


class EmailValidator(ABC):
    """Base validator interface"""

    @abstractmethod
    def validate(self, email: str, company_info: CompanyInfo = None) -> str:
        """Return: valid, invalid, unknown, risky"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ======================================
# CONFIGURATION
# ======================================


class ContactFinderConfig(BaseModel):
    """Configuration using Pydantic Settings"""

    model_config = ConfigDict(env_prefix="CONTACTFINDER_")

    serper_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    email_verifier_key: Optional[str] = None

    # Feature toggles
    enable_mx_validation: bool = True
    enable_domain_validation: bool = True
    enable_email_checker: bool = True
    enable_email_bounceback: bool = True

    # Generation settings
    max_emails: int = 15
    max_domains: int = 5

    # Essential patterns - will be merged with publicly found ones
    essential_patterns: List[str] = Field(
        default_factory=lambda: [
            "first.last@domain.com",
            "first@domain.com",
            "firstlast@domain.com",
            "f.last@domain.com",
            "flast@domain.com",
            "firstl@domain.com",
            "first_last@domain.com",
            "f_last@domain.com",
            "first.last@domain.com",
            "first_last@domain.com",
            "firstlast@domain.com",
            "last@domain.com",
            "lastf@domain.com",
            "fl@domain.com",
            "first.l@domain.com",
        ]
    )


# ======================================
# CORE CONTACT FINDER INTERFACE
# ======================================


class ContactFinder(ABC):
    """Abstract base for contact finders"""

    @abstractmethod
    def find_company_info(
        self, company_name: str, context: Optional[Dict] = None
    ) -> CompanyInfo:
        """Find company information only - for caching"""
        pass

    @abstractmethod
    def find_employee_emails(
        self, employee_name: str, company_info: CompanyInfo
    ) -> List[Email]:
        """Generate employee emails using company info - for cache-enabled searches"""
        pass

    def find_contacts(
        self,
        company_name: str,
        employee_name: Optional[str] = None,
        context: Optional[Dict] = None,
        company_cache: Optional[CompanyInfo] = None,
    ) -> ContactResult:
        """Full contact finding flow with optional cache"""

        # Use cache or find company info
        if company_cache:
            company_info = company_cache
            cache_used = True
        else:
            company_info = self.find_company_info(company_name, context)
            cache_used = False

        # Find employee emails if specified
        emails = []
        if employee_name:
            emails = self.find_employee_emails(employee_name, company_info)

        return ContactResult(
            company=company_info,
            employee_name=employee_name,
            emails=emails,
            cache_used=cache_used,
        )
