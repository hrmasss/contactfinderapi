# Services module exports
from .core import (
    ContactFinderConfig,
    CompanyInfo,
    Email,
    ContactResult,
    CompanyResearchResult
)

from .finder import create_contact_finder, ContactFinder

__all__ = [
    "ContactFinderConfig",
    "CompanyInfo", 
    "Email",
    "ContactResult",
    "CompanyResearchResult",
    "create_contact_finder",
    "ContactFinder"
]
