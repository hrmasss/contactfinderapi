from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from .services import CompanyInfo, ContactResult
from .models import Company, Employee, EmployeeEmail
from fastapi import FastAPI, HTTPException, BackgroundTasks


# ======================================
# REQUEST/RESPONSE MODELS
# ======================================


class CompanyRequest(BaseModel):
    company_name: str = Field(..., description="Company name to search for")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context"
    )
    use_cache: bool = Field(True, description="Whether to use cached data")
    force_refresh: bool = Field(False, description="Force refresh cached data")


class EmployeeRequest(CompanyRequest):
    employee_name: str = Field(..., description="Employee name")


class CompanyResponse(BaseModel):
    """Response model for company info"""

    success: bool
    data: Optional[CompanyInfo] = None
    message: str = ""
    cache_hit: bool = False


class EmployeeResponse(BaseModel):
    """Response model for employee contacts"""

    success: bool
    data: Optional[ContactResult] = None
    message: str = ""
    cache_hit: bool = False


# ======================================
# CACHE FUNCTIONS
# ======================================


async def get_cached_company(company_name: str) -> Optional[Company]:
    """Get company from cache"""
    return await Company.filter(name__iexact=company_name).first()


async def save_company_to_cache(company_info: CompanyInfo) -> Company:
    """Save company info to cache"""
    company, created = await Company.get_or_create(
        name=company_info.name,
        defaults={
            "website": company_info.website,
            "description": company_info.description,
            "domains": company_info.domains,
            "patterns": company_info.patterns,
            "found_emails": company_info.found_emails,
        },
    )

    if not created:
        company.website = company_info.website
        company.description = company_info.description
        company.domains = company_info.domains
        company.patterns = company_info.patterns
        company.found_emails = company_info.found_emails
        await company.save()

    return company


async def get_cached_employee_emails(
    company: Company, employee_name: str
) -> Optional[Employee]:
    """Get employee emails from cache"""
    return (
        await Employee.filter(company=company, name__iexact=employee_name)
        .prefetch_related("emails")
        .first()
    )


async def save_employee_emails_to_cache(
    company: Company, employee_name: str, emails: list
) -> Employee:
    """Save employee emails to cache"""
    employee, created = await Employee.get_or_create(
        company=company, name=employee_name
    )

    if not created:
        await EmployeeEmail.filter(employee=employee).delete()

    for email in emails:
        await EmployeeEmail.create(
            employee=employee,
            address=email.address,
            confidence=email.confidence,
            pattern_type=email.pattern_type,
            status=email.status,
            domain=email.domain,
        )

    return employee


# ======================================
# API FACTORY
# ======================================


def create_api(finder) -> FastAPI:
    """Create FastAPI app with finder instance"""

    app = FastAPI(
        title="ContactFinder API",
        description="AI-powered contact finding with intelligent caching",
        version="1.0.0",
    )

    @app.post("/company", response_model=CompanyResponse)
    async def find_company(request: CompanyRequest, background_tasks: BackgroundTasks):
        """Find company information only"""
        try:
            company_name = request.company_name.strip()
            cache_hit = False

            # Check cache first
            if request.use_cache and not request.force_refresh:
                cached_company = await get_cached_company(company_name)
                if cached_company:
                    cache_hit = True
                    company_info = cached_company.to_company_info()
                    return CompanyResponse(
                        success=True,
                        data=company_info,
                        message="Company info retrieved from cache",
                        cache_hit=cache_hit,
                    )

            # Find fresh company info
            company_info = finder.find_company_info(company_name, request.context)

            # Save to cache in background
            if request.use_cache:
                background_tasks.add_task(save_company_to_cache, company_info)

            return CompanyResponse(
                success=True,
                data=company_info,
                message="Company info found",
                cache_hit=cache_hit,
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error finding company info: {str(e)}"
            )

    @app.post("/employee", response_model=EmployeeResponse)
    async def find_employee(
        request: EmployeeRequest, background_tasks: BackgroundTasks
    ):
        """Find employee contacts (company + employee)"""
        try:
            company_name = request.company_name.strip()
            employee_name = request.employee_name.strip()
            cache_hit = False

            # Get company info (cached or fresh)
            company_cache = None
            if request.use_cache and not request.force_refresh:
                cached_company = await get_cached_company(company_name)
                if cached_company:
                    company_cache = cached_company.to_company_info()
                    cache_hit = True

                    # Check for cached employee emails
                    cached_employee = await get_cached_employee_emails(
                        cached_company, employee_name
                    )
                    if cached_employee:
                        cached_emails = [
                            email.to_email() for email in cached_employee.emails
                        ]

                        result = ContactResult(
                            company=company_cache,
                            employee_name=employee_name,
                            emails=cached_emails,
                            cache_used=True,
                        )

                        return EmployeeResponse(
                            success=True,
                            data=result,
                            message="Employee contacts retrieved from cache",
                            cache_hit=True,
                        )

            # Find contacts (using cache if available)
            result = finder.find_contacts(
                company_name=company_name,
                employee_name=employee_name,
                context=request.context,
                company_cache=company_cache,
            )

            # Save to cache in background
            if request.use_cache and result.emails:
                if not company_cache:
                    company_db = await save_company_to_cache(result.company)
                else:
                    company_db = await get_cached_company(company_name)

                if company_db:
                    background_tasks.add_task(
                        save_employee_emails_to_cache,
                        company_db,
                        employee_name,
                        result.emails,
                    )

            return EmployeeResponse(
                success=True,
                data=result,
                message="Employee contacts found",
                cache_hit=cache_hit,
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error finding employee contacts: {str(e)}"
            )

    return app
