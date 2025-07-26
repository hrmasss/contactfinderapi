import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from tortoise.contrib.fastapi import RegisterTortoise
from fastapi import FastAPI, HTTPException, BackgroundTasks
from models import Company, Employee, EmployeeEmail, get_tortoise_config
from services import (
    ContactFinderConfig,
    create_contact_finder,
    CompanyInfo,
    ContactResult,
)


# Load environment variables
load_dotenv()


# ======================================
# REQUEST/RESPONSE MODELS
# ======================================


class ContactRequest(BaseModel):
    """Request model for contact finding"""

    company_name: str = Field(..., description="Company name to search for")
    employee_name: Optional[str] = Field(None, description="Employee name (optional)")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context"
    )
    use_cache: bool = Field(True, description="Whether to use cached data")
    force_refresh: bool = Field(False, description="Force refresh cached data")


class ContactResponse(BaseModel):
    """Response model for contact finding"""

    success: bool
    data: Optional[ContactResult] = None
    message: str = ""
    cache_hit: bool = False


class CompanyResponse(BaseModel):
    """Response model for company info only"""

    success: bool
    data: Optional[CompanyInfo] = None
    message: str = ""
    cache_hit: bool = False


# ======================================
# DATABASE LIFESPAN
# ======================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage database lifecycle using RegisterTortoise"""
    # Get database URL from environment
    db_url = os.getenv("DATABASE_URL", "sqlite://db.sqlite3")

    # Configure database
    config = get_tortoise_config(db_url)

    async with RegisterTortoise(
        app=app,
        config=config,
        generate_schemas=True,
        add_exception_handlers=True,
    ):
        # Database connected and schemas created
        yield
        # Database connections will be closed automatically


# ======================================
# FASTAPI APP
# ======================================

app = FastAPI(
    title="ContactFinder API",
    description="AI-powered contact finding with intelligent caching",
    version="1.0.0",
    lifespan=lifespan,
)

# Initialize contact finder
config = ContactFinderConfig(
    serper_api_key=os.getenv("SERPER_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    email_verifier_key=os.getenv("EMAIL_VERIFIER_KEY"),
    enable_mx_validation=True,
    enable_domain_validation=True,
    max_emails=15,
)

finder = create_contact_finder(config)


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

    # Update if not created
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

    # Clear existing emails if updating
    if not created:
        await EmployeeEmail.filter(employee=employee).delete()

    # Save new emails
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


async def refresh_company_cache_background(
    company_name: str, context: Optional[Dict] = None
):
    """Background task to refresh company cache"""
    try:
        # Find fresh company info
        fresh_company_info = finder.find_company_info(company_name, context)

        # Save to cache
        await save_company_to_cache(fresh_company_info)
        print(f"✅ Refreshed cache for company: {company_name}")

    except Exception as e:
        print(f"❌ Failed to refresh cache for {company_name}: {e}")


# ======================================
# API ENDPOINTS
# ======================================


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ContactFinder API",
        "version": "1.0.0",
        "endpoints": {
            "/find-contacts": "POST - Find employee contacts",
            "/company-info": "POST - Get company information only",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ContactFinder API"}


@app.post("/company-info", response_model=CompanyResponse)
async def get_company_info(request: ContactRequest, background_tasks: BackgroundTasks):
    """Get company information only (for caching)"""
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


@app.post("/find-contacts", response_model=ContactResponse)
async def find_contacts(request: ContactRequest, background_tasks: BackgroundTasks):
    """Find employee contacts with intelligent caching"""
    try:
        company_name = request.company_name.strip()
        employee_name = request.employee_name.strip() if request.employee_name else None
        cache_hit = False

        # Get company info (cached or fresh)
        company_cache = None
        if request.use_cache and not request.force_refresh:
            cached_company = await get_cached_company(company_name)
            if cached_company:
                company_cache = cached_company.to_company_info()
                cache_hit = True

                # If employee specified, check for cached emails
                if employee_name:
                    cached_employee = await get_cached_employee_emails(
                        cached_company, employee_name
                    )
                    if cached_employee:
                        # Return cached result
                        cached_emails = [
                            email.to_email() for email in cached_employee.emails
                        ]

                        result = ContactResult(
                            company=company_cache,
                            employee_name=employee_name,
                            emails=cached_emails,
                            cache_used=True,
                        )

                        return ContactResponse(
                            success=True,
                            data=result,
                            message="Contact info retrieved from cache",
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
        if request.use_cache and employee_name and result.emails:
            # First ensure company is cached
            if not company_cache:
                company_db = await save_company_to_cache(result.company)
            else:
                company_db = await get_cached_company(company_name)

            # Save employee emails
            if company_db:
                background_tasks.add_task(
                    save_employee_emails_to_cache,
                    company_db,
                    employee_name,
                    result.emails,
                )

        # Schedule cache refresh if using old cache
        if cache_hit and request.use_cache:
            background_tasks.add_task(
                refresh_company_cache_background, company_name, request.context
            )

        return ContactResponse(
            success=True,
            data=result,
            message="Contacts found successfully",
            cache_hit=cache_hit,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding contacts: {str(e)}")


@app.delete("/cache/company/{company_name}")
async def clear_company_cache(company_name: str):
    """Clear cache for a specific company"""
    try:
        company = await get_cached_company(company_name)
        if company:
            # Delete employees and their emails first
            employees = await Employee.filter(company=company).prefetch_related(
                "emails"
            )
            for employee in employees:
                await EmployeeEmail.filter(employee=employee).delete()
            await Employee.filter(company=company).delete()

            # Delete company
            await company.delete()

            return {"message": f"Cache cleared for company: {company_name}"}
        else:
            return {"message": f"No cache found for company: {company_name}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    try:
        company_count = await Company.all().count()
        employee_count = await Employee.all().count()
        email_count = await EmployeeEmail.all().count()

        return {
            "companies": company_count,
            "employees": employee_count,
            "emails": email_count,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting cache stats: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
