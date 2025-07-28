import os
import logging
import requests
import dns.resolver
from datetime import datetime
from .core import EmailValidator, CompanyInfo
from ..models import EmailBounceCheck, EmployeeEmail


logger = logging.getLogger(__name__)


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
    """Bounce validator with integrated scheduling"""

    @property
    def name(self) -> str:
        return "email_bounce"

    def validate(self, email: str, company_info: CompanyInfo = None) -> str:
        """Queue bounce check without blocking"""
        import asyncio

        try:
            # Queue the bounce check
            loop = asyncio.get_event_loop()
            loop.create_task(self._queue_bounce_check(email))
        except Exception:
            pass  # Fail silently if service not available

        return None  # Don't change status immediately

    async def _queue_bounce_check(self, email: str):
        """Queue a bounce check using EmployeeEmail FK"""
        from .scheduler import get_scheduler

        # Find the EmployeeEmail record
        email_record = await EmployeeEmail.filter(address=email).first()
        if not email_record:
            logger.debug(
                f"No EmployeeEmail record found for {email}, skipping bounce check"
            )
            return

        # Check if already processed or pending
        existing = await EmailBounceCheck.filter(
            email_record=email_record, status__in=["pending", "valid", "invalid"]
        ).first()

        if existing:
            logger.debug(f"Bounce check for {email} already exists: {existing.status}")
            return

        # Create bounce check record
        bounce_check = await EmailBounceCheck.create(
            email_record=email_record, status="pending"
        )

        # Schedule immediate bounce email send
        scheduler = get_scheduler()
        scheduler.schedule_delayed_task(
            "send_bounce_email",
            0,  # Immediate
            f"bounce_send_{bounce_check.id}",
            bounce_check.id,
        )

        logger.info(f"Queued bounce check for {email}")

    @staticmethod
    async def send_bounce_email(bounce_check_id: int):
        """Send bounce email and schedule status check"""
        from .scheduler import get_scheduler

        bounce_check = await EmailBounceCheck.get(id=bounce_check_id).prefetch_related(
            "email_record"
        )
        api_key = os.getenv("EMAIL_BOUNCE_API_KEY")

        if not api_key:
            await EmailBounceValidator._mark_failed(
                bounce_check, "No EMAIL_BOUNCE_API_KEY"
            )
            return

        try:
            # Send bounce email
            response = requests.post(
                "http://194.113.195.63:8000/send-email/",
                headers={"Email-API-Key": api_key},
                json={"recipient_email": bounce_check.email_record.address},
                timeout=10,
            )

            if response.status_code == 200 and response.json().get("mail_sent"):
                # Schedule status check after 5 minutes
                scheduler = get_scheduler()
                scheduler.schedule_delayed_task(
                    "check_bounce_status",
                    5,  # 5 minutes
                    f"bounce_check_{bounce_check.id}",
                    bounce_check.id,
                )
                logger.info(
                    f"Bounce email sent for {bounce_check.email_record.address}"
                )
            else:
                await EmailBounceValidator._mark_failed(
                    bounce_check, f"Send failed: {response.status_code}"
                )

        except Exception as e:
            await EmailBounceValidator._mark_failed(
                bounce_check, f"Send error: {str(e)}"
            )

    @staticmethod
    async def check_bounce_status(bounce_check_id: int):
        """Check bounce status and update results"""
        bounce_check = await EmailBounceCheck.get(id=bounce_check_id).prefetch_related(
            "email_record__employee__company"
        )
        api_key = os.getenv("EMAIL_BOUNCE_API_KEY")

        try:
            response = requests.get(
                "http://194.113.195.63:8000/check-status/",
                headers={"Email-API-Key": api_key},
                params={"recipient_email": bounce_check.email_record.address},
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                status = result.get("status")

                if status in ("valid", "invalid"):
                    # Update bounce check
                    bounce_check.status = status
                    bounce_check.completed_at = datetime.now()
                    await bounce_check.save()

                    # Update email record status
                    bounce_check.email_record.status = status
                    await bounce_check.email_record.save()

                    # Always reevaluate company patterns after status change
                    await EmailBounceValidator._reevaluate_company_patterns(
                        bounce_check.email_record.employee.company
                    )

                    logger.info(
                        f"Bounce check completed: {bounce_check.email_record.address} -> {status}"
                    )
                else:
                    # Retry if status is still pending
                    await EmailBounceValidator._retry_check(bounce_check)
            else:
                await EmailBounceValidator._retry_check(bounce_check)

        except Exception as e:
            await EmailBounceValidator._retry_check(bounce_check, str(e))

    @staticmethod
    async def _reevaluate_company_patterns(company):
        """Reevaluate company patterns from all valid/risky employee emails"""
        # Get all valid/risky emails for this company
        valid_emails = await EmployeeEmail.filter(
            employee__company=company,
            status__in=["valid", "risky"],
            pattern__not="",  # Only emails with patterns
        ).all()

        # Extract unique patterns
        patterns = set()
        for email in valid_emails:
            if email.pattern and email.pattern.strip():
                patterns.add(email.pattern.strip())

        # Update company patterns (overwrite, don't append)
        new_patterns = sorted(list(patterns))  # Convert to sorted list for consistency
        if company.patterns != new_patterns:
            old_patterns = company.patterns.copy()
            company.patterns = new_patterns
            await company.save()

            logger.info(
                f"Updated company {company.name} patterns: {old_patterns} -> {new_patterns}"
            )

    @staticmethod
    async def reevaluate_all_company_patterns():
        """Reevaluate patterns for all companies (maintenance utility)"""
        from ..models import Company

        companies = await Company.all()
        for company in companies:
            await EmailBounceValidator._reevaluate_company_patterns(company)

        logger.info(f"Reevaluated patterns for {len(companies)} companies")

    @staticmethod
    async def _retry_check(bounce_check, error: str = None):
        """Retry bounce status check with exponential backoff"""
        from .scheduler import get_scheduler

        bounce_check.retry_count += 1
        max_retries = 3  # Simple constant

        if bounce_check.retry_count < max_retries:
            # Retry after exponential backoff
            delay_minutes = 2**bounce_check.retry_count
            scheduler = get_scheduler()
            scheduler.schedule_delayed_task(
                "check_bounce_status",
                delay_minutes,
                f"bounce_retry_{bounce_check.id}_{bounce_check.retry_count}",
                bounce_check.id,
            )
            await bounce_check.save()
            logger.info(
                f"Retrying bounce check for {bounce_check.email_record.address} in {delay_minutes}m"
            )
        else:
            await EmailBounceValidator._mark_failed(
                bounce_check, f"Max retries exceeded. Last error: {error}"
            )

    @staticmethod
    async def _mark_failed(bounce_check, error: str):
        """Mark bounce check as failed"""
        bounce_check.status = "failed"
        bounce_check.completed_at = datetime.now()
        await bounce_check.save()
        logger.warning(
            f"Bounce check failed for {bounce_check.email_record.address}: {error}"
        )
