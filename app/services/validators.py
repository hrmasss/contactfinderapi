import os
import requests
import dns.resolver
from .core import EmailValidator, CompanyInfo


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
        import threading

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
