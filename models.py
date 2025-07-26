from tortoise.models import Model
from tortoise import fields


class Company(Model):
    """Company model for caching company information"""

    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=255, unique=True)
    website = fields.CharField(max_length=255, null=True)
    description = fields.TextField(null=True)

    # JSON fields for lists
    domains = fields.JSONField(default=list)
    patterns = fields.JSONField(default=list)
    found_emails = fields.JSONField(default=list)

    # Metadata
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    # Reverse relation to employees
    employees: fields.ReverseRelation["Employee"]

    class Meta:
        table = "companies"

    def __str__(self):
        return self.name

    def to_company_info(self):
        """Convert to Pydantic CompanyInfo model"""
        from services.core import CompanyInfo

        return CompanyInfo(
            name=self.name,
            website=self.website or "",
            domains=self.domains or [],
            patterns=self.patterns or [],
            description=self.description or "",
            found_emails=self.found_emails or [],
        )


class Employee(Model):
    """Employee model for caching employee information and emails"""

    id = fields.IntField(primary_key=True)
    name = fields.CharField(max_length=255)

    # Foreign key to company
    company = fields.ForeignKeyField("models.Company", related_name="employees")

    # Metadata
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    # Reverse relation to emails
    emails: fields.ReverseRelation["EmployeeEmail"]

    class Meta:
        table = "employees"
        unique_together = (("name", "company"),)  # Unique combination

    def __str__(self):
        return f"{self.name} ({self.company.name})"


class EmployeeEmail(Model):
    """Generated/found emails for employees"""

    id = fields.IntField(primary_key=True)
    address = fields.CharField(max_length=255)
    confidence = fields.FloatField()
    pattern_type = fields.CharField(max_length=50, default="generated")
    status = fields.CharField(
        max_length=20, default="unknown"
    )  # unknown, valid, invalid, risky
    domain = fields.CharField(max_length=255, default="")

    # Foreign key to employee
    employee = fields.ForeignKeyField("models.Employee", related_name="emails")

    # Metadata
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "employee_emails"
        unique_together = (("address", "employee"),)  # Unique email per employee

    def __str__(self):
        return f"{self.address} ({self.employee.name})"

    def to_email(self):
        """Convert to Pydantic Email model"""
        from services.core import Email

        return Email(
            address=self.address,
            confidence=self.confidence,
            pattern_type=self.pattern_type,
            status=self.status,
            domain=self.domain,
        )


# Database configuration
def get_tortoise_config(db_url: str = "sqlite://db.sqlite3", testing: bool = False):
    """Generate Tortoise ORM configuration"""
    from tortoise import generate_config

    return generate_config(
        db_url,
        app_modules={"models": ["models"]},
        testing=testing,
        connection_label="models",
    )


async def init_db():
    """Initialize database (legacy - use lifespan instead)"""
    from tortoise import Tortoise

    config = get_tortoise_config()
    await Tortoise.init(config=config)
    await Tortoise.generate_schemas()


async def close_db():
    """Close database connections (legacy - use lifespan instead)"""
    from tortoise import Tortoise

    await Tortoise.close_connections()
