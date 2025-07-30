import os
import csv
import asyncio
import aiohttp
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ========================
# CONFIGURATION
# ========================

# Number of test subjects to process (set to None for all)
MAX_TEST_COUNT = 3

# FastAPI server URL (must be running)
API_BASE_URL = "http://localhost:8000"

# Input CSV file
INPUT_CSV = "reports/finder_test_results_20250721_121132.csv"

# Output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BEFORE_CSV = f"reports/api_bounce_test_before_{TIMESTAMP}.csv"
AFTER_CSV = f"reports/api_bounce_test_after_{TIMESTAMP}.csv"

# Test timing
REQUEST_DELAY = 1.0  # Seconds between requests
BOUNCE_WAIT_TIME = 720  # 12 minutes wait for bounce checks (8 min + 4 min buffer)
STATUS_CHECK_INTERVAL = 10  # Check every 10 seconds

# ========================
# API CLIENT
# ========================


class ContactFinderAPIClient:
    """Async HTTP client for ContactFinder API"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def find_employee(
        self,
        company_name: str,
        employee_name: str,
        context: dict = None,
        status_level: str = "risky",
        use_cache: bool = True,
        force_refresh: bool = False,
    ):
        """Call /employee endpoint"""
        url = f"{self.base_url}/employee"
        payload = {
            "company_name": company_name,
            "employee_name": employee_name,
            "context": context or {},
            "status_level": status_level,
            "use_cache": use_cache,
            "force_refresh": force_refresh,
        }

        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"API error {response.status}: {error_text}")


# ========================
# DATABASE MONITORING
# ========================


async def check_api_health(base_url: str) -> bool:
    """Check if FastAPI server is running"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/docs") as response:
                return response.status == 200
    except Exception:
        return False


async def get_scheduled_jobs_count():
    """Get count of scheduled bounce jobs"""
    try:
        import sqlite3

        conn = sqlite3.connect("jobs.sqlite3")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM apscheduler_jobs")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"⚠️  Could not check jobs: {e}")
        return None


async def get_bounce_status_from_db():
    """Get bounce check status directly from database"""
    try:
        from app.models import EmailBounceCheck, get_tortoise_config
        from tortoise import Tortoise

        # Connect to same DB as API
        db_url = os.getenv("DATABASE_URL", "sqlite://db.sqlite3")
        db_config = get_tortoise_config(db_url)

        if not Tortoise._inited:
            await Tortoise.init(config=db_config)

        pending = await EmailBounceCheck.filter(status="pending").count()
        completed = await EmailBounceCheck.filter(
            status__in=["valid", "invalid", "failed"]
        ).count()

        return pending, completed

    except Exception as e:
        print(f"⚠️  Could not check DB status: {e}")
        return None, None


async def get_email_bounce_status(email_address):
    """Get bounce status for a specific email"""
    try:
        from app.models import EmailBounceCheck, EmployeeEmail

        # Get the email record
        email_record = await EmployeeEmail.filter(address=email_address).first()
        if not email_record:
            return None  # Email not in database

        # Get bounce check status
        bounce_check = await EmailBounceCheck.filter(email_record=email_record).first()
        if not bounce_check:
            return None  # No bounce check initiated

        return {
            "email": email_address,
            "bounce_status": bounce_check.status,
            "email_status": email_record.status,
            "completed": bounce_check.status in ["valid", "invalid", "failed"],
        }
    except Exception as e:
        print(f"⚠️  Could not check status for {email_address}: {e}")
        return None


async def update_csv_with_bounce_result(
    csv_file, fieldnames, email_address, new_status, old_status
):
    """Update CSV file in real-time when a bounce result comes in"""
    try:
        # Read current CSV content
        rows = []
        with open(csv_file, "r", newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)

        # Update rows that contain this email
        updated = False
        for row in rows:
            if email_address in row.get("employee_emails", ""):
                # Update the specific email status in the employee_emails field
                original_emails = row["employee_emails"]
                if original_emails:
                    updated_email_parts = []
                    email_parts = (
                        original_emails.split(";")
                        if ";" in original_emails
                        else [original_emails]
                    )

                    for email_part in email_parts:
                        if email_address in email_part and "(" in email_part:
                            # Replace the status in this email part
                            addr = email_part.split("(")[0].strip()
                            if addr == email_address:
                                conf_part = email_part.split("(")[1]
                                confidence = conf_part.split(" - ")[0]
                                updated_email_parts.append(
                                    f"{addr} ({confidence} - {new_status})"
                                )

                                # Update bounce_changes field
                                current_changes = row.get("bounce_changes", "")
                                change_text = (
                                    f"{email_address}: {old_status}→{new_status}"
                                )
                                if current_changes in ["pending", "no_changes", ""]:
                                    row["bounce_changes"] = change_text
                                else:
                                    row["bounce_changes"] = (
                                        f"{current_changes}; {change_text}"
                                    )
                                updated = True
                            else:
                                updated_email_parts.append(email_part)
                        else:
                            updated_email_parts.append(email_part)

                    row["employee_emails"] = ";".join(updated_email_parts)

        # Write updated CSV back to file
        if updated:
            with open(csv_file, "w", newline="", encoding="utf-8") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"📝 Updated CSV with: {email_address} → {new_status}")

    except Exception as e:
        print(f"⚠️  Could not update CSV for {email_address}: {e}")


# ========================
# TEST LOGIC
# ========================


def parse_context(row):
    """Parse context from CSV row"""
    return {"location": row.get("context_location", "")}


def extract_emails_from_response(response_data):
    """Extract email info from API response"""
    if not response_data or not response_data.get("data"):
        return []

    emails = response_data["data"].get("emails", [])
    return [
        (email["address"], email["confidence"], email["status"]) for email in emails
    ]


def format_emails_for_csv(emails):
    """Format emails for CSV output"""
    return ";".join(f"{addr} ({conf:.2f} - {status})" for addr, conf, status in emails)


async def run_api_bounce_test():
    """Main test function using API endpoints with real-time CSV updates"""
    print("🚀 Starting API-based bounce validation test")
    print(f"📊 Processing {MAX_TEST_COUNT or 'ALL'} test subjects")
    print(f"🌐 API Base URL: {API_BASE_URL}")
    print(f"📁 Input: {INPUT_CSV}")
    print(f"📁 Output: {BEFORE_CSV}, {AFTER_CSV}")
    print()

    # Check API health
    print("🔍 Checking API server...")
    if not await check_api_health(API_BASE_URL):
        print("❌ FastAPI server is not running!")
        print("   Please start the server: python main.py")
        print(f"   Server should be available at: {API_BASE_URL}")
        return

    print("✅ FastAPI server is running")

    # Read test data
    test_data = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames) + ["api_response", "bounce_changes"]

        for i, row in enumerate(reader):
            if MAX_TEST_COUNT and i >= MAX_TEST_COUNT:
                break
            test_data.append(row)

    print(f"📋 Loaded {len(test_data)} test subjects")

    # Phase 1: Send API requests
    print("\n🔍 Phase 1: Sending API requests...")

    async with ContactFinderAPIClient(API_BASE_URL) as client:
        before_results = []
        all_email_addresses = set()
        email_original_statuses = {}  # Track original statuses for CSV updates

        for i, row in enumerate(test_data, 1):
            company_name = row["company_name"]
            employee_name = row["employee_name"]
            context = parse_context(row)

            print(f"[{i}/{len(test_data)}] API call: {employee_name} at {company_name}")

            try:
                # Call API endpoint
                response = await client.find_employee(
                    company_name=company_name,
                    employee_name=employee_name,
                    context=context,
                    status_level="risky",
                    force_refresh=True,  # Ensure fresh lookup
                )

                # Extract emails from response
                emails = extract_emails_from_response(response)
                for addr, _, status in emails:
                    all_email_addresses.add(addr)
                    email_original_statuses[addr] = status  # Track original status

                print(f"  📧 Found {len(emails)} emails")
                if emails:
                    for addr, conf, status in emails:
                        print(f"    • {addr} ({conf:.2f}, {status})")

                # Prepare CSV row
                output_row = dict(row)
                if response.get("data"):
                    data = response["data"]
                    company = data.get("company", {})

                    output_row["company_website"] = company.get("website", "")
                    output_row["email_domains"] = ";".join(company.get("domains", []))
                    output_row["email_patterns"] = ";".join(company.get("patterns", []))
                    output_row["employee_emails"] = format_emails_for_csv(emails)
                else:
                    output_row["company_website"] = ""
                    output_row["email_domains"] = ""
                    output_row["email_patterns"] = ""
                    output_row["employee_emails"] = ""

                output_row["seamless_email"] = row.get("seamless_email", "")
                output_row["api_response"] = "success"
                output_row["bounce_changes"] = "pending"

                before_results.append(output_row)

            except Exception as e:
                print(f"  ❌ API error: {e}")

                # Error row
                output_row = dict(row)
                output_row["company_website"] = ""
                output_row["email_domains"] = ""
                output_row["email_patterns"] = ""
                output_row["employee_emails"] = ""
                output_row["seamless_email"] = row.get("seamless_email", "")
                output_row["api_response"] = f"error: {str(e)}"
                output_row["bounce_changes"] = "error"

                before_results.append(output_row)

            # Rate limiting
            if i < len(test_data):
                await asyncio.sleep(REQUEST_DELAY)

    # Write initial results
    print("\n📄 Writing initial results...")
    with open(BEFORE_CSV, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(before_results)

    # Create the after CSV with initial pending status
    with open(AFTER_CSV, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        # Write initial results with "pending" bounce status
        for row in before_results:
            initial_row = dict(row)
            initial_row["bounce_changes"] = "pending"
            writer.writerow(initial_row)

    print(f"✅ Before results saved: {BEFORE_CSV}")
    print(f"✅ After results initialized: {AFTER_CSV} (will update in real-time)")
    print(f"📧 Total unique emails found: {len(all_email_addresses)}")

    # Check if any jobs were scheduled
    jobs_count = await get_scheduled_jobs_count()
    if jobs_count is not None:
        print(f"📅 Scheduled bounce jobs: {jobs_count}")
        if jobs_count == 0:
            print(
                "⚠️  WARNING: No bounce jobs were scheduled! This indicates the bounce validator didn't trigger."
            )
        else:
            print("✅ Bounce jobs were scheduled successfully")

    # Also check bounce records
    pending, completed = await get_bounce_status_from_db()
    if pending is not None:
        print(f"🗃️  Bounce records: {pending} pending, {completed} completed")

    # Phase 2: Real-time bounce validation monitoring with CSV updates
    print("\n⚡ Phase 2: Real-time bounce validation monitoring...")

    if all_email_addresses:
        print(
            f"🔍 Monitoring {len(all_email_addresses)} emails for bounce completion..."
        )
        print(f"⏱️  Max wait time: {BOUNCE_WAIT_TIME} seconds")
        print(f"📝 Results updating in real-time: {AFTER_CSV}")

        start_time = asyncio.get_event_loop().time()
        completed_emails = set()
        total_changes = 0

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            # Check each email that hasn't completed yet
            pending_emails = [
                email for email in all_email_addresses if email not in completed_emails
            ]

            if not pending_emails:
                print("✅ All target emails have been processed!")
                break

            # Check status of pending emails
            newly_completed = []
            for email in pending_emails:
                status = await get_email_bounce_status(email)
                if status and status["completed"]:
                    completed_emails.add(email)
                    newly_completed.append(email)

                    # Get original status and new status
                    old_status = email_original_statuses.get(email, "unknown")
                    new_status = status["email_status"]

                    print(
                        f"✅ {email}: {status['bounce_status']} (email status: {old_status} → {new_status})"
                    )

                    # Update CSV immediately if status changed
                    if old_status != new_status:
                        await update_csv_with_bounce_result(
                            AFTER_CSV, fieldnames, email, new_status, old_status
                        )
                        total_changes += 1
                        print(f"📝 CSV updated with change #{total_changes}")

            # Show progress
            total_completed = len(completed_emails)
            total_pending = len(pending_emails) - len(newly_completed)
            print(
                f"📊 Progress: {total_completed}/{len(all_email_addresses)} completed, {total_pending} pending, {total_changes} changes (elapsed: {elapsed:.1f}s)"
            )

            # Check timeout
            if elapsed > BOUNCE_WAIT_TIME:
                print(f"⏰ Timeout reached ({BOUNCE_WAIT_TIME}s)")
                print(
                    f"📋 Final status: {total_completed} completed, {total_pending} still pending"
                )
                break

            # Wait before next check
            await asyncio.sleep(STATUS_CHECK_INTERVAL)

        completed_count = len(completed_emails)
        print(
            f"🎯 Final summary: {completed_count}/{len(all_email_addresses)} emails completed, {total_changes} status changes detected"
        )
    else:
        print("⚠️  No emails to monitor")
        completed_count = 0
        total_changes = 0

    # Final summary
    print("\n🎉 API bounce test completed!")
    print(f"📊 Processed {len(before_results)} companies")
    print(f"📧 Found {len(all_email_addresses)} unique emails")
    print(f"🔄 Detected {total_changes} email status changes")
    print(f"📁 Before: {BEFORE_CSV}")
    print(f"📁 After:  {AFTER_CSV} (updated in real-time)")


if __name__ == "__main__":
    asyncio.run(run_api_bounce_test())
