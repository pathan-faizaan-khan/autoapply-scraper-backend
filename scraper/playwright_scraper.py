import os
from datetime import datetime
from playwright.async_api import async_playwright
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import json
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from dotenv import load_dotenv
from utils.google_jobs_scraper import scrape_google_jobs

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def run_scraper():
    print("Starting Playwright scraper for personalized Google Jobs...")
    
    async with AsyncSessionLocal() as session:
        # Fetch active campaigns to get user preferences
        try:
            result = await session.execute(text("SELECT target_roles, company_types, location_pref FROM outreach_campaigns WHERE status = 'active'"))
            campaigns = result.fetchall()
        except Exception as e:
            print(f"Failed to fetch campaigns (table might not exist): {e}")
            campaigns = []
        
    if not campaigns:
        print("No active campaigns found. Defaulting to a general startup search.")
        campaigns = [
            ('["Software Engineer"]', '["startup"]', 'Remote')
        ]
        
    all_jobs = []
    
    for camp in campaigns:
        try:
            roles = json.loads(camp[0]) if camp[0] else ["Software Engineer"]
        except Exception:
            roles = ["Software Engineer"]
            
        try:
            company_types = json.loads(camp[1]) if camp[1] else ["startup"]
        except Exception:
            company_types = ["startup"]
            
        location = camp[2] or "Remote"
        
        for role in roles[:3]:  # Limit to avoid too many searches
            for c_type in company_types[:2]:
                print(f"Scraping Google Jobs for: {role} at {c_type} in {location}")
                
                try:
                    jobs = await scrape_google_jobs(
                        query=role,
                        location=location,
                        company_type=c_type,
                        num_results=10
                    )
                    
                    for j in jobs:
                        # Prevent duplicates based on job URL
                        if j.get("job_url"):
                            all_jobs.append({
                                "title": j.get("title", role),
                                "companyName": j.get("company_name", "Unknown"),
                                "jobUrl": j.get("job_url"),
                                "location": j.get("location", location),
                                "description": j.get("description", ""),
                                "launchDate": datetime.utcnow(),
                                "appliedPeoples": 0
                            })
                except Exception as e:
                    print(f"Error scraping for {role}: {e}")
                    
    if all_jobs:
        await save_jobs_to_db(all_jobs)
    else:
        print("No jobs found during this scrape run.")

async def save_jobs_to_db(jobs):
    async with AsyncSessionLocal() as session:
        for job in jobs:
            try:
                query = text("""
                    INSERT INTO scraped_jobs (title, company_name, job_url, location, description, launch_date, applied_peoples, created_at, updated_at)
                    VALUES (:title, :company_name, :job_url, :location, :description, :launch_date, :applied_peoples, NOW(), NOW())
                    ON CONFLICT (job_url) DO NOTHING
                """)
                await session.execute(query, {
                    "title": job["title"],
                    "company_name": job["companyName"],
                    "job_url": job["jobUrl"],
                    "location": job["location"],
                    "description": job["description"],
                    "launch_date": job["launchDate"],
                    "applied_peoples": job["appliedPeoples"],
                })
            except Exception as e:
                print(f"Error saving job: {e}")
        
        await session.commit()
        print(f"Successfully saved {len(jobs)} jobs to the database.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_scraper())
