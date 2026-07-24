import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import os
from dotenv import load_dotenv
from scraper.playwright_scraper import run_scraper
from routers.jobs_search import router as jobs_search_router
from routers.ml_autofill import router as ml_autofill_router
from routers.career import router as career_router

load_dotenv()

app = FastAPI(title="Auto-Apply Job Scraper Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(jobs_search_router)
app.include_router(ml_autofill_router)
app.include_router(career_router)

scheduler = BackgroundScheduler()

def run_scraper_sync():
    import sys
    import asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(run_scraper())

def periodic_scraper():
    print("Running scheduled job scraper...")
    run_scraper_sync()

@app.on_event("startup")
def startup_event():
    # Schedule the job to run twice a day: at 00:00 and 12:00
    scheduler.add_job(periodic_scraper, 'cron', hour='0,12')
    scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

def run_scraper_sync():
    import sys
    import asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(run_scraper())

@app.post("/api/scrape/run")
def trigger_scrape(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_scraper_sync)
        return {"message": "Scraping job has been triggered in the background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
