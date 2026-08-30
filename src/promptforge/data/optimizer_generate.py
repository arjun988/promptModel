from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import pandas as pd

from promptforge.data.generate import clamp
from promptforge.optimizer_chat import (
    SYSTEM_PROMPT,
    build_optimizer_messages,
    format_optimizer_input,
)

# Re-export for backwards compatibility
__all__ = [
    "SYSTEM_PROMPT",
    "CURATED_PAIRS",
    "build_fallback_prompt",
    "build_optimizer_messages",
    "format_optimizer_input",
    "generate_optimizer_dataset",
    "generate_optimizer_example",
    "row_to_analysis",
]

# Light variation pools — never replace the topic; only enrich the strong side.
_AUDIENCES = [
    "beginners",
    "experienced developers",
    "startup founders",
    "students",
    "product managers",
    "general users",
]
_CONTEXTS = [
    "This is a weekend prototype.",
    "This is for a university project.",
    "This is for a small production MVP.",
    "This will run locally first.",
    "This is for a portfolio demo.",
]
_CODING_STACK = [
    "Use Python and FastAPI.",
    "Use TypeScript and React.",
    "Use Python and Flask.",
    "Prefer a simple, readable stack.",
]
_OUTPUTS = [
    "Return a clear Markdown plan.",
    "Return numbered steps.",
    "Return JSON where structure helps.",
    "Include short examples.",
]

# High-quality curated weak → strong seeds (topic must stay identical).
# Each tuple: (weak, domain, strong_template) where {audience}, {context}, {stack}, {output} are optional.
CURATED_PAIRS: list[tuple[str, str, str]] = [
    # --- coding / apps (topic-specific) ---
    (
        "Make an app about social media like facebook and stuff",
        "coding",
        "Build a social media app similar to Facebook for {audience}.\n"
        "{context}\n\n"
        "Core features:\n"
        "- User profiles and friend connections\n"
        "- News feed with posts, likes, and comments\n"
        "- Basic notifications\n\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Keep the first version simple and usable\n"
        "- Include error handling and clear project structure\n\n"
        "Return:\n"
        "1. Feature list\n"
        "2. Project structure\n"
        "3. Core implementation outline\n\n"
        "{output}",
    ),
    (
        "Make a Facebook clone app",
        "coding",
        "Build a Facebook-like social networking app for {audience}.\n"
        "{context}\n\n"
        "Include profiles, a feed, posts, likes, comments, and friend requests.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Focus on core social features first\n\n"
        "{output}",
    ),
    (
        "Make an app.",
        "coding",
        "Build a simple mobile or web app for {audience}.\n"
        "{context}\n\n"
        "Before coding, clarify the app purpose, main screens, and user flow.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Include project structure and setup steps\n\n"
        "{output}",
    ),
    (
        "Build me a website for a startup.",
        "coding",
        "Build a responsive marketing website for a startup for {audience}.\n"
        "{context}\n\n"
        "Pages: home, features, pricing, about, contact.\n"
        "Requirements:\n"
        "- Clean modern UI\n"
        "- Mobile responsive layout\n"
        "- {stack}\n\n"
        "{output}",
    ),
    (
        "Create a website.",
        "coding",
        "Create a responsive website for {audience}.\n"
        "{context}\n\n"
        "Requirements:\n"
        "- Clear navigation and readable layout\n"
        "- Mobile-friendly design\n"
        "- {stack}\n\n"
        "{output}",
    ),
    (
        "Make a fitness tracking app",
        "coding",
        "Build a fitness tracking app for {audience}.\n"
        "{context}\n\n"
        "Features: log workouts, track steps/calories, show weekly progress.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Simple dashboard and history view\n\n"
        "{output}",
    ),
    (
        "Build a chat app",
        "coding",
        "Build a real-time chat application for {audience}.\n"
        "{context}\n\n"
        "Features: 1:1 messaging, online status, message history.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Keep auth and messaging reliable\n\n"
        "{output}",
    ),
    (
        "Make an e-commerce app",
        "coding",
        "Build an e-commerce app for {audience}.\n"
        "{context}\n\n"
        "Features: product catalog, cart, checkout, order history.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Clear product and cart flows\n\n"
        "{output}",
    ),
    (
        "Create a notes app",
        "coding",
        "Build a notes-taking app for {audience}.\n"
        "{context}\n\n"
        "Features: create/edit/delete notes, search, simple folders or tags.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Persist notes locally or with a simple backend\n\n"
        "{output}",
    ),
    (
        "Make a todo list app",
        "coding",
        "Build a todo list app for {audience}.\n"
        "{context}\n\n"
        "Features: add/complete/delete tasks, due dates, basic filtering.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Clean UI and simple state management\n\n"
        "{output}",
    ),
    (
        "Build a weather app",
        "coding",
        "Build a weather app for {audience}.\n"
        "{context}\n\n"
        "Features: current conditions, forecast, location search.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Handle API errors gracefully\n\n"
        "{output}",
    ),
    (
        "Make a recipe app",
        "coding",
        "Build a recipe app for {audience}.\n"
        "{context}\n\n"
        "Features: browse recipes, ingredients list, step-by-step cooking instructions, favorites.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Keep the UI easy to follow while cooking\n\n"
        "{output}",
    ),
    (
        "Create a blog platform",
        "coding",
        "Build a simple blog platform for {audience}.\n"
        "{context}\n\n"
        "Features: create posts, publish drafts, comments, tags.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Include auth for writers\n\n"
        "{output}",
    ),
    (
        "Make a Python API.",
        "coding",
        "Build a Python REST API for {audience}.\n"
        "{context}\n\n"
        "Requirements:\n"
        "- Use FastAPI or Flask\n"
        "- Include CRUD endpoints, validation, and error handling\n"
        "- Add example requests\n\n"
        "{output}",
    ),
    (
        "Make a Python API for beginners.",
        "coding",
        "Build a beginner-friendly Python REST API for {audience}.\n"
        "{context}\n\n"
        "Requirements:\n"
        "- Simple CRUD endpoints\n"
        "- Clear comments and setup instructions\n"
        "- Avoid unnecessary complexity\n\n"
        "{output}",
    ),
    (
        "Build me a project.",
        "coding",
        "Build a small software project for {audience}.\n"
        "{context}\n\n"
        "Choose one clear goal, define features, and implement a working MVP.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Include README and setup steps\n\n"
        "{output}",
    ),
    (
        "Build something.",
        "coding",
        "Build a useful software prototype for {audience}.\n"
        "{context}\n\n"
        "Pick one problem, define the MVP scope, and implement it end-to-end.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Keep the scope small and complete\n\n"
        "{output}",
    ),
    (
        "Make a dashboard for sales data",
        "coding",
        "Build a sales data dashboard for {audience}.\n"
        "{context}\n\n"
        "Show KPIs, trends over time, and top products/regions.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Include charts and filters\n\n"
        "{output}",
    ),
    (
        "Create an inventory management system",
        "coding",
        "Build an inventory management system for {audience}.\n"
        "{context}\n\n"
        "Features: add products, track stock levels, low-stock alerts, basic reports.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Keep data model simple and reliable\n\n"
        "{output}",
    ),
    (
        "Make a habit tracker",
        "coding",
        "Build a habit tracker app for {audience}.\n"
        "{context}\n\n"
        "Features: daily check-ins, streaks, reminders, weekly summary.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Simple UX that encourages consistency\n\n"
        "{output}",
    ),
    (
        "Build a URL shortener",
        "coding",
        "Build a URL shortener service for {audience}.\n"
        "{context}\n\n"
        "Features: shorten links, redirect, click counts, basic admin view.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Include API and simple UI if useful\n\n"
        "{output}",
    ),
    (
        "Make a file upload service",
        "coding",
        "Build a file upload service for {audience}.\n"
        "{context}\n\n"
        "Features: upload, list, download, delete files with size limits.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Validate file types and handle errors\n\n"
        "{output}",
    ),
    (
        "Create a quiz app",
        "coding",
        "Build a quiz app for {audience}.\n"
        "{context}\n\n"
        "Features: multiple-choice questions, scoring, results summary.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Easy to add new questions\n\n"
        "{output}",
    ),
    (
        "Make a booking app for a clinic",
        "coding",
        "Build a clinic appointment booking app for {audience}.\n"
        "{context}\n\n"
        "Features: choose doctor/slot, book appointment, cancel/reschedule, confirmation.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Prevent double-booking\n\n"
        "{output}",
    ),
    (
        "Build a music playlist app",
        "coding",
        "Build a music playlist app for {audience}.\n"
        "{context}\n\n"
        "Features: create playlists, add/remove songs, reorder tracks, play queue.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Keep the player UI simple\n\n"
        "{output}",
    ),
    (
        "Make an expense tracker",
        "coding",
        "Build an expense tracker for {audience}.\n"
        "{context}\n\n"
        "Features: add expenses, categories, monthly totals, simple charts.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Persist data and support basic filters\n\n"
        "{output}",
    ),
    (
        "Create a job board website",
        "coding",
        "Build a job board website for {audience}.\n"
        "{context}\n\n"
        "Features: post jobs, search/filter listings, apply with a short form.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Clear listing and application flow\n\n"
        "{output}",
    ),
    (
        "Make a Reddit-like app",
        "coding",
        "Build a Reddit-like community app for {audience}.\n"
        "{context}\n\n"
        "Features: communities/subreddits, posts, comments, upvotes.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Keep moderation features minimal for v1\n\n"
        "{output}",
    ),
    (
        "Build a Twitter clone",
        "coding",
        "Build a Twitter/X-like microblogging app for {audience}.\n"
        "{context}\n\n"
        "Features: short posts, follow users, timeline feed, likes/replies.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Focus on feed and posting first\n\n"
        "{output}",
    ),
    (
        "Make an Instagram-like photo app",
        "coding",
        "Build an Instagram-like photo sharing app for {audience}.\n"
        "{context}\n\n"
        "Features: upload photos, captions, feed, likes, comments.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Optimize image display for mobile\n\n"
        "{output}",
    ),
    (
        "Create a CLI tool for renaming files",
        "coding",
        "Build a CLI tool that batch-renames files for {audience}.\n"
        "{context}\n\n"
        "Requirements:\n"
        "- Support dry-run mode\n"
        "- Safe conflict handling\n"
        "- Clear help text and examples\n\n"
        "{output}",
    ),
    (
        "Write a script to scrape product prices",
        "coding",
        "Write a Python script that scrapes product prices for {audience}.\n"
        "{context}\n\n"
        "Requirements:\n"
        "- Respect robots.txt / rate limits where possible\n"
        "- Save results to CSV or JSON\n"
        "- Handle missing fields and network errors\n\n"
        "{output}",
    ),
    (
        "Make a REST API for a bookstore",
        "coding",
        "Build a REST API for a bookstore for {audience}.\n"
        "{context}\n\n"
        "Endpoints for books, authors, inventory, and basic search.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Validation and clear error responses\n\n"
        "{output}",
    ),
    (
        "Build a kanban board app",
        "coding",
        "Build a kanban board app for {audience}.\n"
        "{context}\n\n"
        "Features: columns (todo/doing/done), drag cards, edit titles, persist state.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Keep interactions snappy and simple\n\n"
        "{output}",
    ),
    (
        "Make a password generator tool",
        "coding",
        "Build a password generator tool for {audience}.\n"
        "{context}\n\n"
        "Features: length options, character sets, copy button, strength hint.\n"
        "Requirements:\n"
        "- Cryptographically secure randomness\n"
        "- No insecure defaults\n\n"
        "{output}",
    ),
    (
        "Create a markdown editor",
        "coding",
        "Build a Markdown editor for {audience}.\n"
        "{context}\n\n"
        "Features: live preview, save/load files, basic formatting shortcuts.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Clean split editor/preview layout\n\n"
        "{output}",
    ),
    # --- writing ---
    (
        "Write something about AI.",
        "writing",
        "Write a clear article about artificial intelligence for {audience}.\n"
        "{context}\n\n"
        "Cover what AI is, common use cases, benefits, and risks.\n"
        "Requirements:\n"
        "- Use headings and short paragraphs\n"
        "- Include 2 concrete examples\n"
        "- Keep tone professional and accessible\n\n"
        "{output}",
    ),
    (
        "Write something good.",
        "writing",
        "Write a high-quality short article for {audience}.\n"
        "{context}\n\n"
        "Pick one focused topic, state the main point early, and support it with examples.\n"
        "Requirements:\n"
        "- Strong structure and concise wording\n"
        "- No fluff\n\n"
        "{output}",
    ),
    (
        "Write a blog post about remote work",
        "writing",
        "Write a blog post about remote work for {audience}.\n"
        "{context}\n\n"
        "Cover benefits, challenges, and practical tips for staying productive.\n"
        "Requirements:\n"
        "- Clear headings\n"
        "- Actionable advice\n\n"
        "{output}",
    ),
    (
        "Write an email asking for a refund",
        "writing",
        "Write a polite refund-request email for {audience}.\n"
        "{context}\n\n"
        "Include order details placeholder, reason, and a clear ask.\n"
        "Requirements:\n"
        "- Professional and concise tone\n"
        "- Subject line + body\n\n"
        "{output}",
    ),
    (
        "Write a product description for wireless earbuds",
        "writing",
        "Write a product description for wireless earbuds for {audience}.\n"
        "{context}\n\n"
        "Highlight battery life, sound quality, comfort, and key selling points.\n"
        "Requirements:\n"
        "- Persuasive but honest\n"
        "- Short scannable bullets\n\n"
        "{output}",
    ),
    (
        "Write LinkedIn post about launching a startup",
        "writing",
        "Write a LinkedIn post about launching a startup for {audience}.\n"
        "{context}\n\n"
        "Share the story briefly, the problem being solved, and a soft call to action.\n"
        "Requirements:\n"
        "- Authentic tone\n"
        "- Keep under 180 words\n\n"
        "{output}",
    ),
    (
        "Summarize this topic: climate change",
        "writing",
        "Write a clear summary of climate change for {audience}.\n"
        "{context}\n\n"
        "Explain causes, impacts, and common mitigation approaches.\n"
        "Requirements:\n"
        "- Neutral and factual tone\n"
        "- Short sections with headings\n\n"
        "{output}",
    ),
    (
        "Write a cover letter for a software internship",
        "writing",
        "Write a cover letter for a software internship for {audience}.\n"
        "{context}\n\n"
        "Highlight relevant skills, one project, and motivation for the role.\n"
        "Requirements:\n"
        "- One page equivalent\n"
        "- Specific and confident tone\n\n"
        "{output}",
    ),
    (
        "Write release notes for a mobile app update",
        "writing",
        "Write release notes for a mobile app update for {audience}.\n"
        "{context}\n\n"
        "Include new features, bug fixes, and known issues.\n"
        "Requirements:\n"
        "- Bullet format\n"
        "- User-friendly language\n\n"
        "{output}",
    ),
    (
        "Explain machine learning simply",
        "writing",
        "Explain machine learning in simple terms for {audience}.\n"
        "{context}\n\n"
        "Use everyday analogies, one example, and avoid heavy jargon.\n"
        "Requirements:\n"
        "- Short sections\n"
        "- End with when ML is useful vs not\n\n"
        "{output}",
    ),
    # --- data / analysis ---
    (
        "Analyze this.",
        "data",
        "Analyze the provided dataset for {audience}.\n"
        "{context}\n\n"
        "Steps:\n"
        "1. Summarize columns and missing values\n"
        "2. Compute key descriptive stats\n"
        "3. Call out notable patterns or outliers\n"
        "4. Suggest next analyses\n\n"
        "State assumptions clearly.\n"
        "{output}",
    ),
    (
        "Analyze this dataset.",
        "data",
        "Perform an exploratory analysis of this dataset for {audience}.\n"
        "{context}\n\n"
        "Include data quality checks, distributions, correlations, and a short insight summary.\n"
        "{output}",
    ),
    (
        "Find insights in sales CSV",
        "data",
        "Analyze a sales CSV for insights for {audience}.\n"
        "{context}\n\n"
        "Look at revenue trends, top products, seasonality, and conversion if available.\n"
        "Requirements:\n"
        "- State assumptions\n"
        "- End with 3 actionable recommendations\n\n"
        "{output}",
    ),
    (
        "Compare two marketing campaigns",
        "data",
        "Compare two marketing campaigns for {audience}.\n"
        "{context}\n\n"
        "Use metrics like CTR, conversion rate, CAC, and ROI.\n"
        "Requirements:\n"
        "- Fair comparison table\n"
        "- Clear winner recommendation with caveats\n\n"
        "{output}",
    ),
    (
        "Clean this messy spreadsheet",
        "data",
        "Create a data-cleaning plan for a messy spreadsheet for {audience}.\n"
        "{context}\n\n"
        "Cover duplicates, missing values, inconsistent formats, and validation checks.\n"
        "{output}",
    ),
    (
        "Make a chart of monthly revenue",
        "data",
        "Create a monthly revenue chart and short interpretation for {audience}.\n"
        "{context}\n\n"
        "Show trend, peaks/dips, and one hypothesis for changes.\n"
        "{output}",
    ),
    (
        "Predict churn from user data",
        "data",
        "Outline a churn-prediction analysis for {audience}.\n"
        "{context}\n\n"
        "Define target, candidate features, simple baseline model, and evaluation metrics.\n"
        "Keep the first approach practical.\n"
        "{output}",
    ),
    (
        "Explain A/B test results",
        "data",
        "Explain A/B test results clearly for {audience}.\n"
        "{context}\n\n"
        "Cover sample size, metric lift, statistical significance, and business recommendation.\n"
        "{output}",
    ),
    # --- research ---
    (
        "Research React vs Vue",
        "research",
        "Compare React and Vue for {audience}.\n"
        "{context}\n\n"
        "Cover learning curve, ecosystem, performance, hiring, and when to choose each.\n"
        "End with a recommendation for a small startup MVP.\n"
        "{output}",
    ),
    (
        "Research Postgres vs MongoDB",
        "research",
        "Compare PostgreSQL and MongoDB for {audience}.\n"
        "{context}\n\n"
        "Cover data model fit, transactions, query patterns, scaling, and ops complexity.\n"
        "Recommend one for a typical SaaS app.\n"
        "{output}",
    ),
    (
        "Explain transformers in NLP",
        "research",
        "Explain transformer models in NLP for {audience}.\n"
        "{context}\n\n"
        "Cover attention, encoder/decoder roles, and why transformers beat older RNNs for many tasks.\n"
        "Use one concrete example.\n"
        "{output}",
    ),
    (
        "Evaluate whether to use Kubernetes",
        "research",
        "Evaluate whether a small team should use Kubernetes for {audience}.\n"
        "{context}\n\n"
        "Discuss complexity cost, benefits, alternatives (ECS/Compose/PaaS), and a decision checklist.\n"
        "{output}",
    ),
    (
        "Compare REST and GraphQL",
        "research",
        "Compare REST and GraphQL for {audience}.\n"
        "{context}\n\n"
        "Cover over/under-fetching, caching, tooling, and team fit.\n"
        "Recommend one for a mobile client MVP.\n"
        "{output}",
    ),
    # --- general / planning ---
    (
        "Create a workout plan.",
        "general",
        "Create a practical workout plan for {audience}.\n"
        "{context}\n\n"
        "Include weekly schedule, exercises, sets/reps, rest days, and progression tips.\n"
        "Ask for fitness level assumptions if missing, then proceed with sensible defaults.\n"
        "{output}",
    ),
    (
        "Help me with my startup idea.",
        "general",
        "Help refine a startup idea for {audience}.\n"
        "{context}\n\n"
        "Clarify problem, target user, unique value, MVP scope, and first validation steps.\n"
        "Be concrete and skeptical where needed.\n"
        "{output}",
    ),
    (
        "Help me with this.",
        "general",
        "Help solve this problem for {audience}.\n"
        "{context}\n\n"
        "Restate the goal, list missing details, then give a step-by-step plan with a first action.\n"
        "{output}",
    ),
    (
        "Make this better.",
        "general",
        "Improve this draft for {audience}.\n"
        "{context}\n\n"
        "Rewrite for clarity, structure, and actionability while preserving the original intent.\n"
        "Explain the top 3 changes briefly after the rewrite.\n"
        "{output}",
    ),
    (
        "Plan a 7-day study schedule for exams",
        "general",
        "Create a 7-day exam study schedule for {audience}.\n"
        "{context}\n\n"
        "Include daily topics, practice blocks, review time, and rest.\n"
        "Keep it realistic for ~4 hours/day.\n"
        "{output}",
    ),
    (
        "Make a meal plan for busy weekdays",
        "general",
        "Create a weekday meal plan for {audience}.\n"
        "{context}\n\n"
        "Focus on quick prep, balanced meals, and a shopping list.\n"
        "Assume ~30 minutes cooking time.\n"
        "{output}",
    ),
    (
        "Help me prepare for a product manager interview",
        "general",
        "Create a product manager interview prep plan for {audience}.\n"
        "{context}\n\n"
        "Cover product sense, metrics, behavioral stories, and practice questions.\n"
        "{output}",
    ),
    (
        "Design an onboarding flow for a mobile app",
        "general",
        "Design a mobile-app onboarding flow for {audience}.\n"
        "{context}\n\n"
        "Include screens, copy outline, permission asks, and success metric.\n"
        "Keep it to 3–5 screens.\n"
        "{output}",
    ),
    (
        "Create a content calendar for Instagram",
        "general",
        "Create a 2-week Instagram content calendar for {audience}.\n"
        "{context}\n\n"
        "Include post ideas, captions angle, posting cadence, and CTA variety.\n"
        "{output}",
    ),
    (
        "Make a travel itinerary for Tokyo",
        "general",
        "Create a 5-day Tokyo travel itinerary for {audience}.\n"
        "{context}\n\n"
        "Balance sightseeing, food, and transit time. Include neighborhood focus per day.\n"
        "{output}",
    ),
    (
        "Write a meeting agenda for sprint planning",
        "general",
        "Write a sprint planning meeting agenda for {audience}.\n"
        "{context}\n\n"
        "Include timeboxes, goals, backlog review, capacity check, and outcomes.\n"
        "{output}",
    ),
    (
        "Help me quit doomscrolling",
        "general",
        "Create a practical plan to reduce doomscrolling for {audience}.\n"
        "{context}\n\n"
        "Include triggers, phone settings changes, replacement habits, and a 7-day challenge.\n"
        "{output}",
    ),
    (
        "Make a budget for a new grad",
        "general",
        "Create a monthly budget template for a new graduate for {audience}.\n"
        "{context}\n\n"
        "Include rent, food, savings, debt, and discretionary spending with percentage guidelines.\n"
        "{output}",
    ),
    (
        "Plan a launch checklist for a SaaS MVP",
        "general",
        "Create a SaaS MVP launch checklist for {audience}.\n"
        "{context}\n\n"
        "Cover product readiness, analytics, support, marketing, and rollback plan.\n"
        "{output}",
    ),
    # Extra weak variants that still preserve topic
    (
        "I want an app like Facebook",
        "coding",
        "Build a Facebook-like social app for {audience}.\n"
        "{context}\n\n"
        "Must include profiles, feed, posts, likes, and comments.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Ship a usable MVP first\n\n"
        "{output}",
    ),
    (
        "App for sharing photos with friends",
        "coding",
        "Build a photo-sharing app for friends for {audience}.\n"
        "{context}\n\n"
        "Features: upload photos, albums, comments, private friend circles.\n"
        "Requirements:\n"
        "- {stack}\n\n"
        "{output}",
    ),
    (
        "Need a site for my bakery",
        "coding",
        "Build a bakery business website for {audience}.\n"
        "{context}\n\n"
        "Pages: menu, about, location/hours, contact/order form.\n"
        "Requirements:\n"
        "- Mobile responsive\n"
        "- Warm, appetizing visual style\n\n"
        "{output}",
    ),
    (
        "Automate my CSV reports",
        "coding",
        "Build a script or small tool that automates CSV reporting for {audience}.\n"
        "{context}\n\n"
        "Read CSVs, compute summaries, and export a clean report.\n"
        "Requirements:\n"
        "- Clear CLI flags\n"
        "- Handle malformed rows safely\n\n"
        "{output}",
    ),
    (
        "Write docs for my API",
        "writing",
        "Write clear API documentation for {audience}.\n"
        "{context}\n\n"
        "Include auth, endpoints, request/response examples, and error codes.\n"
        "{output}",
    ),
    (
        "Turn this rough idea into a prompt",
        "general",
        "Rewrite this rough idea into a high-quality LLM prompt for {audience}.\n"
        "{context}\n\n"
        "Preserve intent. Add audience, constraints, and expected output format.\n"
        "Return only the optimized prompt.\n"
        "{output}",
    ),
    # --- expanded curated seeds (more topics, same quality bar) ---
    (
        "Make a Discord bot",
        "coding",
        "Build a Discord bot for {audience}.\n"
        "{context}\n\n"
        "Features: slash commands, welcome messages, moderation helpers.\n"
        "Requirements:\n"
        "- Clear command structure\n"
        "- Environment-based token config\n"
        "- Error handling for API failures\n\n"
        "{output}",
    ),
    (
        "Build a Telegram bot for reminders",
        "coding",
        "Build a Telegram reminder bot for {audience}.\n"
        "{context}\n\n"
        "Features: set reminders, list upcoming, cancel reminders.\n"
        "Requirements:\n"
        "- Persist reminders\n"
        "- Timezone-aware scheduling\n\n"
        "{output}",
    ),
    (
        "Make a LinkedIn clone",
        "coding",
        "Build a LinkedIn-like professional networking app for {audience}.\n"
        "{context}\n\n"
        "Features: profiles, connections, posts/feed, job listings (basic).\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Keep v1 focused on profiles + feed\n\n"
        "{output}",
    ),
    (
        "Create a meditation timer app",
        "coding",
        "Build a meditation timer app for {audience}.\n"
        "{context}\n\n"
        "Features: timed sessions, ambient sound option, streak tracking.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Calm, distraction-free UI\n\n"
        "{output}",
    ),
    (
        "Make a language learning flashcard app",
        "coding",
        "Build a language-learning flashcard app for {audience}.\n"
        "{context}\n\n"
        "Features: decks, spaced repetition, progress stats.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Easy card create/edit flow\n\n"
        "{output}",
    ),
    (
        "Build a food delivery app",
        "coding",
        "Build a food delivery app for {audience}.\n"
        "{context}\n\n"
        "Features: restaurant list, menu, cart, checkout, order status.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Clear customer order flow\n\n"
        "{output}",
    ),
    (
        "Make a ride sharing app like Uber",
        "coding",
        "Build a ride-sharing MVP like Uber for {audience}.\n"
        "{context}\n\n"
        "Features: request ride, driver matching stub, fare estimate, trip status.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Separate rider and driver views if feasible\n\n"
        "{output}",
    ),
    (
        "Create a personal finance dashboard",
        "coding",
        "Build a personal finance dashboard for {audience}.\n"
        "{context}\n\n"
        "Show income, expenses, savings rate, and category breakdowns.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Charts + monthly filters\n\n"
        "{output}",
    ),
    (
        "Make an AI chatbot UI",
        "coding",
        "Build a chat UI for an AI assistant for {audience}.\n"
        "{context}\n\n"
        "Features: message list, input box, streaming-friendly layout, conversation history.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Clean accessible design\n\n"
        "{output}",
    ),
    (
        "Build a CRM for freelancers",
        "coding",
        "Build a lightweight CRM for freelancers for {audience}.\n"
        "{context}\n\n"
        "Features: clients, projects, invoices status, follow-up notes.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Simple table + detail views\n\n"
        "{output}",
    ),
    (
        "Make a podcast player app",
        "coding",
        "Build a podcast player app for {audience}.\n"
        "{context}\n\n"
        "Features: subscribe feeds, episode list, play/pause, playback position.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Persist progress\n\n"
        "{output}",
    ),
    (
        "Create a classroom quiz game",
        "coding",
        "Build a classroom quiz game for {audience}.\n"
        "{context}\n\n"
        "Features: teacher creates quiz, students answer live, scoreboard.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Fast join via room code\n\n"
        "{output}",
    ),
    (
        "Make a portfolio website for a designer",
        "coding",
        "Build a designer portfolio website for {audience}.\n"
        "{context}\n\n"
        "Pages: home, projects gallery, about, contact.\n"
        "Requirements:\n"
        "- Strong visual hierarchy\n"
        "- Mobile responsive\n\n"
        "{output}",
    ),
    (
        "Build an event ticketing site",
        "coding",
        "Build an event ticketing website for {audience}.\n"
        "{context}\n\n"
        "Features: event pages, ticket tiers, checkout, confirmation email stub.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Prevent overselling seats\n\n"
        "{output}",
    ),
    (
        "Make a smart home dashboard",
        "coding",
        "Build a smart-home control dashboard for {audience}.\n"
        "{context}\n\n"
        "Show device status, toggles for lights/thermostat, room grouping.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Mock device API is fine for v1\n\n"
        "{output}",
    ),
    (
        "Create a GitHub issue triage bot",
        "coding",
        "Build a GitHub issue triage helper for {audience}.\n"
        "{context}\n\n"
        "Label issues by keywords, suggest assignees, summarize new issues.\n"
        "Requirements:\n"
        "- Clear CLI or webhook entrypoint\n"
        "- Configurable rules\n\n"
        "{output}",
    ),
    (
        "Make a PDF invoice generator",
        "coding",
        "Build a PDF invoice generator for {audience}.\n"
        "{context}\n\n"
        "Inputs: client, line items, tax, due date. Output a clean PDF invoice.\n"
        "Requirements:\n"
        "- Template-based layout\n"
        "- Easy to regenerate\n\n"
        "{output}",
    ),
    (
        "Build a multiplayer tic-tac-toe",
        "coding",
        "Build a multiplayer tic-tac-toe game for {audience}.\n"
        "{context}\n\n"
        "Features: create/join room, turn-based play, win/draw detection.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Simple realtime or polling updates\n\n"
        "{output}",
    ),
    (
        "Make a browser extension for tab groups",
        "coding",
        "Build a browser extension for managing tab groups for {audience}.\n"
        "{context}\n\n"
        "Features: save/restore tab sets, name groups, quick switch.\n"
        "Requirements:\n"
        "- Manifest V3 compatible approach\n"
        "- Minimal permissions\n\n"
        "{output}",
    ),
    (
        "Create a stock watchlist app",
        "coding",
        "Build a stock watchlist app for {audience}.\n"
        "{context}\n\n"
        "Features: add tickers, show price/change, simple charts, refresh.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Handle API rate limits gracefully\n\n"
        "{output}",
    ),
    (
        "Make a recipe meal planner",
        "coding",
        "Build a weekly meal planner with recipes for {audience}.\n"
        "{context}\n\n"
        "Features: plan meals by day, generate shopping list, save favorites.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Printable shopping list view\n\n"
        "{output}",
    ),
    (
        "Build a code snippet manager",
        "coding",
        "Build a code snippet manager for {audience}.\n"
        "{context}\n\n"
        "Features: save snippets, tags, search, syntax highlighting.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Fast search by title/tag\n\n"
        "{output}",
    ),
    (
        "Make an anonymous feedback form app",
        "coding",
        "Build an anonymous feedback form app for {audience}.\n"
        "{context}\n\n"
        "Features: create form, collect responses, basic aggregation dashboard.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- No PII required from submitters\n\n"
        "{output}",
    ),
    (
        "Create a whiteboard collaboration tool",
        "coding",
        "Build a simple collaborative whiteboard for {audience}.\n"
        "{context}\n\n"
        "Features: draw shapes/text, multi-user cursors stub, save board.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Keep v1 drawing tools minimal\n\n"
        "{output}",
    ),
    (
        "Make a resume builder website",
        "coding",
        "Build a resume builder website for {audience}.\n"
        "{context}\n\n"
        "Features: form sections, live preview, export to PDF/Markdown.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Clean printable template\n\n"
        "{output}",
    ),
    (
        "Build a newsletter signup landing page",
        "coding",
        "Build a newsletter signup landing page for {audience}.\n"
        "{context}\n\n"
        "Include hero, benefits, email form, and success state.\n"
        "Requirements:\n"
        "- Mobile responsive\n"
        "- Validate email input\n\n"
        "{output}",
    ),
    (
        "Write a case study about a failed startup",
        "writing",
        "Write a startup failure case study for {audience}.\n"
        "{context}\n\n"
        "Cover idea, what went wrong, lessons, and what to do differently.\n"
        "Requirements:\n"
        "- Structured sections\n"
        "- Concrete takeaways\n\n"
        "{output}",
    ),
    (
        "Write onboarding emails for a SaaS trial",
        "writing",
        "Write a 5-email SaaS trial onboarding sequence for {audience}.\n"
        "{context}\n\n"
        "Day 0 welcome, activation tip, use-case story, objection handling, convert CTA.\n"
        "Requirements:\n"
        "- Subject lines included\n"
        "- Short and actionable\n\n"
        "{output}",
    ),
    (
        "Rewrite this angry customer reply professionally",
        "writing",
        "Rewrite an angry customer reply into a professional support response for {audience}.\n"
        "{context}\n\n"
        "Acknowledge frustration, clarify next steps, set expectations, stay calm.\n"
        "{output}",
    ),
    (
        "Write a FAQ for a password manager",
        "writing",
        "Write a FAQ for a password manager product for {audience}.\n"
        "{context}\n\n"
        "Cover security basics, sync, sharing, recovery, and pricing objections.\n"
        "{output}",
    ),
    (
        "Draft a privacy policy outline",
        "writing",
        "Draft a clear privacy policy outline for {audience}.\n"
        "{context}\n\n"
        "Sections: data collected, use, sharing, retention, rights, contact.\n"
        "Note this is an outline, not legal advice.\n"
        "{output}",
    ),
    (
        "Write a tutorial on git rebase",
        "writing",
        "Write a practical git rebase tutorial for {audience}.\n"
        "{context}\n\n"
        "Explain when to rebase vs merge, common commands, and conflict recovery.\n"
        "Include unsafe pitfalls.\n"
        "{output}",
    ),
    (
        "Create landing page copy for a habit app",
        "writing",
        "Write landing page copy for a habit-tracking app for {audience}.\n"
        "{context}\n\n"
        "Include headline, subhead, benefits, social proof placeholders, CTA.\n"
        "{output}",
    ),
    (
        "Summarize a research paper for executives",
        "writing",
        "Summarize a research paper for busy executives for {audience}.\n"
        "{context}\n\n"
        "One-page brief: problem, method, findings, business implications, open questions.\n"
        "{output}",
    ),
    (
        "Analyze website traffic drop",
        "data",
        "Analyze a website traffic drop for {audience}.\n"
        "{context}\n\n"
        "Check seasonality, channel mix, landing pages, technical issues, and competitor moves.\n"
        "End with ranked hypotheses and next measurements.\n"
        "{output}",
    ),
    (
        "Build a cohort retention table",
        "data",
        "Create a cohort retention analysis plan for {audience}.\n"
        "{context}\n\n"
        "Define cohort key, retention windows, table layout, and interpretation tips.\n"
        "{output}",
    ),
    (
        "Find anomalies in server logs",
        "data",
        "Create a plan to find anomalies in server logs for {audience}.\n"
        "{context}\n\n"
        "Cover parsing, error-rate spikes, latency outliers, and alerting thresholds.\n"
        "{output}",
    ),
    (
        "Score leads from CRM export",
        "data",
        "Design a simple lead-scoring approach from a CRM export for {audience}.\n"
        "{context}\n\n"
        "Choose features, a transparent scoring rule, and validation method.\n"
        "{output}",
    ),
    (
        "Explain SQL join types with examples",
        "data",
        "Explain SQL join types with concrete examples for {audience}.\n"
        "{context}\n\n"
        "Cover inner, left, right, full, and when each is useful.\n"
        "{output}",
    ),
    (
        "Design an experiment for pricing page",
        "data",
        "Design an A/B experiment for a pricing page for {audience}.\n"
        "{context}\n\n"
        "Define hypothesis, primary metric, sample size considerations, and stop rules.\n"
        "{output}",
    ),
    (
        "Research vector databases",
        "research",
        "Research vector databases for {audience}.\n"
        "{context}\n\n"
        "Compare popular options on features, ops cost, and fit for RAG apps.\n"
        "End with a recommendation for a small team.\n"
        "{output}",
    ),
    (
        "Should I use Next.js or Remix",
        "research",
        "Compare Next.js and Remix for {audience}.\n"
        "{context}\n\n"
        "Cover routing, data loading, ecosystem, hosting, and learning curve.\n"
        "Recommend one for a SaaS marketing site + app.\n"
        "{output}",
    ),
    (
        "Research best practices for prompt engineering",
        "research",
        "Summarize prompt-engineering best practices for {audience}.\n"
        "{context}\n\n"
        "Cover role/context, constraints, examples, evaluation, and failure modes.\n"
        "{output}",
    ),
    (
        "Compare SQLite and Postgres for side projects",
        "research",
        "Compare SQLite and PostgreSQL for side projects for {audience}.\n"
        "{context}\n\n"
        "Discuss concurrency, hosting, migrations, and when to upgrade.\n"
        "{output}",
    ),
    (
        "Evaluate open-source LLM options for local use",
        "research",
        "Evaluate open-source LLMs for local use for {audience}.\n"
        "{context}\n\n"
        "Consider VRAM, quality, licensing, and tool-calling support.\n"
        "Recommend 1–2 models for an 8GB laptop.\n"
        "{output}",
    ),
    (
        "Plan a digital detox weekend",
        "general",
        "Create a digital detox weekend plan for {audience}.\n"
        "{context}\n\n"
        "Include prep steps, phone settings, offline activities, and a re-entry plan.\n"
        "{output}",
    ),
    (
        "Help me negotiate a salary offer",
        "general",
        "Create a salary negotiation plan for {audience}.\n"
        "{context}\n\n"
        "Include research checklist, talk track, counteroffer strategy, and walk-away criteria.\n"
        "{output}",
    ),
    (
        "Make a reading list for system design",
        "general",
        "Create a system-design reading and practice list for {audience}.\n"
        "{context}\n\n"
        "Order topics from fundamentals to advanced, with weekly milestones.\n"
        "{output}",
    ),
    (
        "Plan a product roadmap for a notes app",
        "general",
        "Create a 3-month product roadmap for a notes app for {audience}.\n"
        "{context}\n\n"
        "Include themes, prioritized features, success metrics, and risks.\n"
        "{output}",
    ),
    (
        "Create a bug triage process",
        "general",
        "Design a bug triage process for a small engineering team for {audience}.\n"
        "{context}\n\n"
        "Define severity levels, SLAs, ownership, and weekly triage agenda.\n"
        "{output}",
    ),
    (
        "Help me learn Rust in 30 days",
        "general",
        "Create a 30-day Rust learning plan for {audience}.\n"
        "{context}\n\n"
        "Daily goals, project milestones, and resources. Keep it realistic for ~1 hour/day.\n"
        "{output}",
    ),
    (
        "Make a packing list for a hiking trip",
        "general",
        "Create a hiking trip packing list for {audience}.\n"
        "{context}\n\n"
        "Group by clothing, safety, cooking, electronics, and optional gear.\n"
        "{output}",
    ),
    (
        "Design a customer support playbook",
        "general",
        "Design a customer support playbook for {audience}.\n"
        "{context}\n\n"
        "Include response templates, escalation paths, SLAs, and quality checks.\n"
        "{output}",
    ),
    (
        "Create OKRs for an engineering team",
        "general",
        "Write quarterly OKRs for an engineering team for {audience}.\n"
        "{context}\n\n"
        "3 objectives with measurable key results; avoid vanity metrics.\n"
        "{output}",
    ),
    (
        "Make a social media content strategy for a cafe",
        "general",
        "Create a social media content strategy for a cafe for {audience}.\n"
        "{context}\n\n"
        "Define platforms, weekly cadence, content pillars, and sample posts.\n"
        "{output}",
    ),
    (
        "Help me migrate from monolith to services",
        "general",
        "Create a pragmatic monolith-to-services migration plan for {audience}.\n"
        "{context}\n\n"
        "Identify bounded contexts, strangler pattern steps, risks, and rollback.\n"
        "{output}",
    ),
    (
        "Build me a YouTube thumbnail analyzer",
        "coding",
        "Build a YouTube thumbnail analyzer tool for {audience}.\n"
        "{context}\n\n"
        "Score thumbnails on contrast, text readability, and face presence (heuristics OK).\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Return actionable feedback\n\n"
        "{output}",
    ),
    (
        "Make an app for split bills with friends",
        "coding",
        "Build a bill-splitting app for friends for {audience}.\n"
        "{context}\n\n"
        "Features: add expense, split equally/custom, settle balances, history.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Clear balance summary per person\n\n"
        "{output}",
    ),
    (
        "Create a plant watering reminder app",
        "coding",
        "Build a plant watering reminder app for {audience}.\n"
        "{context}\n\n"
        "Features: plant profiles, watering schedules, reminders, care notes.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Simple home screen of plants due today\n\n"
        "{output}",
    ),
    (
        "Make a local-first second brain notes system",
        "coding",
        "Build a local-first second-brain notes system for {audience}.\n"
        "{context}\n\n"
        "Features: markdown notes, backlinks, search, folders/tags.\n"
        "Requirements:\n"
        "- {stack}\n"
        "- Offline-first storage\n\n"
        "{output}",
    ),
    (
        "Write a prompt to generate unit tests",
        "general",
        "Write a high-quality prompt that asks an LLM to generate unit tests for {audience}.\n"
        "{context}\n\n"
        "The prompt should require framework, edge cases, and runnable tests only.\n"
        "Preserve the goal of unit-test generation.\n"
        "{output}",
    ),
    (
        "Optimize this vague hiring prompt",
        "general",
        "Rewrite a vague hiring prompt into a strong interviewer brief for {audience}.\n"
        "{context}\n\n"
        "Include role, must-have skills, scorecard, and example questions.\n"
        "Preserve hiring intent.\n"
        "{output}",
    ),
]


# Back-compat map used by fallback / tests
CANONICAL_WEAK: dict[str, tuple[str, str]] = {
    weak: (domain, weak.rstrip(".").strip())
    for weak, domain, _ in CURATED_PAIRS
}


def _fill_template(template: str, r: random.Random) -> str:
    return (
        template.replace("{audience}", r.choice(_AUDIENCES))
        .replace("{context}", r.choice(_CONTEXTS))
        .replace("{stack}", r.choice(_CODING_STACK))
        .replace("{output}", r.choice(_OUTPUTS))
        .strip()
    )


def _scores_for_weak(r: random.Random) -> dict[str, int]:
    """Weak prompts are low-quality by construction."""
    dims = {
        "clarity": clamp(25 + r.randint(-5, 10)),
        "specificity": clamp(15 + r.randint(-5, 12)),
        "context": clamp(12 + r.randint(-5, 12)),
        "goal_definition": clamp(22 + r.randint(-5, 12)),
        "constraints": clamp(8 + r.randint(-5, 12)),
        "completeness": clamp(14 + r.randint(-5, 12)),
        "actionability": clamp(18 + r.randint(-5, 12)),
    }
    dims["quality_score"] = clamp(float(np.mean(list(dims.values()))))
    return dims


def _missing_from_scores(scores: dict[str, int]) -> list[str]:
    missing: list[str] = []
    if scores["context"] < 45:
        missing.append("context")
    if scores["goal_definition"] < 45:
        missing.append("goal")
    if scores["constraints"] < 45:
        missing.append("constraints")
    if scores["specificity"] < 45:
        missing.append("specific_requirements")
    if scores["completeness"] < 45:
        missing.append("output_format")
    if scores["actionability"] < 45:
        missing.append("actionable_steps")
    return missing


def _issues_from_scores(scores: dict[str, int]) -> list[str]:
    issues: list[str] = []
    if scores["specificity"] < 40:
        issues.append("too_vague")
    if scores["context"] < 40:
        issues.append("missing_context")
    if scores["goal_definition"] < 45:
        issues.append("ambiguous_objective")
    if scores["constraints"] < 40:
        issues.append("insufficient_constraints")
    if scores["completeness"] < 40:
        issues.append("incomplete_prompt")
    return issues


def _pair_to_row(
    weak: str,
    domain: str,
    strong_template: str,
    r: random.Random,
) -> dict[str, Any]:
    optimized = _fill_template(strong_template, r)
    scores = _scores_for_weak(r)
    missing = _missing_from_scores(scores)
    issues = _issues_from_scores(scores)
    analysis = {
        "quality_score": scores["quality_score"],
        "dimensions": {k: v for k, v in scores.items() if k != "quality_score"},
        "issues": issues,
        "missing_information": missing,
    }
    messages = build_optimizer_messages(
        weak, analysis, optimized_prompt=optimized, task_type=domain
    )
    return {
        "prompt": weak,
        "task_type": domain,
        "canonical_task": weak.rstrip(".").strip(),
        "quality_level": 0,
        "quality_score": scores["quality_score"],
        "clarity": scores["clarity"],
        "specificity": scores["specificity"],
        "context": scores["context"],
        "goal_definition": scores["goal_definition"],
        "constraints": scores["constraints"],
        "completeness": scores["completeness"],
        "actionability": scores["actionability"],
        "issues": json.dumps(issues),
        "missing_information": json.dumps(missing),
        "analysis_json": json.dumps(analysis),
        "optimized_prompt": optimized,
        "messages_json": json.dumps(messages),
    }


def build_fallback_prompt(prompt: str, task_type: str = "general") -> str:
    """Intent-preserving fallback when model output fails validation."""
    r = random.Random(hash(prompt.strip().lower()) & 0xFFFFFFFF)
    normalized = prompt.strip().lower()

    for weak, domain, template in CURATED_PAIRS:
        if weak.lower() == normalized:
            return _fill_template(template, r)

    # Soft match: reuse curated strong side if weak keywords overlap a lot
    prompt_tokens = {t for t in normalized.replace(".", " ").split() if len(t) > 2}
    best: tuple[str, str, str] | None = None
    best_overlap = 0
    for weak, domain, template in CURATED_PAIRS:
        weak_tokens = {t for t in weak.lower().replace(".", " ").split() if len(t) > 2}
        overlap = len(prompt_tokens & weak_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best = (weak, domain, template)
    if best is not None and best_overlap >= 2:
        return _fill_template(best[2], r)

    topic = prompt.strip().rstrip(".")
    domain = task_type if task_type != "general" else "general"
    return (
        f"{topic} for {r.choice(_AUDIENCES)}.\n"
        f"{r.choice(_CONTEXTS)}\n\n"
        "Requirements:\n"
        f"- Be specific about the goal, audience, and constraints\n"
        f"- Include the expected output format\n"
        f"- Preserve the original topic: {topic}\n\n"
        f"{r.choice(_OUTPUTS)}"
    )


def generate_optimizer_example(rng: random.Random | None = None) -> dict[str, Any]:
    """Sample one curated weak→strong pair (with light strong-side variation)."""
    r = rng or random.Random()
    weak, domain, template = r.choice(CURATED_PAIRS)
    return _pair_to_row(weak, domain, template, r)


def generate_optimizer_dataset(
    num_examples: int = 800,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a high-quality optimizer dataset from curated topic-preserving pairs.

    Default 800 rows covers every seed multiple times with different fills.
    """
    rng = random.Random(seed)
    n = max(1, int(num_examples))
    rows: list[dict[str, Any]] = []

    # Round-robin through curated seeds for coverage, then fill randomly.
    seeds = list(CURATED_PAIRS)
    rng.shuffle(seeds)
    for i in range(n):
        weak, domain, template = seeds[i % len(seeds)]
        # Fresh RNG draw per row so fills differ even on same seed
        row_rng = random.Random(seed + i * 9973)
        rows.append(_pair_to_row(weak, domain, template, row_rng))

    return pd.DataFrame(rows)


def row_to_analysis(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    if row.get("analysis_json"):
        return json.loads(row["analysis_json"])
    return {
        "quality_score": row.get("quality_score"),
        "dimensions": {
            k: row[k]
            for k in (
                "clarity",
                "specificity",
                "context",
                "goal_definition",
                "constraints",
                "completeness",
                "actionability",
            )
            if k in row
        },
        "issues": json.loads(row["issues"]) if isinstance(row.get("issues"), str) else row.get("issues", []),
        "missing_information": (
            json.loads(row["missing_information"])
            if isinstance(row.get("missing_information"), str)
            else row.get("missing_information", [])
        ),
    }
