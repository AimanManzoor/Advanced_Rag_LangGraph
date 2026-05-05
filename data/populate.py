"""Populate data/ with a small fictional corpus for RAG demos."""
from pathlib import Path

DOCS = {
"01_about_helios.md": """
# About Helios Solar

Helios Solar was founded in 2014 in Austin, Texas by Maya Chen and Daniel Park, both former engineers at a national utility. The company designs and installs residential solar systems across Texas, Arizona, and New Mexico. As of 2025, Helios has installed over 12,000 systems and employs 340 people across three offices.

The mission is to make rooftop solar economically obvious for the average homeowner — meaning a payback period under 7 years without subsidies.
""",

"02_residential_panels.md": """
# Residential Solar Panel Lineup (2025)

Helios offers three panel tiers. The H-300 is the entry tier at 300W per panel, monocrystalline, with a 20-year power warranty. The H-400 is the mid-tier at 400W with bifacial cells and a 25-year warranty. The flagship H-450 produces 450W and includes integrated microinverters.

A typical 4-bedroom home in Austin needs 18-22 panels of the H-400, generating roughly 7.6 kW peak.
""",

"03_battery_storage.md": """
# Helios Battery Storage

The Helios Vault is a lithium iron phosphate (LFP) battery system rated at 13.5 kWh per unit. Up to four units can be stacked for a total of 54 kWh. The Vault includes a hybrid inverter and is compatible with all Helios panel tiers.

In Texas, customers who pair the Vault with H-400 panels qualify for a $4,000 rebate through the state's 2025 Resilient Homes program.
""",

"04_installation_process.md": """
# Installation Process

A typical installation takes 1-3 days on-site after a 2-week permitting period. The crew arrives at 7am, completes roof mounting and panel placement on day one, runs electrical and inverter connections on day two, and performs commissioning and inspection on day three. Single-story homes with simple roof geometry often finish in a single day.

The customer is not required to be present after the initial walkthrough.
""",

"05_warranty.md": """
# Warranty Terms

The H-400 and H-450 panels carry a 25-year linear power output warranty: at year 25, panels are guaranteed to produce at least 87% of their original rated output. Workmanship on the mounting and electrical work is warrantied for 10 years. The Helios Vault battery carries a 10-year warranty or 6,000 cycles, whichever comes first.

Warranty claims are processed within 30 days of submission.
""",

"06_q3_2025_earnings.md": """
# Q3 2025 Earnings Summary

Revenue was $48.2 million, up 22% year-over-year. Installations totaled 1,140 systems, with the H-400 accounting for 71% of panels shipped. Battery attach rate reached 38%, up from 24% in Q3 2024. Gross margin expanded to 31.4%.

Texas remained the largest market at 62% of revenue. Arizona grew fastest at 41% YoY. The company added 28 employees during the quarter and opened a new warehouse in Phoenix.
""",

"07_q4_2025_earnings.md": """
# Q4 2025 Earnings Summary

Revenue was $54.7 million, up 19% year-over-year. The full-year 2025 total reached $192 million, exceeding the $180 million guidance issued in Q1. Battery attach rate climbed to 44% in Q4. Operating cash flow was positive for the third consecutive quarter at $6.1 million.

The company guided to $230-240 million in revenue for 2026, citing the new Resilient Homes rebate as a key tailwind.
""",

"08_customer_case_chen.md": """
# Customer Case: The Chen Household, Round Rock TX

The Chens installed 22 H-400 panels and one Helios Vault in March 2025. Their average pre-install electric bill was $310/month. Six months post-install, the average bill is $14/month — primarily a connection fee. The system generated 9,820 kWh in those six months, exceeding modeled output by 4%.

The Chens financed through a 12-year Helios loan at 6.9% APR.
""",

"09_financing.md": """
# Financing Options

Helios offers three financing paths. Cash purchase: 5% discount off list price, fastest payback. Helios Loan: 6.9-8.4% APR over 12 or 20 years, no down payment, the customer owns the system. Power Purchase Agreement (PPA): no upfront cost, customer pays a fixed per-kWh rate (currently 9.2 cents/kWh in Texas) for 25 years, Helios owns the system.

The PPA is unavailable in Arizona and New Mexico.
""",

"10_engineering_blog_microinverters.md": """
# Engineering Notes: Why We Moved to Microinverters in the H-450

The H-450 ships with integrated microinverters rather than a string inverter. This decision was driven by partial-shading performance: in our field tests across 240 Texas installations, microinverter systems produced 8-13% more energy per panel under partial shading conditions versus string inverters.

The tradeoff is per-panel cost: microinverters add roughly $80 per panel. We absorbed this into the H-450 premium price.
"""
}

DATA_DIR = Path(__file__).parent
for filename, content in DOCS.items():
    (DATA_DIR / filename).write_text(content.strip() + "\n")
    print(f"Wrote {filename}")

print(f"\nDone. {len(DOCS)} documents written to {DATA_DIR}")
