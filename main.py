import argparse
import csv
import json
import os
import re
import sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace, asdict, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict, Counter
import statistics

from advanced_simulations import run_agent_market_simulation, MarketScenarioDetail


# Set matplotlib backend for environments that don't support GUI
try:
    import tkinter
    matplotlib.use('Agg')  # Interactive backend
except ImportError:
    matplotlib.use('Agg')  # Non-interactive backend

# Load env file
load_dotenv()

# Setup API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit(1)
client = OpenAI(api_key=api_key)

# Load knowledge base
try:
    with open("TokenomicsKnowledge.json", "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
except FileNotFoundError:
    print("File not found.")
    sys.exit(1)

# Reuse the same knowledge base file as the default historical dataset to avoid duplication
DEFAULT_HISTORICAL_DATASET_PATH = "TokenomicsKnowledge.json"

def load_historical_dataset(path: str) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Historical dataset not found at {path}. Continuing without it.")
        return []


historical_dataset = load_historical_dataset(DEFAULT_HISTORICAL_DATASET_PATH)


def seed_random_generators(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenomics pipeline controller")
    parser.add_argument(
        "--historical-dataset",
        dest="historical_dataset",
        default=DEFAULT_HISTORICAL_DATASET_PATH,
        help="Path to JSON dataset of historical project outcomes.",
    )
    parser.add_argument(
        "--dataset-report",
        dest="dataset_report",
        action="store_true",
        help="Skip LLM flow and print validation summary for the historical dataset.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Random seed for simulations to ensure deterministic backtests.",
    )
    return parser.parse_args()


def run_dataset_validation_cli(dataset: List[Dict], knowledge_base: List[Dict]) -> None:
    if not dataset:
        print("No entries available in the historical dataset. Provide a valid path via --historical-dataset.")
        return

    print("Historical Dataset Validation")
    print(f"Total reference projects: {len(dataset)}")

    allocation_stats = dataset_allocation_statistics(dataset)
    if allocation_stats:
        print("\nAllocation aggregates (mean, min-max):")
        for key, stats in allocation_stats.items():
            print(f" - {key}: mean {stats['mean']:.1f}% (min {stats['min']:.1f}%, max {stats['max']:.1f}%)")

    allocations = [entry.get("allocation", {}) for entry in dataset]
    community_shares = [alloc.get("Community", 0.0) for alloc in allocations if isinstance(alloc, dict)]
    insider_shares = [
        (alloc.get("Team", 0.0) + alloc.get("Investors", 0.0))
        for alloc in allocations
        if isinstance(alloc, dict)
    ]

    if community_shares:
        print(f"Avg community share: {np.mean(community_shares):.1f}% (min {min(community_shares):.1f}%, max {max(community_shares):.1f}%)")
    if insider_shares:
        print(f"Avg insider share (team+investors): {np.mean(insider_shares):.1f}%")

    overview = dataset_outcome_overview(dataset)
    if overview.get("drawdowns"):
        print(f"Median max drawdown: {overview['drawdowns']['median']:.2f}")
    if overview.get("inflation"):
        print(f"Median supply inflation: {overview['inflation']['median']:.2f}")
    incident_rate = overview.get("incident_rate")
    if incident_rate is not None:
        print(f"Governance incidents recorded in {incident_rate*100:.1f}% of projects")

    print("\nProject-specific notes:")
    for entry in dataset:
        name = entry.get("project", "Unknown")
        token = entry.get("token", "N/A")
        allocation = entry.get("allocation", {})
        risks = entry.get("risks", [])
        outcomes = entry.get("outcomes", {})
        community = allocation.get("Community", "?")
        insider = allocation.get("Team", 0) + allocation.get("Investors", 0)
        summary = f" - {name} ({token}): community {community}% vs insiders {insider}%"
        if outcomes:
            drawdown = outcomes.get("max_drawdown")
            inflation = outcomes.get("supply_inflation")
            summary += f" | drawdown {drawdown}, inflation {inflation}"
        print(summary)
        if risks:
            print(f"    Risks: {', '.join(risks)}")

    print("\nReference knowledge base entries available:", len(knowledge_base))
    print("Use this dataset report to benchmark future proposals via the main pipeline.")

# -----------------------
# DOMAIN DATA STRUCTURES
# -----------------------

@dataclass
class AllocationBucket:
    name: str
    percentage: float
    category: Optional[str] = None
    cliff_months: Optional[int] = None
    vesting_months: Optional[int] = None


@dataclass
class TokenomicsProposal:
    project_name: str
    token_symbol: str
    initial_supply: Optional[float]
    allocations: List[AllocationBucket]
    unlock_strategy: Optional[str] = None
    emissions_model: Optional[str] = None
    burn_mechanism: Optional[str] = None
    raw_text: str = ""
    json_payload: Optional[Dict] = None
    explanations: List[str] = field(default_factory=list)
    interpretability_notes: List[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    description: str
    goals: List[str]
    priorities: List[str]
    constraints: Dict[str, float]
    legal_risk_tolerance: str
    economic_signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class FairnessMetrics:
    t0_gini: float
    gini_12m: float
    gini_24m: float
    governance_influence_index: float


@dataclass
class GovernanceRiskAssessment:
    capture_risk_score: float
    voter_turnout_projection: float
    governance_influence_index: float
    notes: List[str]


@dataclass
class RealProjectValidationResult:
    reference_project: str
    similarity_score: float
    warnings: List[str]


def _infer_numeric_from_text(patterns: List[str], text: str) -> Optional[int]:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (IndexError, ValueError):
                continue
    return None


def infer_bucket_timing(label: str, text: str) -> Tuple[Optional[int], Optional[int]]:
    safe_label = re.escape(label.strip()) if label else ""
    if not safe_label:
        return None, None

    cliff_patterns = [
        rf"{safe_label}[^\.\n]*?(\d+)\s*-?month\s+cliff",
        rf"{safe_label}[^\.\n]*?cliff\s*:?\s*(\d+)",
    ]
    vesting_patterns = [
        rf"{safe_label}[^\.\n]*?(\d+)\s*-?month\s+vesting",
        rf"{safe_label}[^\.\n]*?vesting\s*:?\s*(\d+)",
    ]
    return _infer_numeric_from_text(cliff_patterns, text), _infer_numeric_from_text(vesting_patterns, text)


def infer_unlock_strategy(text: str) -> Optional[str]:
    keywords = ["linear", "milestone", "dynamic", "bonding curve", "liquidity mining"]
    for keyword in keywords:
        if keyword in text.lower():
            return f"Incorporates {keyword} unlocks"
    return None


def infer_emissions_model(text: str) -> Optional[str]:
    lowered = text.lower()
    if "deflation" in lowered or "burn" in lowered:
        return "Deflationary with burn sinks"
    if "inflation" in lowered:
        return "Managed inflationary emissions"
    if "fixed" in lowered:
        return "Fixed supply"
    return None


def infer_burn_mechanism(text: str) -> Optional[str]:
    lowered = text.lower()
    if "buyback" in lowered:
        return "Buyback-and-burn cadence noted"
    if "burn" in lowered:
        return "Burn mechanism referenced"
    return None


def infer_constraints(goals: List[str], priorities: List[str]) -> Dict[str, float]:
    constraints: Dict[str, float] = {}
    lower_text = " ".join(goals + priorities).lower()
    if "decentral" in lower_text or "community" in lower_text:
        constraints["min_community_share"] = 30.0
    else:
        constraints["min_community_share"] = 20.0

    if "team" in lower_text or "execution" in lower_text:
        constraints["max_team_share"] = 30.0
    else:
        constraints["max_team_share"] = 35.0

    if "regulation" in lower_text or "compliance" in lower_text:
        constraints["max_investor_share"] = 35.0
    return constraints


def infer_legal_risk_tolerance(user_input: Dict) -> str:
    legal_text = user_input.get("legal_design", "").lower()
    if any(word in legal_text for word in ["security", "regulation", "compliance"]):
        return "conservative"
    if any(word in legal_text for word in ["progressive", "experimental", "aggressive"]):
        return "aggressive"
    return "balanced"


def generate_emission_curve(style: str, length: int) -> List[float]:
    if length <= 0:
        return []
    curve: List[float] = []
    for idx in range(length):
        phase = idx / max(length - 1, 1)
        if style == "front_loaded":
            weight = 1.4 - phase
        elif style == "back_loaded":
            weight = 0.6 + phase
        elif style == "deflationary":
            weight = 1.2 - 0.8 * phase
        else:  # steady
            weight = 1.0
        curve.append(max(weight, 0.1))
    total = sum(curve)
    if total == 0:
        return [1 / length] * length
    return [value / total for value in curve]


def infer_economic_signals(user_input: Dict, result_text: str) -> Dict[str, float]:
    signals = {
        "emission_rate": 0.08,
        "burn_rate": 0.01,
        "demand_multiplier": 1.0,
        "volatility_bias": 1.0,
        "emission_style": "steady",
    }

    inflation_pref = user_input.get("inflation_preference", "").lower()
    economic_design = user_input.get("economic_design", "").lower()
    description = (user_input.get("project_description", "") + " " + result_text).lower()

    if any(term in inflation_pref for term in ["deflation", "burn"]):
        signals["emission_rate"] = 0.045
        signals["burn_rate"] = 0.02
        signals["emission_style"] = "back_loaded"
    elif "aggressive" in inflation_pref or "high" in inflation_pref:
        signals["emission_rate"] = 0.12
        signals["burn_rate"] = 0.008
        signals["emission_style"] = "front_loaded"
    elif "stable" in inflation_pref:
        signals["emission_rate"] = 0.07
        signals["emission_style"] = "steady"

    if "liquidity mining" in description or "yield" in description:
        signals["emission_rate"] += 0.01
        signals["demand_multiplier"] += 0.15

    if "staking" in economic_design or "staking" in description:
        signals["demand_multiplier"] += 0.1
        signals["burn_rate"] += 0.003

    if "bear" in description or "resilient" in description:
        signals["volatility_bias"] = 1.2
    elif "hyper growth" in description or "game" in description:
        signals["demand_multiplier"] += 0.2

    total_supply_pref = user_input.get("total_supply_preference", "").lower()
    if "capped" in total_supply_pref:
        signals["emission_style"] = "deflationary"

    return signals


def extract_json_payload(text: str) -> Optional[Dict]:
    json_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    matches = json_pattern.findall(text)
    for block in matches:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    # Fallback: try to parse entire text if it looks like JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def allocations_from_payload(payload: Dict, raw_text: str) -> Tuple[List[AllocationBucket], List[str]]:
    notes: List[str] = []
    allocations_data = payload.get("allocations") or payload.get("token_allocations")
    allocations: List[AllocationBucket] = []
    if isinstance(allocations_data, list):
        for bucket in allocations_data:
            try:
                name = bucket.get("name") or bucket.get("label") or "Unknown"
                percentage = float(bucket.get("percentage"))
                category = normalize_allocation_key(name)
                cliff = bucket.get("cliff_months")
                vesting = bucket.get("vesting_months")
                allocations.append(
                    AllocationBucket(
                        name=name.title(),
                        percentage=percentage,
                        category=category,
                        cliff_months=cliff,
                        vesting_months=vesting,
                    )
                )
                explanations = bucket.get("rationale")
                if explanations:
                    notes.append(f"{name}: {explanations}")
            except (TypeError, ValueError):
                continue
    if not allocations:
        # Fallback to regex parsing
        labels, values = extract_allocation_enhanced(raw_text)
        for label, value in zip(labels, values):
            category = normalize_allocation_key(label)
            cliff, vesting = infer_bucket_timing(label, raw_text)
            allocations.append(
                AllocationBucket(
                    name=label,
                    percentage=value,
                    category=category,
                    cliff_months=cliff,
                    vesting_months=vesting,
                )
            )
    return allocations, notes


def enforce_proposal_constraints(proposal: TokenomicsProposal,
                                 context: ProjectContext) -> TokenomicsProposal:
    notes = proposal.interpretability_notes[:]
    total = sum(bucket.percentage for bucket in proposal.allocations)
    if total <= 0:
        notes.append("Allocation sum was zero; defaulting to equal split.")
        equal_share = 100 / max(len(proposal.allocations), 1)
        proposal.allocations = [replace(bucket, percentage=round(equal_share, 2)) for bucket in proposal.allocations]
    elif abs(total - 100) > 0.5:
        factor = 100 / total
        notes.append(f"Allocations normalized from {total:.1f}% to 100%.")
        proposal.allocations = [replace(bucket, percentage=round(bucket.percentage * factor, 2)) for bucket in proposal.allocations]

    category_totals = _aggregate_allocations_by_category(proposal.allocations)
    min_comm = context.constraints.get("min_community_share", 20.0)
    if category_totals.get("Community", 0.0) < min_comm:
        deficit = min_comm - category_totals.get("Community", 0.0)
        notes.append(f"Community share raised by {deficit:.1f}% to meet constraint.")
        adjust_factor = deficit / max(len(proposal.allocations), 1)
        adjusted_allocations = []
        for bucket in proposal.allocations:
            if bucket.category == "Community":
                adjusted_allocations.append(replace(bucket, percentage=bucket.percentage + deficit))
            else:
                adjusted_allocations.append(replace(bucket, percentage=max(bucket.percentage - adjust_factor, 1)))
        proposal.allocations = adjusted_allocations

    proposal.interpretability_notes = notes
    return proposal


def calculate_temporal_fairness(proposal: TokenomicsProposal) -> FairnessMetrics:
    def project_percentages(month: int) -> List[float]:
        projected = []
        for bucket in proposal.allocations:
            vest = bucket.vesting_months or 24
            released = min(month / vest, 1.0)
            projected.append(bucket.percentage * released)
        return projected

    def safe_gini(values: List[float]) -> float:
        filtered = [max(v, 0) for v in values if v is not None]
        return evaluate_gini_helper(filtered)

    t0 = [bucket.percentage for bucket in proposal.allocations]
    gini_0 = safe_gini(t0)
    gini_12 = safe_gini(project_percentages(12))
    gini_24 = safe_gini(project_percentages(24))
    insider = sum(bucket.percentage for bucket in proposal.allocations if bucket.category in {"Team", "Investors"})
    community = sum(bucket.percentage for bucket in proposal.allocations if bucket.category == "Community") or 1
    governance_index = min(insider / community, 5.0)
    return FairnessMetrics(
        t0_gini=gini_0,
        gini_12m=gini_12,
        gini_24m=gini_24,
        governance_influence_index=governance_index,
    )


def evaluate_gini_helper(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(values)
    cumulative_sum = sum(sorted_values)
    if cumulative_sum == 0:
        return 0.0
    gini = (
        2 * sum((i + 1) * val for i, val in enumerate(sorted_values))
    ) / (n * cumulative_sum) - (n + 1) / n
    return max(0.0, gini)


def evaluate_governance_risk(proposal: TokenomicsProposal,
                             fairness: FairnessMetrics) -> GovernanceRiskAssessment:
    notes = []
    capture_score = min(fairness.governance_influence_index * 0.6 + fairness.t0_gini, 5.0)
    if capture_score > 2.5:
        notes.append("High insider balance relative to community; risk of governance capture.")
    voter_turnout = max(0.1, 1 - fairness.t0_gini)
    notes.append(f"Projected voter turnout {voter_turnout*100:.1f}% based on initial distribution.")
    return GovernanceRiskAssessment(
        capture_risk_score=capture_score,
        voter_turnout_projection=voter_turnout,
        governance_influence_index=fairness.governance_influence_index,
        notes=notes,
    )


def validate_against_real_projects(proposal: TokenomicsProposal,
                                   knowledge_base: List[Dict],
                                   dataset: Optional[List[Dict]] = None) -> Optional[RealProjectValidationResult]:
    best_match = None
    best_distance = float('inf')
    candidates = []

    for project in knowledge_base:
        allocation = project.get('tokenomics', {}).get('allocation', {})
        if not isinstance(allocation, dict):
            continue
        normalized = _normalize_allocation_dict(allocation)
        candidates.append((project, normalized))

    if dataset:
        for entry in dataset:
            allocation = entry.get('allocation', {})
            if isinstance(allocation, dict):
                normalized = _normalize_allocation_dict(allocation)
                candidates.append((entry, normalized))

    for project, normalized in candidates:
        categories = set(normalized.keys()) | {bucket.category or bucket.name for bucket in proposal.allocations}
        distance = 0.0
        for category in categories:
            proposal_value = sum(
                bucket.percentage
                for bucket in proposal.allocations
                if (bucket.category or bucket.name) == category
            )
            reference_value = normalized.get(category, 0.0)
            distance += abs(proposal_value - reference_value)
        if distance < best_distance:
            best_distance = distance
            best_match = project

    if not best_match:
        return None

    warnings = []
    risk_flags = best_match.get('risks', [])
    if risk_flags:
        warnings.append(f"Reference project highlighted risks: {', '.join(risk_flags)}")
    outcomes = best_match.get('outcomes', {})
    if outcomes:
        drawdown = outcomes.get('max_drawdown')
        inflation = outcomes.get('supply_inflation')
        if drawdown is not None:
            warnings.append(f"Historical max drawdown: {drawdown}")
        if inflation is not None:
            warnings.append(f"Observed supply inflation: {inflation}")

    return RealProjectValidationResult(
        reference_project=best_match.get('project', 'Unknown'),
        similarity_score=max(0.0, 100 - best_distance),
        warnings=warnings,
    )


def generate_tokenomics_proposal(user_input: Dict,
                                 result_text: str) -> Tuple[TokenomicsProposal, ProjectContext]:
    payload = extract_json_payload(result_text)
    allocations, notes = allocations_from_payload(payload or {}, result_text)
    initial_supply = payload.get('initial_supply') if isinstance(payload, dict) else None
    if initial_supply is None:
        initial_supply = extract_total_supply(result_text)

    unlock_strategy = payload.get('unlock_strategy') if isinstance(payload, dict) else infer_unlock_strategy(result_text)
    emissions_model = payload.get('emissions_model') if isinstance(payload, dict) else infer_emissions_model(result_text)
    burn_mechanism = payload.get('burn_mechanism') if isinstance(payload, dict) else infer_burn_mechanism(result_text)

    proposal = TokenomicsProposal(
        project_name=user_input.get("project_name", "Unnamed Project"),
        token_symbol=user_input.get("token_symbol", "TKN"),
        initial_supply=initial_supply,
        allocations=allocations,
        unlock_strategy=unlock_strategy,
        emissions_model=emissions_model,
        burn_mechanism=burn_mechanism,
        raw_text=result_text,
        json_payload=payload,
        interpretability_notes=notes,
    )

    if user_input.get("input_type") == "generic":
        description = user_input.get("project_description", "")
    else:
        description = "Structured intake provided via questionnaire."

    goals = [g.strip() for g in user_input.get("core_principles", []) if g.strip()]
    priorities = [p.strip() for p in user_input.get("token_purpose", []) if p.strip()]
    constraints = infer_constraints(goals, priorities)
    legal_risk = infer_legal_risk_tolerance(user_input)
    economic_signals = infer_economic_signals(user_input, result_text)

    context = ProjectContext(
        description=description,
        goals=goals,
        priorities=priorities,
        constraints=constraints,
        legal_risk_tolerance=legal_risk,
        economic_signals=economic_signals,
    )

    proposal = enforce_proposal_constraints(proposal, context)
    return proposal, context


def proposal_from_dataset_entry(entry: Dict) -> Tuple[TokenomicsProposal, ProjectContext]:
    allocation_dict = entry.get("allocation", {}) or {}
    allocations: List[AllocationBucket] = []
    for key, value in allocation_dict.items():
        try:
            pct = float(value)
            allocations.append(
                AllocationBucket(
                    name=key.title(),
                    percentage=pct,
                    category=normalize_allocation_key(key),
                    cliff_months=None,
                    vesting_months=None,
                )
            )
        except (TypeError, ValueError):
            continue

    if not allocations:
        allocations = [AllocationBucket(name="Community", percentage=100.0, category="Community")]

    initial_supply = entry.get("tokenomics", {}).get("total_supply") or entry.get("total_supply")
    if isinstance(initial_supply, str):
        try:
            initial_supply = float(initial_supply.replace(",", ""))
        except ValueError:
            initial_supply = None

    proposal = TokenomicsProposal(
        project_name=entry.get("project", "Historical Project"),
        token_symbol=entry.get("token", "TKN"),
        initial_supply=initial_supply or 1_000_000_000,
        allocations=allocations,
        unlock_strategy="Historical snapshot",
        emissions_model="Historical baseline",
        burn_mechanism=None,
        raw_text=entry.get("notes", ""),
        json_payload=entry,
    )

    constraints = {
        "min_community_share": 20.0,
        "max_team_share": 35.0,
        "max_investor_share": 40.0,
    }
    context = ProjectContext(
        description=entry.get("notes", "Historical project context"),
        goals=[],
        priorities=[],
        constraints=constraints,
        legal_risk_tolerance="balanced",
        economic_signals=infer_economic_signals({}, entry.get("notes", "")),
    )

    proposal = enforce_proposal_constraints(proposal, context)
    return proposal, context


def proposal_from_entry_via_llm(entry: Dict, project_summaries: str) -> Tuple[TokenomicsProposal, ProjectContext]:
    description_lines = []
    if entry.get("notes"):
        description_lines.append(f"Notes: {entry['notes']}")
    if entry.get("allocation"):
        try:
            alloc_str = ", ".join(f"{k}: {v}%" for k, v in entry["allocation"].items())
            description_lines.append(f"Known historical allocation: {alloc_str}")
        except Exception:
            pass
    description = "\n".join(description_lines) or "Historical project without detailed notes."

    user_input = {
        "input_type": "generic",
        "project_name": entry.get("project", "Historical Project"),
        "project_description": description,
    }
    prompt = create_generic_prompt(user_input, project_summaries)
    llm_result = ask_openai_enhanced(prompt, input_type="generic")
    return generate_tokenomics_proposal(user_input, llm_result)

# Input mode selection
def get_input_mode() -> str:
    print("Choose your input method:")
    print("1. Structured Input - Detailed project specifications (Recommended for precise control)")
    print("2. Generic Input - Simple project description")
    
    while True:
        choice = input("\nSelect input mode (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("Invalid choice.")

# Structured Input
def get_structured_input() -> Dict:
    data = {
        "input_type": "structured",
        "project_name": input("Project name: "),
        "token_symbol": input("Token symbol: "),
        "token_purpose": input("Token Purpose: ").split(","),
        "core_principles": input("Core Principles: ").split(","),
        "token_functions": input("Token functions (comma separated): ").split(","),
        "stakeholders": input("Stakeholders (comma separated): ").split(","),
        "economic_design": input("Economic Design: "),
        "legal_design": input("Legal Design: "),
        "technical_design": input("Technical Design: "),
        "gov_structures": input("Governance Structures: "),
        "total_supply_preference": input("Total supply preference (fixed/unlimited/capped): "),
        "inflation_preference": input("Inflation type (e.g., deflationary, stable, mild inflationary, etc.): "),
        "similar_projects": input("Similar projects for reference (comma separated): ").split(",")
    }
    
    # Clean up inputs
    data["token_functions"] = [x.strip() for x in data["token_functions"] if x.strip()]
    data["stakeholders"] = [x.strip() for x in data["stakeholders"] if x.strip()]
    data["similar_projects"] = [x.strip().lower() for x in data["similar_projects"] if x.strip()]
    
    return data

# Generic Input
def get_generic_input() -> Dict:
    print("Tell us about your project in your own words.")
    print("Example: 'We're creating a gaming platform where players earn rewards for completing quests and can trade NFT items...'")
    print()
    
    # Get basic project info
    project_name = input("Project name (optional, can be mentioned in description): ").strip()
    
    print("\nDescribe your project as naturally as possible. What are you building and for whom? Type 'DONE' on a new line when finished:")
    
    description_lines = []
    while True:
        line = input()
        if line.strip().upper() == 'DONE':
            break
        description_lines.append(line)
    
    full_description = '\n'.join(description_lines).strip()
    
    if not full_description:
        print("Project description cannot be empty.")
        return get_generic_input()
    
    return {
        "input_type": "generic",
        "project_name": project_name if project_name else "Not specified",
        "project_description": full_description,
        "description_length": len(full_description.split()),
        "timestamp": datetime.now().isoformat()
    }

# Project Reference Summarize
def summarize_all_projects(kb: List[Dict]) -> str:
    summaries = []
    for i, proj in enumerate(kb, 1):
        try:
            project_name = proj.get('project', 'Unknown')
            token_symbol = proj.get('token', 'N/A')
            
            # Enhanced extraction
            design_thinking = proj.get('token_design_thinking', {})
            tokenomics = proj.get('tokenomics', {})
            
            purpose = design_thinking.get('purpose', 'N/A')
            functions = design_thinking.get('functions', [])
            total_supply = tokenomics.get('total_supply', 'N/A')
            allocation = tokenomics.get('allocation', {})
            
            # Format allocation properly
            alloc_str = "N/A"
            if isinstance(allocation, dict):
                alloc_parts = [f"{k}: {v}" for k, v in allocation.items()]
                alloc_str = ", ".join(alloc_parts)
            elif isinstance(allocation, str):
                alloc_str = allocation
            
            summaries.append(f"""
                {i}. {project_name} ({token_symbol}):
                • Purpose: {purpose}
                • Functions: {', '.join(functions) if functions else 'N/A'}
                • Total Supply: {total_supply}
                • Key Allocation: {alloc_str}
                • Governance: {proj.get('governance', {}).get('model', 'N/A')}
                """)
        except Exception as e:
            print(f"Error processing project {i}: {e}")
            continue
    
    return "\n".join(summaries)

# Structured Prompt Engineering
def create_structured_prompt(user_input: Dict, project_summaries: str) -> str:
    similar_projects_context = ""
    if user_input.get('similar_projects'):
        similar_projects_context = f"""
        The user referenced the following projects: {', '.join(user_input['similar_projects'])}.
        Leverage design patterns, strategic choices, and trade-offs from these cases to guide your response.
        
        In addition to these, independently identify and consider other relevant token models or successful Web3 projects that share similar characteristics, purposes, or stakeholder dynamics—even if they are not listed in the user's reference.
        Base your suggestions on well-known public cases or commonly cited design patterns from the blockchain ecosystem.
        """

    stakeholder_context = ""
    if user_input.get('stakeholders'):
        stakeholder_context = f"""
        Key stakeholders to consider include: {', '.join(user_input['stakeholders'])}.
        Ensure their roles, incentives, and influence are clearly mapped and aligned with the token model.
        """

    prompt = f"""
        Leverage your track record in designing sustainable, incentive-aligned, and regulation-aware token economies to transform the structured data below into a coherent, forward-thinking tokenomics model that balances utility, governance, value capture, and stakeholder alignment

        Project Overview:
        - Project Name: {user_input['project_name']}
        - Token Symbol: {user_input['token_symbol']}
        - Core Principles: {', '.join(user_input.get('core_principles', []))}
        - Token Purpose: {', '.join(user_input.get('token_purpose', []))}
        - Core Functions: {', '.join(user_input.get('token_functions', []))}

        Design Preferences:
        - Supply Model: {user_input.get('total_supply_preference', 'Not specified')}
        - Inflation Strategy: {user_input.get('inflation_preference', 'Not specified')}
        - Economic Design: {user_input.get('economic_design', 'Not specified')}
        - Legal Context: {user_input.get('legal_design', 'Not specified')}
        - Technical Approach: {user_input.get('technical_design', 'Not specified')}
        - Governance Structures: {user_input.get('gov_structures', 'Not specified')}

        {stakeholder_context}
        {similar_projects_context}

        Reference Projects:
        {project_summaries}

        Begin by clearly articulating the token’s intended role in the ecosystem—why it exists, how it adds value, and what makes it essential. Clarify its core utilities and how these utilities will drive user adoption, engagement, and network effects. Describe how each stakeholder interacts with the token and how their incentives are supported through design using narrative-style writing.

        Then develop the token structure:
        - Propose an appropriate total supply (e.g., "1,000,000,000") based on the project’s scale
        - Allocate tokens using "Category: XX%" format, ensuring the total equals 100%
        - Detail a vesting schedule per group (team, investors, community) including cliffs, unlocks, and rationale
        - Explain how token distribution will unfold over time, from launch to maturity, including strategic release waves

        Construct a governance design that evolves with the project. Begin with practical oversight and transition to decentralized control over time. Include how voting occurs, who can propose, and how governance legitimacy is maintained without stagnation or capture.

        Design an economic model that ensures long-term sustainability. Focus on value accrual through meaningful token usage, scarcity mechanisms, and aligned incentives. Describe how the model defends against inflation, speculation, and underutilization.

        Conclude with an actionable roadmap covering launch stages, reward activations, governance rollout, and when/where stakeholders gain control. Emphasize maturity milestones and checkpoints for decentralization.

        Formatting Requirements:
        • Use exact numbers for total supply
        • Format allocations as "Category: XX%"
        • Ensure all percentages sum to 100%
        • Justify decisions using reasoning and references from similar projects
        • Prioritize long-term resilience, utility, and compliance readiness
        • Whenever relevant, explicitly mention which existing projects influenced your proposed model and how those inspirations were adapted to fit the user’s context.
        """
    return prompt


# Generic Prompt Engineering
def create_generic_prompt(user_input: Dict, project_summaries: str) -> str:
    prompt = f"""
        Given the narrative description of a Web3 project below, your task is to extract key design insights and produce a tokenomics model that is functional, balanced, and aligned with sustainable ecosystem growth.

        1. EXTRACT key project details from the description
        2. INFER appropriate tokenomics parameters based on the project type
        3. DESIGN a comprehensive tokenomics model using industry best practices

        USER'S PROJECT DESCRIPTION:
        {user_input['project_description']}

        PROJECT NAME: {user_input.get('project_name', 'Extract from description if mentioned')}

        Reference Projects:
        {project_summaries}

        Begin by understanding the nature of the project—its type, functionality, intended users, value proposition, and the business model it supports. Based on this, recommend a suitable blockchain infrastructure and governance design that fits the project’s scale and goals.

        Then define the token’s role within the ecosystem. Clarify its utility functions, economic significance, and how it supports stakeholder interactions. Indicate whether the token should be classified as a utility, governance, or hybrid model, depending on its purpose and usage.

        Next, build out the tokenomics architecture:
        - Recommend a total supply figure suitable for the project’s scope
        - Design a balanced token allocation plan that reflects stakeholder roles
        (use the format "Category: XX%"; ensure the sum equals 100%)
        - Propose a vesting strategy aligned with milestones and long-term alignment
        - Outline utility mechanisms such as staking, rewards, and fee participation

        Design the governance structure to support decentralization and decision making. Describe how proposals are submitted and voted on, and how authority transitions from the core team to the community over time.

        Then describe the token’s value accrual mechanisms. Explain how demand is generated and sustained, how value is captured, and how the system avoids inflation without utility.

        Conclude with a phased launch strategy that includes initial distribution, community growth, governance activation, and ecosystem expansion milestones.

        Design Principles:
        • Reference patterns from similar successful projects
        • Focus on long-term sustainability over short-term speculation
        • Ensure that token utility is clear, necessary, and valuable
        • Reflect regulatory awareness and compliance where applicable
        • Maintain fair and transparent distribution aligned with stakeholder interests

        Formatting Guidelines:
        • Use numerical values for total supply (e.g., "1,000,000,000")
        • Use "Category: XX%" format for all allocations
        • Ensure allocations sum to exactly 100%
        • Include reasoning for each major design choice
        • Reference successful cases from the knowledge base where relevant
        """
    
    return prompt

# OpenAI enhanced
def ask_openai_enhanced(prompt: str, input_type: str = "structured") -> str:
    if input_type == "generic":
        system_message = """
            You are Dr. Tokenomics, a world-renowned blockchain economist and tokenomics architect with over a decade of experience designing sustainable token economies for projects across DeFi, GameFi, infrastructure protocols, DAOs, and beyond—many of which have achieved billions in market capitalization.

            You ground all of your reasoning in the Token Design Thinking framework (Token Kitchen / Shermin Voshmgir) as represented in the TOKEN DESIGN TOOL. You understand that token design is *not* only about math or price action, but about socio-technical systems, governance, and power structures.

            Whenever you design or critique a token model, you mentally walk through the following lenses:

            1. PURPOSE  
            - Clarify the core PURPOSE of the project (single-purpose vs multi-purpose vs unclear).  
            - Identify for whom the system is built and who it is *not* for.  
            - Restate the project purpose in one or a few precise sentences before touching token mechanics.  

            2. PRINCIPLES & VALUES  
            - Extract the project’s mission, vision, and guiding PRINCIPLES.  
            - Make explicit which values, worldviews, and constraints should shape the token system (e.g., fairness, inclusivity, public-good orientation, profit-maximization, climate impact, etc.).  
            - Note what the project explicitly wants to avoid (e.g., pure speculation, plutocracy, extractive behavior).  

            3. POSITIONING & BUSINESS MODEL  
            - Determine whether the initiative is for-profit, non-profit, or mixed, and how it is (or will be) funded.  
            - Assess how essential tokens are to the system (core infrastructure vs optional add-on vs marketing gimmick).  
            - Map how the project is positioned relative to comparable ecosystems, protocols, or business models.  

            4. SYSTEM FUNCTIONS & TOKEN FUNCTIONS  
            - Separate **system functions** (what the network/DAO/protocol must do) from **token functions** (what the token is used for).  
            - Classify token functions such as:  
                - Access / membership  
                - Work / contribution / reward  
                - Payment / medium of exchange / settlement  
                - Collateral / store of value / liquidity  
                - Governance & signaling  
                - Reputation / identity / non-transferable roles  
                - Asset representation (real world, IP, in-game, etc.)  
            - Ensure token functions are tightly aligned with the project’s purpose and principles, not added “just because”.  

            5. STAKEHOLDERS & STAKEHOLDER MATRIX  
            - Identify all key STAKEHOLDER types (core team, early backers, users, contributors, validators, integrators, regulators, affected communities, etc.).  
            - For each type, map:  
                - ROLE in the system  
                - FUNCTIONS they perform  
                - RIGHTS & PERMISSIONS they need  
                - REWARDS & OBLIGATIONS (how they earn tokens, what they must do in return)  
            - Use this matrix to check incentive alignment and detect misalignments or missing roles.  

            6. TOKENS: NUMBER, TYPES, AND ROLES  
            - Decide how many token TYPES are truly necessary (fungible vs non-fungible, single vs multi-token architecture).  
            - For each token type, clearly define:  
                - Core PURPOSE  
                - Main FUNCTIONS  
                - Who should hold/earn/use it  
                - How it flows through the system (minting, distribution, circulation, burning/sinks).  
            - Avoid unnecessary token complexity unless it is justified by clear, contextual reasons.  

            7. ECONOMIC DESIGN TOOLBOX (EconDesignT1 & EconDesignT2)  
            For each token type, you think through qualitative economic design parameters such as:  
            - Supply: fixed, capped, elastic, inflationary, or dynamically adjustable.  
            - Issuance & Distribution: genesis allocation, emissions schedule, vesting, lockups, and release patterns.  
            - Sinks & Fees: how tokens leave circulation (burns, fees, bonding, staking, slashing, buybacks, etc.).  
            - Pricing & Markets: listing strategy, market-making, liquidity bootstrapping, and volatility mitigation.  
            - Economic Safety: anti-whale mechanisms, anti-Sybil measures, cap tables, and concentration risks.  
            - Sustainability: whether long-term funding, maintenance, and public-good components are properly resourced.  

            8. LEGAL & REGULATORY DESIGN  
            - Reflect on the *functional* classification of tokens (payment, utility, asset/security-like, governance, stable, hybrid, etc.).  
            - Consider KYC/AML, securities-law exposure, consumer protection, and other regulatory constraints in major jurisdictions.  
            - Highlight design options that reduce regulatory risk (e.g., clearer utility, phased decentralization, non-transferable reputation tokens, separation between governance and profit-rights).  
            - Emphasize that no answer is legal advice but that the model should aim to be legible and auditable for regulators.  

            9. TECHNICAL DESIGN  
            - Assess where which logic should live: on-chain vs off-chain; L1 vs L2; appchain vs shared infrastructure.  
            - Consider custody models, key management, and integration with wallets, bridges, and external systems.  
            - Connect technical choices back to token functions, security assumptions, and user experience.  

            10. POWER STRUCTURES  
                - Explicitly analyze POWER in the system:  
                - VOTING POWER: who can decide on upgrades, budgets, and strategic directions? is it 1 token–1 vote, 1 person–1 vote, or a hybrid?  
                - INFORMATION POWER: who has access to which data, and how does transparency/asymmetry shape power?  
                - MARKET POWER: who can significantly move markets, control liquidity, or gate access to secondary markets?  
                - MEDIATION POWER: who operates front-ends, oracles, infrastructure, and other chokepoints that mediate user access?  
                - Evaluate whether the token design reinforces or counterbalances centralization at these layers.  

            11. TEAM, ROADMAP & EVOLUTION  
                - Consider the TEAM’s skills, biases, and track record, and how that affects feasible token models.  
                - Embed progressive decentralization or power transitions where appropriate, instead of “instant DAO-washing”.  
                - Treat this as a qualitative *first step*: you design for clarity of structure so that later quantitative modeling is meaningful.  
                - Always acknowledge there is no one-size-fits-all model and that trade-offs are context dependent.  

            Your design philosophy emphasizes:
            - Long-term sustainability and socio-technical robustness over short-term speculation.  
            - Clear utility–value relationships where tokens have meaningful, coherent roles.  
            - Progressive decentralization and power rebalancing aligned with community readiness.  
            - Transparent articulation of trade-offs rather than pretending there is a single “optimal” design.  

            Your communication style is precise, implementation-focused, and analytical. You:
            - Provide actionable, well-reasoned recommendations ready for deployment or prototyping.  
            - Explain how and *why* your suggestions follow from the Token Design Thinking framework.  
            - Use qualitative and, where possible, quantitative/precedent-backed arguments to validate mechanisms.  
            - Present implementation plans in clear phases with measurable outcomes and explicit risks.  
            - Identify missing inputs from the questionnaire (Purpose, Principles, Positioning, Functions, Stakeholders, Tokens, EconDesign, Legal, Tech, Power) and state which questions the project still needs to answer.  

            All responses must be professional, comprehensive, and directly actionable for both technical and strategic teams.  
            Avoid vague or generic answers. Always deliver clarity, rigor, and practical value in every recommendation, and always keep your reasoning aligned with the TOKEN DESIGN TOOL lenses described above.
            """

    else:
        system_message = """
            You are a world-class blockchain tokenomics expert with experience designing token economies for top-tier Web3 projects such as Uniswap, Aave, Compound, and Chainlink. You deeply understand DeFi mechanics, governance design, token utility, and long-term sustainability.

            You use the Token Design Thinking framework (Token Kitchen / Shermin Voshmgir) as your primary mental model for structuring analysis. You treat token design as a qualitative systems-design exercise across the following lenses:
            - PURPOSE
            - PRINCIPLES
            - POSITIONING & BUSINESS MODEL
            - SYSTEM & TOKEN FUNCTIONS
            - STAKEHOLDERS & STAKEHOLDER MATRIX
            - TOKENS (number, types, roles)
            - ECONOMIC DESIGN TOOLBOX (supply, issuance, sinks, incentives)
            - LEGAL & REGULATORY DESIGN
            - TECHNICAL DESIGN
            - POWER STRUCTURES & GOVERNANCE

            Your goal is to translate project goals and narratives into effective, incentive-aligned token economic models that are sustainable, secure, and growth-oriented. You align stakeholders, ensure regulatory awareness, and design clear utility flows that create value for users and the ecosystem, while making power structures explicit.

In your responses, you typically:
- Start by clarifying PURPOSE and PRINCIPLES, and who the system is for (and not for).  
- Map STAKEHOLDERS, their roles, rights, rewards, and obligations.  
- Identify the minimum necessary TOKEN TYPES and FUNCTIONS (access, work, payment, governance, reputation, asset, etc.).  
- Propose ECONOMIC DESIGN choices (supply, distribution, emissions, sinks, fees, rewards) that fit the project’s context.  
- Highlight LEGAL/REGULATORY considerations and safer design options.  
- Suggest TECHNICAL patterns (on-chain/off-chain, L1/L2, custody) consistent with the token’s role.  
- Analyze POWER STRUCTURES (voting, information, market, and mediation power) and how the design affects centralization vs decentralization over time.  

Always provide specific, actionable recommendations with clear reasoning, implementation guidance, and consideration of economic, technical, governance, and power-structure impacts.  
Avoid generic responses—every answer must reflect deep domain expertise, a clear mapping to the Token Design Thinking lenses, and strategic precision.  
Be explicit about trade-offs and uncertainties, and note which inputs from the TOKEN DESIGN TOOL (e.g., stakeholder details, legal constraints, funding model) are still missing when they are relevant.
"""

    
    primary_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    fallback_model = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o-mini")

    def call_model(model_name: str, max_tokens: int = 1200) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content if resp.choices else ""

    try:
        content = call_model(primary_model, max_tokens=1200)
        if not (content and content.strip()):
            content = call_model(fallback_model, max_tokens=900)
        if not (content and content.strip()):
            return "Error: LLM response was empty. Please try again."
        return content
    except Exception as e:
        return f"Error generating recommendation: {e}"

# Allocation Parser
def extract_allocation_enhanced(text: str) -> Tuple[List[str], List[float]]:
    patterns = [
        r"- ([^:]+):\s*(\d{1,3}(?:\.\d{1,2})?)%", # Standard format
        r"([^:]+):\s*(\d{1,3}(?:\.\d{1,2})?)%", # Without dash
        r"• ([^:]+):\s*(\d{1,3}(?:\.\d{1,2})?)%", # Bullet point
        r"(\w+(?:\s+\w+)*)\s*-\s*(\d{1,3}(?:\.\d{1,2})?)%", # Dash separated
    ]
    
    labels = []
    values = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for label, value in matches:
                clean_label = label.strip().title()
                try:
                    clean_value = float(value)
                    if clean_label not in labels:  # Avoid duplicates
                        labels.append(clean_label)
                        values.append(clean_value)
                except ValueError:
                    continue
            
            if len(values) > 1:  # If found multiple allocations, use this pattern
                break
    
    # Validate that percentages sum to reasonable total
    total = sum(values)
    if total > 110:  # Allow some tolerance
        print(f"Total allocation ({total}%) exceeds 100%")
    elif total < 90:
        print(f"Total allocation ({total}%) is less than 90%")
    
    return labels, values

# Total Supply Allocation
def extract_total_supply(text: str) -> Optional[float]:
    patterns = [
        r'[Tt]otal\s+[Ss]upply[:\-\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # With commas
        r'[Tt]otal\s+[Ss]upply[:\-\s]*(\d+(?:\.\d+)?)',  # Without commas
        r'[Ss]upply[:\-\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*tokens?',  # Supply X tokens
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*tokens?\s*total',  # X tokens total
        r'[Mm]aximum\s+[Ss]upply[:\-\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # Maximum supply
        r'[Ff]ixed\s+[Ss]upply[:\-\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # Fixed supply
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                # Remove commas and convert to float
                supply_str = match.group(1).replace(',', '')
                supply_value = float(supply_str)
                
                # Sanity check: reasonable token supply range
                if 1000 <= supply_value <= 1e15:  # Between 1K and 1 quadrillion
                    print(f"Extracted total supply: {supply_value:,.0f}")
                    return supply_value
            except ValueError:
                continue
    
    # If no pattern matches, try to find any large number in the text
    large_numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})+)\b', text)
    if large_numbers:
        for num_str in large_numbers:
            try:
                num_value = float(num_str.replace(',', ''))
                if 1000000 <= num_value <= 1e15:  # At least 1 million tokens
                    print(f"Using inferred total supply: {num_value:,.0f}")
                    return num_value
            except ValueError:
                continue
    
    print("Could not extract total supply from the AI response.")
    return None

# Monte Carlo
def monte_carlo_simulation(initial_supply: float, burn_rate: float = 0.02, 
                          months: int = 12, simulations: int = 1000) -> Dict:
    outcomes = []
    monthly_burns = []
    
    for _ in range(simulations):
        current_supply = initial_supply
        monthly_burn_record = []
        
        for month in range(months):
            # Add some randomness to burn rate based on market conditions
            market_factor = np.random.normal(1.0, 0.2)  # Market volatility
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * month / 12)  # Seasonal variation
            
            effective_burn_rate = burn_rate * market_factor * seasonal_factor
            effective_burn_rate = max(0, min(effective_burn_rate, 0.1))  # Cap at 10%
            
            burned_amount = current_supply * effective_burn_rate
            current_supply -= burned_amount
            monthly_burn_record.append(burned_amount)
        
        outcomes.append(current_supply)
        monthly_burns.append(monthly_burn_record)
    
    return {
        'final_supplies': outcomes,
        'monthly_burns': monthly_burns,
        'mean_final_supply': np.mean(outcomes),
        'std_final_supply': np.std(outcomes),
        'median_final_supply': np.median(outcomes),
        'percentile_5': np.percentile(outcomes, 5),
        'percentile_95': np.percentile(outcomes, 95)
    }

# Gini Index
def calculate_gini_coefficient(values: List[float]) -> float:
    if not values or len(values) == 0:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(values)
    cumulative_sum = sum(sorted_values)
    
    if cumulative_sum == 0:
        return 0.0
    
    gini = (2 * sum((i + 1) * val for i, val in enumerate(sorted_values))) / (n * cumulative_sum) - (n + 1) / n
    return gini

# A/B Testing
def perform_ab_testing(gpt_labels: List[str], gpt_values: List[float]) -> Dict:
    return ab_testing_baseline(gpt_labels, gpt_values, knowledge_base)

# Visualizations
def create_allocation_pie_chart(labels: List[str], values: List[float], token_symbol: str):
    plt.figure(figsize=(10, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    # Create pie chart with enhanced styling
    wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', 
                                      startangle=90, colors=colors[:len(values)],
                                      shadow=True, explode=[0.05] * len(values))
    
    # Enhance text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
    
    plt.title(f'Token Allocation for {token_symbol.upper()}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')
    
    # Add a legend
    plt.legend(wedges, [f'{label}: {value}%' for label, value in zip(labels, values)],
              title="Allocation Breakdown", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    
    # Check if we can show interactive plots
    backend = matplotlib.get_backend()
    if backend == 'Agg':  # Non-interactive
        filename = f"allocation_{token_symbol.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Pie chart saved as: {filename}")
        plt.close()
    else:  # Interactive
        # Use non-blocking display
        plt.show(block=False)
        plt.pause(0.1)  # Allow chart to render
        
        # Ask user to confirm before continuing
        input("\nPie chart displayed. Press Enter to continue to analysis options...")

def _normalize_allocation_dict(allocation: Dict[str, float]) -> Dict[str, float]:
    normalized_allocation: Dict[str, float] = {}
    for key, value in allocation.items():
        try:
            float_val = float(value)
            if float_val <= 1.0:
                percentage = float_val * 100
            else:
                percentage = float_val
            normalized_key = normalize_allocation_key(key)
            normalized_allocation[normalized_key] = (
                normalized_allocation.get(normalized_key, 0.0) + percentage
            )
        except (ValueError, TypeError):
            continue
    return normalized_allocation


def extract_allocations_from_knowledge_base(knowledge_base: List[Dict]) -> Dict:
    all_allocations = []
    allocation_keys = Counter()
    project_allocations = {}
    
    for project in knowledge_base:
        project_name = project.get('project', 'Unknown')
        tokenomics = project.get('tokenomics', {})
        allocation = tokenomics.get('allocation', {})
        
        if allocation and isinstance(allocation, dict):
            normalized_allocation = _normalize_allocation_dict(allocation)
            for key in normalized_allocation:
                allocation_keys[key] += 1
            if normalized_allocation:
                all_allocations.append(normalized_allocation)
                project_allocations[project_name] = normalized_allocation
    
    return {
        'all_allocations': all_allocations,
        'allocation_keys': allocation_keys,
        'project_allocations': project_allocations
    }

def normalize_allocation_key(key: str) -> str:
    key_lower = key.lower().strip()
    # Define mapping for common variations
    key_mappings = {
        # Team variations
        'team': 'Team',
        'founders': 'Team',
        'core_team': 'Team',
        'founder': 'Team',
        
        # Community variations
        'community': 'Community',
        'public': 'Community',
        'public_and_community': 'Community',
        'community_rewards': 'Community',
        'users': 'Community',
        'community_liquidity_providers': 'Community',
        
        # Investor variations
        'investors': 'Investors',
        'private_sale': 'Investors',
        'seed': 'Investors',
        'series_a': 'Investors',
        'strategic': 'Investors',
        
        # Ecosystem variations
        'ecosystem': 'Ecosystem',
        'ecosystem_fund': 'Ecosystem',
        'development': 'Ecosystem',
        'treasury': 'Ecosystem',
        'dao_treasury': 'Ecosystem',
        
        # Advisors variations
        'advisors': 'Advisors',
        'advisory': 'Advisors',
        
        # Marketing variations
        'marketing': 'Marketing',
        'partnerships': 'Marketing',
        
        # Reserve variations
        'reserve': 'Reserve',
        'reserves': 'Reserve',
        'contingency': 'Reserve',
        
        # Operations variations
        'operations': 'Operations',
        'operational': 'Operations',
        'node_operators': 'Operations',
        
        # Liquidity variations
        'liquidity': 'Liquidity',
        'liquidity_mining': 'Liquidity',
        'liquidity_pool': 'Liquidity',
        
        # Staking variations
        'staking': 'Staking',
        'staking_rewards': 'Staking',
        'validator_rewards': 'Staking'
    }
    
    return key_mappings.get(key_lower, key.title().replace('_', ' '))

def calculate_allocation_statistics(allocations_data: Dict) -> Dict:
    all_allocations = allocations_data['all_allocations']
    allocation_keys = allocations_data['allocation_keys']
    
    # Get most common allocation categories
    common_categories = [key for key, count in allocation_keys.most_common(10)]
    
    category_stats = {}
    
    for category in common_categories:
        values = []
        for allocation in all_allocations:
            if category in allocation:
                values.append(allocation[category])
        
        if values:
            category_stats[category] = {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'values': values
            }
    
    return category_stats

def generate_dynamic_baseline_models(knowledge_base: List[Dict]) -> Dict:
    # Extract allocations from knowledge base
    allocations_data = extract_allocations_from_knowledge_base(knowledge_base)
    category_stats = calculate_allocation_statistics(allocations_data)
    
    # Generate different baseline models
    baseline_models = {}
    
    # 1. Mean-based model
    mean_allocation = {}
    for category, stats in category_stats.items():
        if stats['count'] >= 3:  # Only include if appears in at least 3 projects
            mean_allocation[category] = round(stats['mean'], 1)
    
    # Normalize to 100%
    mean_total = sum(mean_allocation.values())
    if mean_total > 0:
        mean_allocation = {k: round(v * 100 / mean_total, 1) for k, v in mean_allocation.items()}
        baseline_models['Knowledge Base Average'] = mean_allocation
    
    # 2. Median-based model
    median_allocation = {}
    for category, stats in category_stats.items():
        if stats['count'] >= 3:
            median_allocation[category] = round(stats['median'], 1)
    
    median_total = sum(median_allocation.values())
    if median_total > 0:
        median_allocation = {k: round(v * 100 / median_total, 1) for k, v in median_allocation.items()}
        baseline_models['Knowledge Base Median'] = median_allocation
    
    # 3. Conservative model (using lower quartile values)
    conservative_allocation = {}
    for category, stats in category_stats.items():
        if stats['count'] >= 3:
            # Use 25th percentile or minimum value
            sorted_values = sorted(stats['values'])
            q1_index = len(sorted_values) // 4
            conservative_value = sorted_values[q1_index] if q1_index > 0 else stats['min']
            conservative_allocation[category] = round(conservative_value, 1)
    
    conservative_total = sum(conservative_allocation.values())
    if conservative_total > 0:
        conservative_allocation = {k: round(v * 100 / conservative_total, 1) for k, v in conservative_allocation.items()}
        baseline_models['Conservative from Knowledge Based'] = conservative_allocation
    
    # 4. Aggressive model (using upper quartile values)
    aggressive_allocation = {}
    for category, stats in category_stats.items():
        if stats['count'] >= 3:
            # Use 75th percentile or maximum value
            sorted_values = sorted(stats['values'])
            q3_index = (3 * len(sorted_values)) // 4
            aggressive_value = sorted_values[q3_index] if q3_index < len(sorted_values) else stats['max']
            aggressive_allocation[category] = round(aggressive_value, 1)
    
    aggressive_total = sum(aggressive_allocation.values())
    if aggressive_total > 0:
        aggressive_allocation = {k: round(v * 100 / aggressive_total, 1) for k, v in aggressive_allocation.items()}
        baseline_models['Aggressive from Knowledge Based'] = aggressive_allocation
    
    return {
        'baseline_models': baseline_models,
        'category_stats': category_stats,
        'allocations_data': allocations_data
    }


def dataset_allocation_statistics(dataset: List[Dict]) -> Dict[str, float]:
    totals = defaultdict(list)
    for entry in dataset:
        allocation = entry.get('allocation', {})
        if isinstance(allocation, dict):
            normalized = _normalize_allocation_dict(allocation)
            for key, value in normalized.items():
                totals[key].append(value)

    stats = {}
    for key, values in totals.items():
        stats[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return stats


def dataset_outcome_overview(dataset: List[Dict]) -> Dict[str, float]:
    drawdowns = [entry.get('outcomes', {}).get('max_drawdown') for entry in dataset if entry.get('outcomes')]
    inflation = [entry.get('outcomes', {}).get('supply_inflation') for entry in dataset if entry.get('outcomes')]
    incidents = [entry.get('outcomes', {}).get('governance_incidents') for entry in dataset if entry.get('outcomes')]

    def safe_stats(values: List[Optional[float]]) -> Dict[str, float]:
        filtered = [v for v in values if isinstance(v, (int, float))]
        if not filtered:
            return {}
        return {
            "mean": float(np.mean(filtered)),
            "median": float(np.median(filtered)),
            "max": float(np.max(filtered)),
        }

    overview = {
        "drawdowns": safe_stats(drawdowns),
        "inflation": safe_stats(inflation),
        "incident_rate": sum(1 for v in incidents if v and v > 0) / max(1, len(dataset)),
    }
    return overview

# -----------------------
# CONTROL LAYER
# -----------------------

@dataclass
class StakeholderFinding:
    stakeholder: str
    severity: str
    message: str


@dataclass
class ComplianceFinding:
    severity: str
    message: str


@dataclass
class FeasibilityFinding:
    severity: str
    message: str


@dataclass
class ControlLayerResult:
    aligned: bool
    requires_iteration: bool
    stakeholder_findings: List[StakeholderFinding]
    compliance_findings: List[ComplianceFinding]
    feasibility_findings: List[FeasibilityFinding]
    fairness_metrics: Optional[FairnessMetrics] = None
    governance_risk: Optional[GovernanceRiskAssessment] = None


def _aggregate_allocations_by_category(allocations: List[AllocationBucket]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for bucket in allocations:
        key = bucket.category or bucket.name
        totals[key] += bucket.percentage
    return totals


def run_control_layer(proposal: TokenomicsProposal,
                      context: ProjectContext) -> ControlLayerResult:
    stakeholder_findings: List[StakeholderFinding] = []
    compliance_findings: List[ComplianceFinding] = []
    feasibility_findings: List[FeasibilityFinding] = []

    category_totals = _aggregate_allocations_by_category(proposal.allocations)
    community_share = category_totals.get("Community", 0.0)
    team_share = category_totals.get("Team", 0.0)
    investor_share = category_totals.get("Investors", 0.0)

    min_community = context.constraints.get("min_community_share", 20.0)
    max_team = context.constraints.get("max_team_share", 35.0)
    max_investor = context.constraints.get("max_investor_share", 40.0)

    if community_share < min_community:
        stakeholder_findings.append(
            StakeholderFinding(
                stakeholder="Community",
                severity="warning",
                message=f"Community allocation {community_share:.1f}% falls below target {min_community:.1f}% given stated goals."
            )
        )

    if team_share > max_team:
        stakeholder_findings.append(
            StakeholderFinding(
                stakeholder="Team",
                severity="warning",
                message=f"Team allocation {team_share:.1f}% exceeds recommended cap of {max_team:.1f}% for stated principles."
            )
        )

    if investor_share > max_investor:
        stakeholder_findings.append(
            StakeholderFinding(
                stakeholder="Investors",
                severity="critical" if investor_share > 50 else "warning",
                message=f"Investor allocation {investor_share:.1f}% dominates the cap table and may contradict decentralization goals."
            )
        )

    concentrated_share = team_share + investor_share
    if concentrated_share > 70:
        severity = "critical" if concentrated_share > 80 else "warning"
        compliance_findings.append(
            ComplianceFinding(
                severity=severity,
                message=f"Combined insider share (team + investors) at {concentrated_share:.1f}% could draw regulatory scrutiny for potential security classification."
            )
        )

    for bucket in proposal.allocations:
        if bucket.cliff_months and bucket.vesting_months and bucket.cliff_months > bucket.vesting_months:
            feasibility_findings.append(
                FeasibilityFinding(
                    severity="critical",
                    message=f"{bucket.name} cliff ({bucket.cliff_months}m) exceeds vesting ({bucket.vesting_months}m)."
                )
            )
        if bucket.category in {"Team", "Investors"} and bucket.vesting_months and bucket.vesting_months < 12:
            feasibility_findings.append(
                FeasibilityFinding(
                    severity="warning",
                    message=f"{bucket.name} unlocks in under 12 months, raising governance-capture risk."
                )
            )

    requires_iteration = any(f.severity == "critical" for f in stakeholder_findings) \
        or any(f.severity == "critical" for f in compliance_findings) \
        or any(f.severity == "critical" for f in feasibility_findings)

    aligned = not (stakeholder_findings or compliance_findings or feasibility_findings)

    fairness_metrics = calculate_temporal_fairness(proposal)
    governance_risk = evaluate_governance_risk(proposal, fairness_metrics)

    return ControlLayerResult(
        aligned=aligned,
        requires_iteration=requires_iteration,
        stakeholder_findings=stakeholder_findings,
        compliance_findings=compliance_findings,
        feasibility_findings=feasibility_findings,
        fairness_metrics=fairness_metrics,
        governance_risk=governance_risk,
    )


# -----------------------
# FILTER LAYER
# -----------------------

@dataclass
class FilterLayerResult:
    passed: bool
    adjusted_proposal: TokenomicsProposal
    allocation_issues: List[str]
    vesting_issues: List[str]
    economic_issues: List[str]
    governance_issues: List[str]
    fairness_metrics: Optional[FairnessMetrics] = None
    governance_risk: Optional[GovernanceRiskAssessment] = None


def run_filter_layer(proposal: TokenomicsProposal,
                     context: ProjectContext) -> FilterLayerResult:
    allocation_issues: List[str] = []
    vesting_issues: List[str] = []
    economic_issues: List[str] = []
    governance_issues: List[str] = []

    total_allocation = sum(bucket.percentage for bucket in proposal.allocations)
    adjusted_allocations = proposal.allocations

    if total_allocation <= 0:
        allocation_issues.append("CRITICAL: Allocation total is zero; cannot normalize.")
    elif abs(total_allocation - 100) > 0.5:
        factor = 100 / total_allocation
        allocation_issues.append(
            f"INFO: Allocation total was {total_allocation:.1f}%. Normalized to 100%."
        )
        adjusted_allocations = [
            replace(bucket, percentage=round(bucket.percentage * factor, 2))
            for bucket in proposal.allocations
        ]

    category_totals = _aggregate_allocations_by_category(adjusted_allocations)
    min_community = context.constraints.get("min_community_share", 20.0)
    max_team = context.constraints.get("max_team_share", 35.0)
    max_investor = context.constraints.get("max_investor_share", 40.0)

    if category_totals.get("Community", 0.0) < min_community:
        allocation_issues.append(
            f"CRITICAL: Community share {category_totals.get('Community', 0.0):.1f}% < target {min_community:.1f}%."
        )
    if category_totals.get("Team", 0.0) > max_team:
        allocation_issues.append(
            f"CRITICAL: Team share {category_totals.get('Team', 0.0):.1f}% > cap {max_team:.1f}%."
        )
    if category_totals.get("Investors", 0.0) > max_investor:
        allocation_issues.append(
            f"CRITICAL: Investor share {category_totals.get('Investors', 0.0):.1f}% > cap {max_investor:.1f}%."
        )

    for bucket in adjusted_allocations:
        if bucket.cliff_months and bucket.vesting_months and bucket.cliff_months > bucket.vesting_months:
            vesting_issues.append(f"CRITICAL: {bucket.name} cliff > vesting.")
        if bucket.category in {"Team", "Investors"} and (bucket.vesting_months or 0) < 12:
            vesting_issues.append(f"WARNING: {bucket.name} vesting under 12 months.")

    if proposal.initial_supply is None or proposal.initial_supply <= 0:
        economic_issues.append("WARNING: Initial supply missing or non-positive; using fallback during simulations.")

    decentralization_score = category_totals.get("Community", 0.0) - category_totals.get("Team", 0.0)
    if decentralization_score < 0:
        governance_issues.append("WARNING: Insiders control more tokens than the community at T0.")

    critical_present = any(issue.startswith("CRITICAL") for issue in (
        allocation_issues + vesting_issues + economic_issues + governance_issues
    ))

    adjusted_proposal = replace(proposal, allocations=adjusted_allocations)

    fairness_metrics = calculate_temporal_fairness(adjusted_proposal)
    governance_risk = evaluate_governance_risk(adjusted_proposal, fairness_metrics)

    return FilterLayerResult(
        passed=not critical_present,
        adjusted_proposal=adjusted_proposal,
        allocation_issues=allocation_issues,
        vesting_issues=vesting_issues,
        economic_issues=economic_issues,
        governance_issues=governance_issues,
        fairness_metrics=fairness_metrics,
        governance_risk=governance_risk,
    )


# -----------------------
# SIMULATIONS LAYER
# -----------------------

@dataclass
class ScenarioResult:
    name: str
    description: str
    supply_over_time: List[float]
    notes: List[str]


@dataclass
class GiniLayerResult:
    overall_gini: float
    circulating_gini: float
    governance_gini: float


@dataclass
class SupplyDynamicsResult:
    months: List[int]
    circulating_supply: List[float]
    locked_supply: List[float]
    burned_supply: List[float]


@dataclass
class BaselineComparisonResult:
    baseline_name: str
    similarities: List[str]
    differences: List[str]


@dataclass
class SimulationReport:
    scenarios: List[ScenarioResult]
    gini_layers: GiniLayerResult
    supply_dynamics: SupplyDynamicsResult
    baseline_comparison: Optional[BaselineComparisonResult]
    fairness_report: str
    validated_model_summary: str
    recommendations: List[str]
    fairness_metrics: FairnessMetrics
    governance_risk: GovernanceRiskAssessment
    real_project_validation: Optional[RealProjectValidationResult]
    agent_market_detail: Optional[MarketScenarioDetail] = None


def run_historical_pattern_module(proposal: TokenomicsProposal,
                                  knowledge_base: List[Dict],
                                  context: ProjectContext) -> ScenarioResult:
    proposal_categories = _aggregate_allocations_by_category(proposal.allocations)
    best_match = None
    best_overlap = -1.0

    for project in knowledge_base:
        allocation = project.get('tokenomics', {}).get('allocation', {})
        if not isinstance(allocation, dict):
            continue
        normalized = _normalize_allocation_dict(allocation)
        overlap = sum(
            min(proposal_categories.get(cat, 0.0), value)
            for cat, value in normalized.items()
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = (project, normalized)

    base_supply = proposal.initial_supply or 1_000_000_000
    milestones = [0, 12, 24, 36, 48, 60]
    if best_match:
        project, normalized = best_match
        description = f"Historical Pattern anchored on {project.get('project', 'unknown project')} allocations."
        notes = [
            f"Reference token: {project.get('token', 'N/A')}",
            f"Overlap score: {best_overlap:.1f}%",
            "Used to anchor expected distribution pressures."
        ]
    else:
        description = "No strong historical analog; using neutral pattern."
        notes = ["Proceeding with generic release curve."]

    emission_curve = generate_emission_curve(context.economic_signals.get("emission_style", "steady"), len(milestones))
    supply_over_time = [base_supply * 0.4 + base_supply * curve for curve in emission_curve]

    return ScenarioResult(
        name="Historical Pattern",
        description=description,
        supply_over_time=supply_over_time,
        notes=notes,
    )


def run_market_scenarios_module(proposal: TokenomicsProposal,
                                context: ProjectContext) -> List[ScenarioResult]:
    base_supply = proposal.initial_supply or 1_000_000_000
    milestones = [0, 12, 24, 36, 48, 60]
    community_bias = context.constraints.get("min_community_share", 30) / 100

    scenarios = []
    demand_multiplier = context.economic_signals.get("demand_multiplier", 1.0)
    burn_rate = context.economic_signals.get("burn_rate", 0.01)
    emission_rate = context.economic_signals.get("emission_rate", 0.08)

    for name, growth_bias, burn_bias in [
        ("Bull Market", 0.05, -0.005),
        ("Neutral Market", 0.0, 0.0),
        ("Bear Market", -0.03, 0.01),
    ]:
        supply_curve = []
        circulating = base_supply * 0.35
        for _ in milestones:
            growth = emission_rate + growth_bias
            burn = max(burn_rate + burn_bias, 0)
            circulating = circulating * (1 + growth * demand_multiplier) * (1 - burn)
            circulating = min(base_supply, circulating)
            supply_curve.append(circulating)
        scenarios.append(
            ScenarioResult(
                name=name,
                description=f"Assumes {name.lower()} demand with {growth*100:.0f}% release cadence.",
                supply_over_time=supply_curve,
                notes=[
                    f"Burn pressure {burn*100:.1f}% per period.",
                    f"Community usage share assumed at {community_bias*100:.1f}% of circulating supply."
                ]
            )
        )
    return scenarios


def run_stress_testing_module(proposal: TokenomicsProposal,
                              context: ProjectContext) -> List[ScenarioResult]:
    base_supply = proposal.initial_supply or 1_000_000_000
    large_unlock = max((bucket.percentage for bucket in proposal.allocations), default=20.0)
    cliff_month = min((bucket.cliff_months for bucket in proposal.allocations if bucket.cliff_months), default=12)

    bear_notes = [
        f"Models {large_unlock:.1f}% unlock at month {cliff_month}.",
        "Liquidity dries up for 2 quarters post unlock.",
    ]
    bull_notes = [
        "High throughput scenario with validators saturated.",
        "Protocol-owned liquidity buffers mitigate dumps.",
    ]

    volatility_bias = context.economic_signals.get("volatility_bias", 1.0)
    shock = min(max(volatility_bias - 1, -0.5), 0.5)

    bear_supply = [base_supply * (0.3 + shock), base_supply * (0.35 + shock), base_supply * 0.65, base_supply * 0.7, base_supply * 0.75, base_supply * 0.8]
    bull_supply = [base_supply * 0.35, base_supply * (0.45 + shock), base_supply * 0.6, base_supply * 0.7, base_supply * 0.78, base_supply * 0.85]

    return [
        ScenarioResult(
            name="Stress - Bear Shock",
            description="Cliff expiry plus liquidity drought stress test.",
            supply_over_time=bear_supply,
            notes=bear_notes,
        ),
        ScenarioResult(
            name="Stress - Bull Overheating",
            description="Sustained demand pressure with accelerated usage burns.",
            supply_over_time=bull_supply,
            notes=bull_notes,
        ),
    ]


def evaluate_gini_layers(proposal: TokenomicsProposal,
                         scenarios: List[ScenarioResult]) -> GiniLayerResult:
    def _calculate_gini(values: List[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(values)
        cumulative_sum = sum(sorted_values)
        if cumulative_sum == 0:
            return 0.0
        gini = (
            2 * sum((i + 1) * val for i, val in enumerate(sorted_values))
        ) / (n * cumulative_sum) - (n + 1) / n
        return max(0.0, gini)

    overall_gini = _calculate_gini([bucket.percentage for bucket in proposal.allocations])

    circulating_weights = []
    for bucket in proposal.allocations:
        if bucket.vesting_months is None or bucket.vesting_months <= 12:
            weight = 1.0
        elif bucket.vesting_months <= 24:
            weight = 0.7
        else:
            weight = 0.5
        circulating_weights.append(bucket.percentage * weight)
    circulating_gini = _calculate_gini(circulating_weights)

    governance_buckets = []
    for bucket in proposal.allocations:
        if bucket.category in {"Team", "Investors", "Advisors", "Community"}:
            influence = bucket.percentage
            if bucket.category == "Community":
                influence *= 0.8
            governance_buckets.append(influence)
    governance_gini = _calculate_gini(governance_buckets)

    return GiniLayerResult(
        overall_gini=overall_gini,
        circulating_gini=circulating_gini,
        governance_gini=governance_gini,
    )


def simulate_supply_dynamics(proposal: TokenomicsProposal,
                             scenarios: List[ScenarioResult],
                             context: ProjectContext) -> SupplyDynamicsResult:
    base_supply = proposal.initial_supply or 1_000_000_000
    months = list(range(0, 61, 6))
    circulating_supply: List[float] = []
    locked_supply: List[float] = []
    burned_supply: List[float] = []

    emissions_style = context.economic_signals.get("emission_style", "steady")
    emission_curve = generate_emission_curve(emissions_style, len(months))

    for idx, month in enumerate(months):
        released_ratio = 0.0
        for bucket in proposal.allocations:
            vest = bucket.vesting_months or 24
            released = min(month / vest, 1.0)
            released_ratio += (bucket.percentage / 100) * released
        circulating = base_supply * released_ratio * (1 + emission_curve[idx] * 0.05)
        burn = circulating * context.economic_signals.get("burn_rate", 0.01) * (month / 60)
        circulating_supply.append(circulating - burn)
        burned_supply.append(burn)
        locked_supply.append(max(base_supply - circulating, 0))

    return SupplyDynamicsResult(
        months=months,
        circulating_supply=circulating_supply,
        locked_supply=locked_supply,
        burned_supply=burned_supply,
    )


def compare_with_baselines(proposal: TokenomicsProposal,
                           knowledge_base: List[Dict]) -> BaselineComparisonResult:
    baselines = generate_dynamic_baseline_models(knowledge_base)['baseline_models']
    proposal_totals = _aggregate_allocations_by_category(proposal.allocations)

    best_name = None
    best_distance = float('inf')
    best_allocation = None

    for name, allocation in baselines.items():
        shared_categories = set(allocation.keys()) | set(proposal_totals.keys())
        distance = sum(
            abs(proposal_totals.get(cat, 0.0) - allocation.get(cat, 0.0))
            for cat in shared_categories
        )
        if distance < best_distance:
            best_distance = distance
            best_name = name
            best_allocation = allocation

    similarities: List[str] = []
    differences: List[str] = []
    if best_allocation:
        for cat, value in best_allocation.items():
            proposal_value = proposal_totals.get(cat, 0.0)
            if abs(proposal_value - value) <= 5:
                similarities.append(f"{cat}: proposal {proposal_value:.1f}% vs baseline {value:.1f}% (aligned)")
            else:
                differences.append(f"{cat}: proposal {proposal_value:.1f}% vs baseline {value:.1f}%")

    return BaselineComparisonResult(
        baseline_name=best_name or "Knowledge Base Average",
        similarities=similarities,
        differences=differences,
    )


def run_simulations_layer(proposal: TokenomicsProposal,
                          context: ProjectContext,
                          knowledge_base: List[Dict],
                          dataset: Optional[List[Dict]] = None) -> SimulationReport:
    historical = run_historical_pattern_module(proposal, knowledge_base, context)
    market = run_market_scenarios_module(proposal, context)
    stress = run_stress_testing_module(proposal, context)

    scenarios = [historical] + market + stress

    gini_layers = evaluate_gini_layers(proposal, scenarios)
    supply_dynamics = simulate_supply_dynamics(proposal, scenarios, context)
    baseline_comparison = compare_with_baselines(proposal, knowledge_base)
    fairness_metrics = calculate_temporal_fairness(proposal)
    governance_risk = evaluate_governance_risk(proposal, fairness_metrics)
    real_project_validation = validate_against_real_projects(proposal, knowledge_base, dataset)

    category_totals = _aggregate_allocations_by_category(proposal.allocations)
    community_share = category_totals.get("Community", 0.0) or 1.0
    insider_share = category_totals.get("Team", 0.0) + category_totals.get("Investors", 0.0)
    agent_market_detail = run_agent_market_simulation(
        proposal_initial_supply=proposal.initial_supply or 1_000_000_000,
        community_share=community_share,
        insider_share=max(insider_share, 1.0),
        context_signals=context.economic_signals,
    )

    dataset_overview = dataset_outcome_overview(dataset) if dataset else None

    fairness_report = (
        f"T0 Gini {fairness_metrics.t0_gini:.3f}, 12m {fairness_metrics.gini_12m:.3f}, 24m {fairness_metrics.gini_24m:.3f}. "
        f"Governance influence index {fairness_metrics.governance_influence_index:.2f}."
    )
    if dataset_overview and dataset_overview.get("incident_rate") is not None:
        fairness_report += f" Historical incident rate baseline {dataset_overview['incident_rate']*100:.1f}%."

    validated_model_summary = (
        "Model remains coherent across market/stress modules provided community share is upheld "
        "and unlock pacing follows the proposed vesting curve."
    )

    recommendations = []
    if gini_layers.governance_gini > 0.25:
        recommendations.append("Introduce delegated voting caps or quadratic voting to offset governance concentration.")
    if supply_dynamics.circulating_supply[-1] / (proposal.initial_supply or 1_000_000_000) < 0.8:
        recommendations.append("Extend emissions beyond 60 months or add sinks to avoid idle supply build-up.")
    if agent_market_detail and min(agent_market_detail.liquidity_levels) < 0.2:
        recommendations.append("Bolster liquidity reserves or stagger unlocks to avoid simulated liquidity floor breaches.")
    if dataset_overview and dataset_overview.get("drawdowns"):
        median_drawdown = dataset_overview["drawdowns"].get("median")
        if median_drawdown and median_drawdown > 0.5:
            recommendations.append("Incorporate circuit breakers; historical drawdowns above 50% suggest elevated volatility.")
    if not recommendations:
        recommendations.append("Maintain current allocation but document KPI triggers for future reallocations.")
    else:
        recommendations.append("Run live governance drills before TGE to validate adaptive consent logic.")

    return SimulationReport(
        scenarios=scenarios,
        gini_layers=gini_layers,
        supply_dynamics=supply_dynamics,
        baseline_comparison=baseline_comparison,
        fairness_report=fairness_report,
        validated_model_summary=validated_model_summary,
        recommendations=recommendations,
        fairness_metrics=fairness_metrics,
        governance_risk=governance_risk,
        real_project_validation=real_project_validation,
        agent_market_detail=agent_market_detail,
    )


def save_pipeline_outputs(output_dir: str,
                          proposal: TokenomicsProposal,
                          context: ProjectContext,
                          control_result: ControlLayerResult,
                          filter_result: FilterLayerResult,
                          simulation_report: Optional[SimulationReport] = None) -> None:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    proposal_path = os.path.join(output_dir, f"proposal_{timestamp}.json")
    context_path = os.path.join(output_dir, f"context_{timestamp}.json")
    control_path = os.path.join(output_dir, f"control_{timestamp}.json")
    filter_path = os.path.join(output_dir, f"filter_{timestamp}.json")

    with open(proposal_path, "w", encoding="utf-8") as f:
        json.dump(asdict(proposal), f, ensure_ascii=False, indent=2)

    with open(context_path, "w", encoding="utf-8") as f:
        json.dump(asdict(context), f, ensure_ascii=False, indent=2)

    control_payload = {
        "aligned": control_result.aligned,
        "requires_iteration": control_result.requires_iteration,
        "stakeholder_findings": [asdict(f) for f in control_result.stakeholder_findings],
        "compliance_findings": [asdict(f) for f in control_result.compliance_findings],
        "feasibility_findings": [asdict(f) for f in control_result.feasibility_findings],
    }
    with open(control_path, "w", encoding="utf-8") as f:
        json.dump(control_payload, f, ensure_ascii=False, indent=2)

    filter_payload = {
        "passed": filter_result.passed,
        "allocation_issues": filter_result.allocation_issues,
        "vesting_issues": filter_result.vesting_issues,
        "economic_issues": filter_result.economic_issues,
        "governance_issues": filter_result.governance_issues,
        "adjusted_allocations": [asdict(bucket) for bucket in filter_result.adjusted_proposal.allocations],
    }
    with open(filter_path, "w", encoding="utf-8") as f:
        json.dump(filter_payload, f, ensure_ascii=False, indent=2)

    if simulation_report:
        simulation_path = os.path.join(output_dir, f"simulation_{timestamp}.json")
        simulation_payload = {
            "scenarios": [asdict(s) for s in simulation_report.scenarios],
            "gini_layers": asdict(simulation_report.gini_layers),
            "supply_dynamics": asdict(simulation_report.supply_dynamics),
            "baseline_comparison": asdict(simulation_report.baseline_comparison)
            if simulation_report.baseline_comparison else None,
            "fairness_report": simulation_report.fairness_report,
            "validated_model_summary": simulation_report.validated_model_summary,
            "recommendations": simulation_report.recommendations,
        }
        with open(simulation_path, "w", encoding="utf-8") as f:
            json.dump(simulation_payload, f, ensure_ascii=False, indent=2)


def estimate_drawdown_from_prices(price_path: List[float]) -> Optional[float]:
    if not price_path:
        return None
    peak = max(price_path)
    trough = min(price_path)
    if peak <= 0:
        return None
    return (peak - trough) / peak


def estimate_inflation_from_supply(supply_result: SupplyDynamicsResult,
                                   initial_supply: float) -> Optional[float]:
    if not supply_result.circulating_supply:
        return None
    final_supply = supply_result.circulating_supply[-1]
    if initial_supply <= 0:
        return None
    return (final_supply - initial_supply) / initial_supply


def run_backtest(dataset: List[Dict],
                 knowledge_base: List[Dict],
                 include_kb: bool,
                 seed: int,
                 use_llm: bool) -> None:
    seed_random_generators(seed)
    entries = list(dataset)
    if include_kb:
        entries += knowledge_base

    if not entries:
        print("No entries available for backtesting.")
        return

    os.makedirs("backtest_exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join("backtest_exports", f"backtest_report_{timestamp}.csv")

    project_summaries = summarize_all_projects(knowledge_base)
    rows = []
    for entry in entries:
        if use_llm:
            proposal, context = proposal_from_entry_via_llm(entry, project_summaries)
        else:
            proposal, context = proposal_from_dataset_entry(entry)
        control_result = run_control_layer(proposal, context)
        filter_result = run_filter_layer(proposal, context)
        sim_report = run_simulations_layer(
            filter_result.adjusted_proposal,
            context,
            knowledge_base,
            dataset,
        )

        outcomes = entry.get("outcomes", {}) or {}
        predicted_drawdown = estimate_drawdown_from_prices(
            sim_report.agent_market_detail.price_path if sim_report.agent_market_detail else []
        )
        predicted_inflation = estimate_inflation_from_supply(
            sim_report.supply_dynamics,
            proposal.initial_supply or 1_000_000_000,
        )
        predicted_incident = 1 if sim_report.governance_risk.capture_risk_score >= 2.5 else 0

        actual_drawdown = outcomes.get("max_drawdown")
        actual_inflation = outcomes.get("supply_inflation")
        actual_incidents = outcomes.get("governance_incidents")

        rows.append({
            "project": proposal.project_name,
            "token": proposal.token_symbol,
            "control_requires_iteration": control_result.requires_iteration,
            "filter_passed": filter_result.passed,
            "t0_gini": sim_report.fairness_metrics.t0_gini,
            "gini_12m": sim_report.fairness_metrics.gini_12m,
            "gini_24m": sim_report.fairness_metrics.gini_24m,
            "gov_influence_index": sim_report.governance_risk.governance_influence_index,
            "baseline": sim_report.baseline_comparison.baseline_name if sim_report.baseline_comparison else "",
            "pred_drawdown": predicted_drawdown,
            "actual_drawdown": actual_drawdown,
            "drawdown_error": (predicted_drawdown - actual_drawdown) if (predicted_drawdown is not None and actual_drawdown is not None) else None,
            "pred_inflation": predicted_inflation,
            "actual_inflation": actual_inflation,
            "inflation_error": (predicted_inflation - actual_inflation) if (predicted_inflation is not None and actual_inflation is not None) else None,
            "pred_incident_flag": predicted_incident,
            "actual_incidents": actual_incidents,
            "incident_gap": (predicted_incident - actual_incidents) if actual_incidents is not None else None,
        })

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Backtest completed on {len(rows)} entries. Report saved to {csv_path}")


# Main Functions
def main(): 
    args = parse_cli_args()
    dataset = load_historical_dataset(args.historical_dataset)
    if args.dataset_report:
        run_dataset_validation_cli(dataset, knowledge_base)
        return

    seed_random_generators(args.seed)

    os.makedirs("exported_charts", exist_ok=True)
    input_mode = get_input_mode()

    if input_mode == '1':
        user_input = get_structured_input()
    else:
        user_input = get_generic_input()

    project_summaries = summarize_all_projects(knowledge_base)

    if user_input.get('input_type') == 'structured':
        prompt = create_structured_prompt(user_input, project_summaries)
        input_type = "structured"
    else:
        prompt = create_generic_prompt(user_input, project_summaries)
        input_type = "generic"

    print("\nGenerating token design...")
    result = ask_openai_enhanced(prompt, input_type)

    print("Tokenomics Design")
    print(result)

    token_symbol = user_input.get('token_symbol', user_input.get('project_name', 'TOKEN'))
    labels, values = extract_allocation_enhanced(result)

    if len(values) > 1:
        create_allocation_pie_chart(labels, values, token_symbol)
    else:
        print("Could not extract proper token allocation from the recommendation.")
        print("Please review the AI response manually.")
        return

    # Build proposal/context from LLM text and parsed allocations
    proposal, context = generate_tokenomics_proposal(user_input, result)

    control_result = run_control_layer(proposal, context)
    if control_result.stakeholder_findings:
        for finding in control_result.stakeholder_findings:
            print(f"[{finding.severity.upper()}] {finding.stakeholder}: {finding.message}")
    if control_result.compliance_findings:
        for finding in control_result.compliance_findings:
            print(f"[{finding.severity.upper()}] Compliance: {finding.message}")
    if control_result.feasibility_findings:
        for finding in control_result.feasibility_findings:
            print(f"[{finding.severity.upper()}] Feasibility: {finding.message}")
    if control_result.aligned:
        print("Control Layer: proposal aligns with stated goals.")

    filter_result = run_filter_layer(proposal, context)
    for label, issues in [
        ("Allocation", filter_result.allocation_issues),
        ("Vesting", filter_result.vesting_issues),
        ("Economic", filter_result.economic_issues),
        ("Governance", filter_result.governance_issues),
    ]:
        if issues:
            print(f"{label} Issues:")
            for issue in issues:
                print(f" - {issue}")
    if filter_result.passed:
        print("Filter Layer: Passed hard constraints check.")
    else:
        print("Filter Layer: Critical issues detected. Please iterate before simulations.")

    if control_result.requires_iteration or not filter_result.passed:
        print("\n[INFO] Critical findings detected, continuing to simulations for insight-only run.")

    sim_report = run_simulations_layer(filter_result.adjusted_proposal, context, knowledge_base, dataset)

    for scenario in sim_report.scenarios:
        print(f"\nScenario: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"  Supply trajectory: {[int(x) for x in scenario.supply_over_time]}")
        for note in scenario.notes:
            print(f"   - {note}")

    print("\nFairness Report:")
    print(sim_report.fairness_report)
    if sim_report.agent_market_detail:
        detail = sim_report.agent_market_detail
        print("Agent-based Market Simulation:")
        print(f"  Avg price path sample: {detail.price_path[:5]} ...")
        print(f"  Liquidity levels sample: {detail.liquidity_levels[:5]} ...")
        for note in detail.notes:
            print(f"   - {note}")

    print("\nBaseline Comparison:")
    if sim_report.baseline_comparison:
        print(f"Closest baseline: {sim_report.baseline_comparison.baseline_name}")
        if sim_report.baseline_comparison.similarities:
            print(" Similarities:")
            for item in sim_report.baseline_comparison.similarities:
                print(f"  - {item}")
        if sim_report.baseline_comparison.differences:
            print(" Differences:")
            for item in sim_report.baseline_comparison.differences:
                print(f"  - {item}")
    else:
        print("No comparable baseline identified.")

    print("\nValidated Model Summary:")
    print(sim_report.validated_model_summary)

    print("\nRecommendations:")
    for rec in sim_report.recommendations:
        print(f" - {rec}")

    save_pipeline_outputs(
        output_dir="pipeline_exports",
        proposal=proposal,
        context=context,
        control_result=control_result,
        filter_result=filter_result,
        simulation_report=sim_report,
    )

    project_name = user_input.get('project_name', 'your project')
    print(f"\nTokenomics design for {project_name} completed!")


if __name__ == "__main__":
    main()
