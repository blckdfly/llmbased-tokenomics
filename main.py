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
from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict, Counter
import statistics

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

@dataclass
class AllocationBucket:
    name: str
    percentage: float
    category: Optional[str] = None
    cliff_months: Optional[int] = None
    vesting_months: Optional[int] = None
    sub_allocations: List['AllocationBucket'] = field(default_factory=list)


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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_allocation_bucket(data: Dict) -> Optional[AllocationBucket]:
    try:
        name = data.get("category") or data.get("name") or "Unknown"
        percentage = float(data.get("percentage", 0))
        cliff = data.get("cliff_months")
        vesting = data.get("vesting_months")
        
        bucket = AllocationBucket(
            name=name,
            percentage=percentage,
            category=normalize_allocation_key(name),
            cliff_months=cliff,
            vesting_months=vesting,
        )
        
        children = data.get("children", [])
        if isinstance(children, list):
            for child_data in children:
                child_bucket = parse_allocation_bucket(child_data)
                if child_bucket:
                    bucket.sub_allocations.append(child_bucket)
                    
        return bucket
    except (TypeError, ValueError):
        return None


def allocations_from_payload(payload: Dict, raw_text: str) -> Tuple[List[AllocationBucket], List[str]]:
    notes: List[str] = []
    allocations_data = payload.get("allocations") or payload.get("token_allocations")
    allocations: List[AllocationBucket] = []

    if isinstance(allocations_data, list):
        for item in allocations_data:
            bucket = parse_allocation_bucket(item)
            if bucket:
                allocations.append(bucket)
                rationale = item.get("rationale")
                if rationale:
                    notes.append(f"{bucket.name}: {rationale}")
    
    if not allocations:
        pass

    return allocations, notes


def enforce_proposal_constraints(proposal: TokenomicsProposal,
                                 context: ProjectContext) -> TokenomicsProposal:
    notes = proposal.interpretability_notes[:]
    total = sum(bucket.percentage for bucket in proposal.allocations)
    if total <= 0:
        notes.append("Allocation sum was zero; defaulting to equal split.")
        equal_share = 100 / max(len(proposal.allocations), 1)
        proposal.allocations = [replace(bucket, percentage=round(equal_share, 2)) for bucket in proposal.allocations]
    elif 0.1 < abs(total - 100) <= 2.0: # Only normalize small deviations (up to 2%)
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
    if not payload:
         # In Strict JSON mode, if no payload is found, we should flag an error or try to parse the whole text as JSON if possible
         try:
             payload = json.loads(result_text)
         except json.JSONDecodeError:
             print("Error: content is not valid JSON and no JSON block found.")
             payload = {}

    allocations, notes = allocations_from_payload(payload, result_text)
    
    # Use JSON values strictly
    initial_supply = payload.get('initial_supply')
    if initial_supply is None:
         # Fallback only if JSON valid but field missing
         initial_supply = extract_total_supply(result_text)

    # Use 'rationale' field if available, otherwise raw text
    raw_rationale = payload.get('rationale') or result_text

    unlock_strategy = payload.get('unlock_strategy') if isinstance(payload, dict) else infer_unlock_strategy(result_text)
    emissions_model = payload.get('emissions_model') if isinstance(payload, dict) else infer_emissions_model(result_text)
    burn_mechanism = payload.get('burn_mechanism') if isinstance(payload, dict) else infer_burn_mechanism(result_text)

    proposal = TokenomicsProposal(
        project_name=payload.get("project_name") or user_input.get("project_name", "Unnamed Project"),
        token_symbol=payload.get("token_symbol") or user_input.get("token_symbol", "TKN"),
        initial_supply=initial_supply,
        allocations=allocations,
        unlock_strategy=unlock_strategy,
        emissions_model=emissions_model,
        burn_mechanism=burn_mechanism,
        raw_text=raw_rationale,
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
        You MUST output a SINGLE valid JSON object containing the tokenomics design.
        Do NOT include markdown formatting (like ```json), introduction, or conclusion outside the JSON.
        
        The JSON structure must be:
        {{
          "project_name": "Name",
          "token_symbol": "SYMBOL",
          "initial_supply": 1000000000,
          "allocations": [
            {{
              "category": "Team",
              "percentage": 20.0,
              "cliff_months": 12,
              "vesting_months": 48,
              "children": [] 
            }},
            {{
              "category": "Community",
              "percentage": 50.0,
              "cliff_months": 0,
              "vesting_months": 0,
              "children": [
                 {{ "category": "Rewards", "percentage": 30.0, "cliff_months": 0, "vesting_months": 24 }}
              ]
            }}
          ],
          "unlock_strategy": "linear | cliff_then_linear | milestone_based",
          "emissions_model": "fixed | deflationary | inflationary | dynamic",
          "burn_mechanism": "fee_burn | buyback_burn | none",
          "rationale": "Markdown formatted explanation of the design choices..."
        }}
        
        Constraints:
        1. "allocations" must sum strict top-level items to 100%. 
        2. "children" sub-allocations must sum to their parent's percentage.
        3. "initial_supply" must be a number.
        4. "unlock_strategy" must be one of: linear, cliff_then_linear, milestone_based.
        5. "emissions_model" must be one of: fixed, deflationary, inflationary, dynamic.
        6. "burn_mechanism" must be one of: fee_burn, buyback_burn, none.
        7. "rationale" field should contain the human-readable explanation, logic, and breakdown.
        """
    return prompt


# Generic Prompt Engineering
def create_generic_prompt(user_input: Dict, project_summaries: str) -> str:
    prompt = f"""
        Given the narrative description of a Web3 project below, your task is to act as a diagnostic expert to EXTRACT key structured design variables and then GENERATE a tokenomics model.

        STEP 1: INFERENCE & EXTRACTION
        Based *strictly* on the user's description, infer the following variables. If not explicitly stated, make a reasonable assumption based on the project type:
        - Core Principles (e.g., decentralization, sustainability, rapid growth)
        - Token Purpose (e.g., governance, utility, payment, reward)
        - Core Functions (e.g., staking, voting, access)
        - Target Stakeholders (e.g., whales, retail, institutions, developers)

        STEP 2: TOKENOMICS DESIGN
        Using the extracted variables from Step 1, design a comprehensive tokenomics model.

        USER'S PROJECT DESCRIPTION:
        {user_input['project_description']}

        PROJECT NAME: {user_input.get('project_name', 'Extract from description if mentioned')}

        Reference Projects for Context:
        {project_summaries}

        Follow the Dr. Tokenomics Framework:
        1. Purpose & Positioning: Clearly state the inferred Purpose and Core Principles.
        2. Token utility: Define the token's role using the inferred Functions.
        3. Allocation Strategy:
           - Recommend a Total Supply.
           - Allocate tokens using "Category: XX%" format.
           - Ensure sum is exactly 100%.
           - Group stakeholders into: Team, Investors, Community/Ecosystem, Treasury/Reserve.
        4. Vesting & release: Define cliffs and vesting periods.
        5. Governance: Define the balance of power (Community vs Team).

         Format:
        You MUST output a SINGLE valid JSON object containing the tokenomics design.
        Do NOT include markdown formatting (like ```json), introduction, or conclusion outside the JSON.
        
        The JSON structure must be:
        {{
          "project_name": "Name",
          "token_symbol": "SYMBOL",
          "initial_supply": 1000000000,
          "allocations": [
            {{
              "category": "Team",
              "percentage": 20.0,
              "cliff_months": 12,
              "vesting_months": 48,
              "children": [] 
            }},
            {{
              "category": "Community",
              "percentage": 50.0,
              "cliff_months": 0,
              "vesting_months": 0,
              "children": [
                 {{ "category": "Rewards", "percentage": 30.0, "cliff_months": 0, "vesting_months": 24 }}
              ]
            }}
          ],
          "unlock_strategy": "linear | cliff_then_linear | milestone_based",
          "emissions_model": "fixed | deflationary | inflationary | dynamic",
          "burn_mechanism": "fee_burn | buyback_burn | none",
          "rationale": "Markdown formatted explanation of the design choices..."
        }}
        
        Constraints:
        1. "allocations" must sum strict top-level items to 100%. 
        2. "children" sub-allocations must sum to their parent's percentage.
        3. "initial_supply" must be a number.
        4. "unlock_strategy" must be one of: linear, cliff_then_linear, milestone_based.
        5. "emissions_model" must be one of: fixed, deflationary, inflationary, dynamic.
        6. "burn_mechanism" must be one of: fee_burn, buyback_burn, none.
        7. "rationale" field should contain the human-readable explanation, logic, and breakdown.
        """
    return prompt

# OpenAI enhanced
def ask_openai_enhanced(prompt: str, input_type: str = "structured") -> str:
    # Unified Persona: Dr. Tokenomics
    system_message = """
            You are Dr. Tokenomics, a world-renowned blockchain economist and tokenomics architect with over a decade of experience designing sustainable token economies for projects across DeFi, GameFi, infrastructure protocols, DAOs, and beyond.

            You ground all of your reasoning in the Token Design Thinking framework (Token Kitchen / Shermin Voshmgir). You understand that token design is *not* only about math or price action, but about socio-technical systems, governance, and power structures.

            Whenever you design or critique a token model, you mentally walk through the following lenses:

            1. PURPOSE  
            - Clarify the core PURPOSE of the project (single-purpose vs multi-purpose vs unclear).  
            - Identify for whom the system is built and who it is *not* for.  

            2. PRINCIPLES & VALUES  
            - Extract the project’s mission, vision, and guiding PRINCIPLES.  
            - Make explicit which values, worldviews, and constraints should shape the token system.

            3. STAKEHOLDERS & ALLOCATIONS
            - Identify key STAKEHOLDER types (Team, Investors, Community, Foundation).
            - Balance incentives to avoid governance capture.

            4. ECONOMIC DESIGN TOOLBOX
            - Supply: fixed, capped, elastic, inflationary, or dynamically adjustable.  
            - Issuance & Distribution: genesis allocation, emissions schedule, vesting, lockups, and release patterns.  
            - Sinks & Fees: how tokens leave circulation.

            5. POWER STRUCTURES
            - Explicitly analyze POWER in the system (Voting vs Econ Power).
            - Evaluate whether the token design reinforces or counterbalances centralization.

            Your design philosophy emphasizes:
            - Long-term sustainability and socio-technical robustness over short-term speculation.  
            - Clear utility–value relationships where tokens have meaningful, coherent roles.  
            - Progressive decentralization and power rebalancing aligned with community readiness.  
            - Transparent articulation of trade-offs rather than pretending there is a single “optimal” design.  

            Your communication style is precise, implementation-focused, and analytical. You:
            - Provide actionable, well-reasoned recommendations ready for deployment or prototyping.  
            - Explain how and *why* your suggestions follow from the Token Design Thinking framework.  
            - Use qualitative and, where possible, quantitative/precedent-backed arguments to validate mechanisms.  

            All responses must be professional, comprehensive, and directly actionable for both technical and strategic teams.  
        """
    
    primary_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    fallback_model = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o-mini")

    def call_model(model_name: str, max_tokens: int = 2000, json_mode: bool = True) -> str:
        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
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

def extract_allocation_enhanced(text: str) -> Tuple[List[str], List[float]]:

    return [], []

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


def plot_behavior_metrics(sim_output: 'SimulationReport', token_symbol: str):
    try:
        # Extract data from the first scenario (usually Base Case)
        if not sim_output.scenarios:
             print("No simulation scenarios to plot.")
             return
             
        # Find Base Case or just take first
        scenario = next((s for s in sim_output.scenarios if "Base" in s.name), sim_output.scenarios[0])
        
        output = scenario.raw_output
        if not output:
             # Try to gracefully handle missing raw output by just skipping plot
             print("Raw simulation output not available. Skipping advanced behavior plots.")
             return

        months = output.months
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Supply Dynamics
        axes[0, 0].plot(months, output.circulating_supply_history, color='blue', label='Circulating')
        axes[0, 0].plot(months, output.total_supply_history, color='gray', linestyle='--', label='Total')
        axes[0, 0].set_title(f'Supply Dynamics ({token_symbol})')
        axes[0, 0].set_ylabel('Tokens')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Insider Governance Dominance
        axes[0, 1].plot(months, output.insider_share_history, color='red', linewidth=2, label='Insider Share')
        axes[0, 1].axhline(y=0.50, color='black', linestyle='--', linewidth=1, label='51% Attack Threshold')
        axes[0, 1].set_title('Insider Governance Dominance')
        axes[0, 1].set_ylabel('Share of Supply')
        axes[0, 1].set_ylim(0, 1.0)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Wealth Concentration (Gini)
        axes[1, 0].plot(months, output.gini_history, color='purple', label='Gini Coeff')
        axes[1, 0].set_title('Wealth Concentration (Gini Protocol)')
        axes[1, 0].set_ylabel('Gini (0=Equal, 1=Centralized)')
        axes[1, 0].set_ylim(0, 1.0)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Governance Concentration Proxy
        axes[1, 1].plot(months, output.governance_concentration_history, color='green', label='Gov Concentration')
        axes[1, 1].set_title('Governance Concentration')
        axes[1, 1].set_ylabel('Concentration Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"behavior_metrics_{token_symbol.lower()}.png"
        plt.savefig(filename)
        print(f"Advanced behavior charts saved as: {filename}")
        plt.close()

    except Exception as e:
        print(f"Error creating behavior plots: {e}")
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
        'community/ecosystem': 'Community',
        'ecosystem': 'Community',
        'ecosystem_fund': 'Community',
        'community_ecosystem': 'Community',
        
        # Investor variations
        'investors': 'Investors',
        'private_sale': 'Investors',
        'seed': 'Investors',
        'series_a': 'Investors',
        'strategic': 'Investors',
        
        # Treasury/Reserve variations (NOT counted as Community)
        'treasury': 'Treasury',
        'treasury/reserve': 'Treasury',
        'dao_treasury': 'Treasury',
        'reserve': 'Treasury',
        'reserves': 'Treasury',
        'contingency': 'Treasury',
        'development': 'Treasury',
        
        # Advisors variations
        'advisors': 'Advisors',
        'advisory': 'Advisors',
        
        # Marketing variations
        'marketing': 'Marketing',
        'partnerships': 'Marketing',
        
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
    passed: bool
    errors: List[str]
    warnings: List[str]
    validations_passed: List[str] = field(default_factory=list)
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
    errors: List[str] = []
    warnings: List[str] = []
    validations: List[str] = []

    # Allocation Integrity (Sum = 100%)
    total_allocation = sum(bucket.percentage for bucket in proposal.allocations)
    if abs(total_allocation - 100.0) > 0.5:
        errors.append(f"Allocations sum to {total_allocation:.2f}%. Must be 100% (±0.5%).")
    else:
        validations.append(f"Allocation Sum Check: {total_allocation:.2f}% (Passed 100% Target)")

    # Initial Supply Validity
    if proposal.initial_supply is None or proposal.initial_supply <= 0:
        errors.append("Initial supply must be positive.")
    else:
        validations.append("Initial Supply Check: Positive Value (Passed)")

    # Vesting Logic (Cliff <= Vesting)
    vesting_issues = False
    for bucket in proposal.allocations:
        if bucket.cliff_months and bucket.vesting_months and bucket.cliff_months > bucket.vesting_months:
            errors.append(f"{bucket.name} cliff ({bucket.cliff_months}m) > vesting ({bucket.vesting_months}m).")
            vesting_issues = True
    if not vesting_issues:
        validations.append("Vesting Schedule Logic: Cliffs <= Vesting Periods (Passed)")

    # Stakeholder Constraints
    category_totals = _aggregate_allocations_by_category(proposal.allocations)
    community_share = category_totals.get("Community", 0.0)
    team_share = category_totals.get("Team", 0.0)
    investor_share = category_totals.get("Investors", 0.0)
    insider_share = team_share + investor_share

    # 1. Community share >= min threshold (default 20%)
    min_community = context.constraints.get("min_community_share", 20.0)
    if community_share < min_community:
        warnings.append(f"Community share {community_share:.1f}% < min {min_community}%.")
    else:
        validations.append(f"Community Share: {community_share:.1f}% >= {min_community}% (Passed)")

    # 2. Team share <= max threshold (default 35%)
    max_team = context.constraints.get("max_team_share", 35.0)
    if team_share > max_team:
        warnings.append(f"Team share {team_share:.1f}% > max {max_team}%.")
    else:
        validations.append(f"Team Share: {team_share:.1f}% <= {max_team}% (Passed)")

    # 3. Investor share <= max threshold (default 40%)
    max_investor = context.constraints.get("max_investor_share", 40.0)
    if investor_share > max_investor:
        warnings.append(f"Investor share {investor_share:.1f}% > max {max_investor}%.")
    else:
        validations.append(f"Investor Share: {investor_share:.1f}% <= {max_investor}% (Passed)")

    # 4. Insider share <= 70% (Team + Investors)
    if insider_share > 70.0:
        severity = "CRITICAL" if insider_share > 90 else "Warning"
        msg = f"Insider share {insider_share:.1f}% exceeds 70% threshold."
        if severity == "CRITICAL":
            errors.append(msg)
        else:
            warnings.append(msg)
    else:
        validations.append(f"Insider Share: {insider_share:.1f}% <= 70% (Passed)")

    # 5. Insider vesting >= 12 months (Team and Investors)
    for bucket in proposal.allocations:
        if bucket.category in {"Team", "Investors"}:
            vesting = bucket.vesting_months or 0
            if vesting < 12:
                warnings.append(f"{bucket.name} vesting ({vesting}m) < 12 months minimum.")

    fairness_metrics = calculate_temporal_fairness(proposal)
    governance_risk = evaluate_governance_risk(proposal, fairness_metrics)

    return ControlLayerResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        validations_passed=validations,
        fairness_metrics=fairness_metrics,
        governance_risk=governance_risk,
    )

def print_control_layer_report(proposal: TokenomicsProposal, control_result: 'ControlLayerResult') -> None:
    category_totals = _aggregate_allocations_by_category(proposal.allocations)
    community_share = category_totals.get("Community", 0.0)
    team_share = category_totals.get("Team", 0.0)
    investor_share = category_totals.get("Investors", 0.0)
    insider_share = team_share + investor_share
    
    vesting_ok = all(
        (b.cliff_months or 0) <= (b.vesting_months or 0) or (b.vesting_months or 0) == 0
        for b in proposal.allocations
    )
    
    print("\Control Layer")
    print(f"  Community Share:  {community_share:.1f}%")
    print(f"  Team Share:       {team_share:.1f}%")
    print(f"  Investor Share:   {investor_share:.1f}%")
    print(f"  Insider Share:    {insider_share:.1f}%")
    print(f"  Vesting Logic:    {'OK' if vesting_ok else 'ERROR'}")
    
    if control_result.warnings:
        for warn in control_result.warnings:
            print(f"  [WARN] {warn}")
    if control_result.errors:
        for err in control_result.errors:
            print(f"  [ERROR] {err}")


def print_filter_layer_report(
    proposal: TokenomicsProposal,
    total_allocation: float,
    allocation_valid: bool,
    initial_supply_valid: bool,
    governance_balanced: bool,
    community_share: float,
    team_share: float,
    all_passed: bool
) -> None:
    print("\Filter Layer")
    print(f"  Allocation Sum:   {total_allocation:.2f}%")
    print(f"  Initial Supply:   {proposal.initial_supply:,}")
    print(f"  Governance Rule:  {community_share:.1f}% >= {team_share:.1f}%")
    print(f"  Status:           {'AUTHORIZED' if all_passed else 'BLOCKED'}")


def print_simulation_layer_report(proposal: TokenomicsProposal, sim_report: 'SimulationReport') -> None:
    print("\Simulation Layer (60 months, deterministic, price-free)")
    
    for scenario in sim_report.scenarios:
        supply = scenario.supply_over_time[-1]
        gini = scenario.raw_output.gini_history[-1]
        insider = scenario.raw_output.insider_share_history[-1] * 100
        print(f"  {scenario.name:<18} Supply={supply:,.0f} | Gini={gini:.3f} | Insider={insider:.1f}%")
    
    # Base case key indicators
    base = sim_report.scenarios[0]
    base_insider = base.raw_output.insider_share_history[-1] * 100
    base_gini = base.raw_output.gini_history[-1]
    gov_conc = sim_report.gini_layers.governance_gini
    
    print(f"\ Key Metrics (Base Case):")
    print(f"    Insider Share:  {base_insider:.1f}%")
    print(f"    Gini Coeff:     {base_gini:.3f}")
    print(f"    Governance:     {gov_conc:.3f}")

def build_explainer_prompt(
    proposal: TokenomicsProposal,
    control_result: 'ControlLayerResult',
    filter_passed: bool,
    simulation_report: 'SimulationReport'
) -> str:
    category_totals = _aggregate_allocations_by_category(proposal.allocations)
    community_share = category_totals.get("Community", 0.0)
    team_share = category_totals.get("Team", 0.0)
    investor_share = category_totals.get("Investors", 0.0)
    insider_share = team_share + investor_share
    
    # Simulation metrics
    base_scenario = simulation_report.scenarios[0] if simulation_report.scenarios else None
    final_supply = base_scenario.supply_over_time[-1] if base_scenario else 0
    final_gini = simulation_report.gini_layers.overall_gini
    final_insider = base_scenario.raw_output.insider_share_history[-1] if base_scenario else 0
    gov_concentration = simulation_report.gini_layers.governance_gini
    
    # Build allocation summary
    allocations_text = "\n".join([
        f"  - {b.name}: {b.percentage}% (cliff: {b.cliff_months or 0}m, vesting: {b.vesting_months or 0}m)"
        for b in proposal.allocations
    ])
    
    # Build scenario results
    scenarios_text = "\n".join([
        f"  - {s.name}: Supply={s.supply_over_time[-1]:,.0f}, Gini={s.raw_output.gini_history[-1]:.3f}, Insider={s.raw_output.insider_share_history[-1]*100:.1f}%"
        for s in simulation_report.scenarios
    ])
    
    prompt = f"""You are acting as an Analysis Explanation Engine for an academic tokenomics system.

        IMPORTANT CONSTRAINTS:
        - You are NOT designing tokenomics.
        - You are NOT simulating anything.
        - You MUST NOT invent new numbers.
        - All numerical values provided are FINAL and MUST be treated as ground truth.

        Your sole task is to explain the outputs of an already executed pipeline in clear, academic prose.

        INPUT DATA:

        1. TOKENOMICS PROPOSAL
        Project: {proposal.project_name or 'Unnamed'}
        Token Symbol: {proposal.token_symbol or 'N/A'}
        Initial Supply: {proposal.initial_supply:,} tokens
        Allocations:
        {allocations_text}

        2. CONTROL LAYER RESULTS (Diagnostic Validation)
        Community Share: {community_share:.1f}%
        Team Share: {team_share:.1f}%
        Investor Share: {investor_share:.1f}%
        Insider Share (Team + Investors): {insider_share:.1f}%
        Structural Errors: {len(control_result.errors)}
        Warnings: {len(control_result.warnings)}

        3. FILTER LAYER RESULTS (Hard Constraints)
        Allocation Sum: {sum(b.percentage for b in proposal.allocations):.2f}%
        Initial Supply Valid: {proposal.initial_supply > 0}
        Governance Balance (Community >= Team): {community_share >= team_share}
        Filter Passed: {filter_passed}

        4. SIMULATION OUTPUTS (60-month State-Based Simulation)
        {scenarios_text}
        Final Circulating Supply (Base): {final_supply:,.0f}
        Final Insider Share: {final_insider*100:.1f}%
        Final Gini Coefficient: {final_gini:.3f}
        Governance Concentration Index: {gov_concentration:.3f}

        YOUR TASK:

        Write a SHORT interpretive summary (2-3 paragraphs MAXIMUM) focusing ONLY on:
        1. What the health flags mean: Is insider share healthy? Is Gini concerning? Is governance decentralized?
        2. Key risks or strengths: What should a reviewer pay attention to?
        3. Brief structural assessment: Does the allocation structure support long-term sustainability?

        CRITICAL CONSTRAINTS:
        - DO NOT restate numbers that were already printed above
        - DO NOT explain what each layer does (the numeric output already shows this)
        - DO NOT write section headers or bullet points
        - BE CONCISE: Maximum 150-200 words total
        - Focus on INTERPRETATION and IMPLICATIONS only"""

    return prompt


def generate_analysis_explanation(
    proposal: TokenomicsProposal,
    control_result: 'ControlLayerResult',
    filter_passed: bool,
    simulation_report: 'SimulationReport'
) -> str:
    prompt = build_explainer_prompt(proposal, control_result, filter_passed, simulation_report)
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    
    # System message explicitly constrains the LLM to explanation-only role
    system_message = """You are an academic writing assistant specializing in blockchain and tokenomics analysis.
        Your role is to explain technical analysis results in clear, formal academic prose.
        You must not invent data or make claims beyond what is provided.
        You must not suggest design changes or optimizations.
        Write in a style suitable for academic publications or thesis chapters."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=1500,
        )
        return response.choices[0].message.content if response.choices else ""
    except Exception as e:
        return f"[Analysis explanation generation failed: {e}]"

@dataclass
class AdvisoryLayerResult:
    recommendations: List[str]


def run_advisory_layer(proposal: TokenomicsProposal,
                       context: ProjectContext) -> AdvisoryLayerResult:
    tips: List[str] = []

    category_totals = _aggregate_allocations_by_category(proposal.allocations)
    team_share = category_totals.get("Team", 0.0)
    community_share = category_totals.get("Community", 0.0)
    
    # Governance Balance Advice
    if community_share < team_share:
        tips.append("Recommendation: Increase Community share to exceed Team share for better decentralization optics.")

    # Vesting Optimization
    for bucket in proposal.allocations:
        if bucket.category in {"Team", "Investors"}:
            if (bucket.vesting_months or 0) < 24:
                tips.append(f"Recommendation: Extend {bucket.name} vesting to at least 24 months to signal long-term alignment.")
            if (bucket.cliff_months or 0) < 6:
                tips.append(f"Recommendation: Add at least 6-month cliff for {bucket.name}.")

    return AdvisoryLayerResult(recommendations=tips)

@dataclass
class ScenarioResult:
    name: str
    description: str
    supply_over_time: List[float]
    notes: List[str]
    raw_output: Optional[Any] = None # Holds the full SimulationOutput object


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
    agent_market_detail: Optional[Any] = None


from advanced_simulations import SimulationEngine, SimulationOutput

def run_simulations_layer(proposal: TokenomicsProposal,
                          knowledge_base: List[Dict],
                          context: ProjectContext) -> SimulationReport:
    # Setup Simulation Engine
    sim_allocations = [asdict(b) for b in proposal.allocations]
    
    # Map emissions_model to emission rate
    emissions_map = {
        "fixed": 0.03,          # 3% annual - stable supply
        "deflationary": 0.01,   # 1% annual - minimal emissions
        "inflationary": 0.08,   # 8% annual - growth focused
        "dynamic": 0.05,        # 5% annual - adaptive (default)
    }
    base_emission = emissions_map.get(proposal.emissions_model or "dynamic", 0.05)
    
    # Map burn_mechanism to burn rate
    burn_map = {
        "fee_burn": 0.02,       # 2% annual - active burn
        "buyback_burn": 0.015,  # 1.5% annual - buyback & burn
        "none": 0.0,            # 0% - no burn
    }
    base_burn = burn_map.get(proposal.burn_mechanism or "none", 0.005)
    
    # We will run 3 key scenarios by tweaking context signals
    scenarios_config = [
        ("Base Case", {"emission_rate": base_emission, "burn_rate": base_burn}),
        ("High Demand (Bull)", {"emission_rate": base_emission * 0.8, "burn_rate": base_burn * 2.0, "demand_multiplier": 1.5}),
        ("Low Demand (Bear)", {"emission_rate": base_emission * 1.2, "burn_rate": base_burn * 0.5, "demand_multiplier": 0.5})
    ]
    
    scenario_results = []
    base_output = None
    
    for name, signals_override in scenarios_config:
        # Merge context
        run_signals = context.economic_signals.copy()
        run_signals.update(signals_override)
        
        engine = SimulationEngine(
            initial_supply=proposal.initial_supply or 1_000_000_000,
            allocations=sim_allocations,
            context_signals=run_signals
        )
        output = engine.run(months=60)
        
        if name == "Base Case":
            base_output = output
            
        scenario_results.append(
            ScenarioResult(
                name=name,
                description=f"Simulation under {name} conditions.",
                supply_over_time=output.circulating_supply_history, # Visualizing Circulating Supply
                notes=[
                    f"Final Circulating: {output.circulating_supply_history[-1]:,.0f}",
                    f"Final Gini: {output.gini_history[-1]:.3f}",
                f"Insider Share: {output.insider_share_history[-1]*100:.1f}%"
                ],
                raw_output=output
            )
        )
    
    # Use Base Case for reporting details
    if not base_output:
        base_output = scenario_results[0] if scenario_results else None

    # Extract Metrics for Report
    months = base_output.months
    circ_supply = base_output.circulating_supply_history
    total_supply = base_output.total_supply_history
    locked_supply = [t - c for t, c in zip(total_supply, circ_supply)]
    burned_supply = [ (proposal.initial_supply or 1_000_000_000) - t for t in total_supply] # Approx
    
    supply_dynamics = SupplyDynamicsResult(
        months=months,
        circulating_supply=circ_supply,
        locked_supply=locked_supply,
        burned_supply=burned_supply
    )
    
    gini_result = GiniLayerResult(
        overall_gini=base_output.gini_history[-1],
        circulating_gini=base_output.gini_history[-1], # Simplified for now
        governance_gini=base_output.governance_concentration_history[-1]
    )

    # Fairness & Risk (Recalculate or use last)
    fairness_metrics = calculate_temporal_fairness(proposal)
    governance_risk = evaluate_governance_risk(proposal, fairness_metrics)
    
    fairness_report = (
        f"Fairness Analysis (Base Case T60):\n"
        f"Gini Coefficient: {gini_result.overall_gini:.3f}\n"
        f"Insider Share: {base_output.insider_share_history[-1]*100:.1f}%\n"
        f"Governance Concentration (Proxy): {gini_result.governance_gini:.3f}"
    )

    # Recommendations
    recommendations = []
    if base_output.insider_share_history[-1] > 0.5:
        recommendations.append("High insider concentration persists at year 5. Consider faster dilution or broader community incentives.")
    if len(base_output.circulating_supply_history) > 12 and (base_output.circulating_supply_history[12] / (proposal.initial_supply or 1)) > 0.4:
        recommendations.append("High year 1 inflation detected. Review vesting cliffs.")

    return SimulationReport(
        scenarios=scenario_results,
        gini_layers=gini_result,
        supply_dynamics=supply_dynamics,
        baseline_comparison=None,
        fairness_report=fairness_report,
        validated_model_summary="Model simulated using State-Mechanics engine. Dynamics appear stable under base assumptions.",
        recommendations=recommendations,
        fairness_metrics=fairness_metrics,
        governance_risk=governance_risk,
        real_project_validation=None,
        agent_market_detail=None 
    )




def save_pipeline_outputs(output_dir: str,
                          proposal: TokenomicsProposal,
                          context: ProjectContext,
                          control_result: ControlLayerResult,
                          advisory_result: AdvisoryLayerResult,
                          simulation_report: Optional[SimulationReport] = None) -> None:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    proposal_path = os.path.join(output_dir, f"proposal_{timestamp}.json")
    context_path = os.path.join(output_dir, f"context_{timestamp}.json")
    control_path = os.path.join(output_dir, f"control_{timestamp}.json")
    advisory_path = os.path.join(output_dir, f"advisory_{timestamp}.json")

    # Create clean proposal dict (exclude redundant fields)
    proposal_dict = asdict(proposal)
    proposal_dict.pop("json_payload", None)  # Remove duplicate LLM raw output
    proposal_dict.pop("raw_text", None)      # Remove long rationale text (already in json_payload if needed)
    
    with open(proposal_path, "w", encoding="utf-8") as f:
        json.dump(proposal_dict, f, ensure_ascii=False, indent=2)

    with open(context_path, "w", encoding="utf-8") as f:
        json.dump(asdict(context), f, ensure_ascii=False, indent=2)

    control_payload = {
        "passed": control_result.passed,
        "errors": control_result.errors,
        "warnings": control_result.warnings
    }
    with open(control_path, "w", encoding="utf-8") as f:
        json.dump(control_payload, f, ensure_ascii=False, indent=2)

    advisory_payload = {
        "recommendations": advisory_result.recommendations
    }
    with open(advisory_path, "w", encoding="utf-8") as f:
        json.dump(advisory_payload, f, ensure_ascii=False, indent=2)

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

    # Retry loop for filter failures (max 3 attempts)
    MAX_ATTEMPTS = 3
    attempt = 0
    all_constraints_met = False
    
    while attempt < MAX_ATTEMPTS and not all_constraints_met:
        attempt += 1
        
        if attempt > 1:
            print(f"\nAttempt {attempt}/{MAX_ATTEMPTS} - Regenerating tokenomics design...")
        else:
            print("\nGenerating token design...")
            
        result = ask_openai_enhanced(prompt, input_type)

        # Parse the result to display in clean format
        try:
            design_data = json.loads(result) if isinstance(result, str) else result

            print(f"  Project:    {design_data.get('project_name', 'N/A')}")
            print(f"  Symbol:     {design_data.get('token_symbol', 'N/A')}")
            print(f"  Supply:     {design_data.get('initial_supply', 0):,}")
            
            print("\n  Allocations:")
            for alloc in design_data.get('allocations', []):
                cat = alloc.get('category', 'Unknown')
                pct = alloc.get('percentage', 0)
                cliff = alloc.get('cliff_months', 0)
                vest = alloc.get('vesting_months', 0)
                print(f"    {cat:<20} {pct:>5.1f}%  (cliff: {cliff}m, vest: {vest}m)")
                
                # Show children if any
                for child in alloc.get('children', []):
                    child_cat = child.get('category', 'Sub')
                    child_pct = child.get('percentage', 0)
                    child_cliff = child.get('cliff_months', 0)
                    child_vest = child.get('vesting_months', 0)
                    print(f"      └─ {child_cat:<16} {child_pct:>5.1f}%  (cliff: {child_cliff}m, vest: {child_vest}m)")
            
            if design_data.get('rationale'):
                print(f"\n  Rationale: {design_data['rationale'][:200]}...")
                
        except (json.JSONDecodeError, TypeError):
            # Fallback to raw output if parsing fails
            print("Tokenomics Design")
            print(result)

        # Build proposal/context from LLM JSON
        proposal, context = generate_tokenomics_proposal(user_input, result)
        
        # Generate pie chart from structured allocations (not regex parsing!)
        if proposal.allocations:
            labels = [bucket.name for bucket in proposal.allocations]
            values = [bucket.percentage for bucket in proposal.allocations]
            token_symbol = proposal.token_symbol or 'TOKEN'
            
            if len(values) > 1:
                create_allocation_pie_chart(labels, values, token_symbol)
            else:
                print("Warning: Only one allocation category found.")
        else:
            print("Error: No allocations found in proposal. JSON may be malformed.")
            print("Please review the AI response manually.")
            return

        control_result = run_control_layer(proposal, context)
        
        # Print Control Layer report (structured output)
        print_control_layer_report(proposal, control_result)
        
        # Check filter constraints
        category_totals = _aggregate_allocations_by_category(proposal.allocations)
        community_share = category_totals.get("Community", 0.0)
        team_share = category_totals.get("Team", 0.0)
        total_allocation = sum(b.percentage for b in proposal.allocations)
        initial_supply_valid = (proposal.initial_supply or 0) > 0
        governance_balanced = community_share >= team_share
        allocation_valid = abs(total_allocation - 100.0) <= 0.5
        all_constraints_met = allocation_valid and initial_supply_valid and governance_balanced and control_result.passed
        
        # Print Filter Layer report (structured output)
        print_filter_layer_report(
            proposal=proposal,
            total_allocation=total_allocation,
            allocation_valid=allocation_valid,
            initial_supply_valid=initial_supply_valid,
            governance_balanced=governance_balanced,
            community_share=community_share,
            team_share=team_share,
            all_passed=all_constraints_met
        )
        
        if not all_constraints_met and attempt < MAX_ATTEMPTS:
            print(f"\nCritical issues detected. Triggering recalculation...")
    
    # After all attempts
    if not all_constraints_met:
        print(f"\nFilter failed after {MAX_ATTEMPTS} attempts. Manual revision required.")
        save_pipeline_outputs("pipeline_exports", proposal, context, control_result, AdvisoryLayerResult(recommendations=[]), None)
        return
    
    # Advisory recommendations (non-blocking)
    advisory_result = run_advisory_layer(proposal, context)
    
    sim_report = run_simulations_layer(proposal, knowledge_base, context)
    
    # Print Simulation Layer report (structured output)
    print_simulation_layer_report(proposal, sim_report)
    
    # Print any recommendations
    if advisory_result.recommendations:
        print("\nOptimization Recommendations:")
        for rec in advisory_result.recommendations:
            print(f"  - {rec}")
    
    if sim_report.recommendations:
        print("\n  Recommendations:")
        for rec in sim_report.recommendations:
            print(f"    - {rec}")
    
    print("\nInterpretation:")
    explanation = generate_analysis_explanation(
        proposal=proposal,
        control_result=control_result,
        filter_passed=all_constraints_met,
        simulation_report=sim_report
    )
    print(explanation)
    
    # Generate charts
    plot_behavior_metrics(sim_report, token_symbol)
    
    # Save all outputs (proposal, control, advisory, simulation)
    save_pipeline_outputs("pipeline_exports", proposal, context, control_result, advisory_result, sim_report)
    
    print(f"\nCharts saved. Analysis complete for {user_input.get('project_name', 'project')}.")


if __name__ == "__main__":
    main()