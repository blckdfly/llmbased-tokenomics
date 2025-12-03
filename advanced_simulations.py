import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class MarketScenarioDetail:
    name: str
    price_path: List[float]
    liquidity_levels: List[float]
    sentiment_trend: List[float]
    notes: List[str]


def simulate_price_impact(initial_price: float,
                          demand_shock: float,
                          liquidity_depth: float,
                          periods: int = 24) -> List[float]:
    """Simple square-root impact curve with mean reversion."""
    price = initial_price
    path = []
    k = 0.05 + max(0, demand_shock) * 0.02
    for t in range(periods):
        shock = np.random.normal(demand_shock, 0.5)
        impact = k * math.sqrt(abs(shock)) * (1 if shock >= 0 else -1)
        price = max(0.01, price * (1 + impact) * (1 - 0.3 / liquidity_depth))
        price = price * (0.98 + np.random.rand() * 0.04)  # small noise
        path.append(round(price, 4))
    return path


def simulate_liquidity_shocks(base_liquidity: float,
                              severity: str,
                              periods: int = 24) -> List[float]:
    severity_map = {
        "low": 0.05,
        "medium": 0.15,
        "high": 0.3,
    }
    sigma = severity_map.get(severity, 0.15)
    liquidity = base_liquidity
    levels = []
    for _ in range(periods):
        shock = np.random.normal(0, sigma)
        liquidity = max(0.05, liquidity * (1 - shock))
        levels.append(round(liquidity, 4))
    return levels


def run_agent_market_simulation(proposal_initial_supply: float,
                                community_share: float,
                                insider_share: float,
                                context_signals: Dict[str, float]) -> MarketScenarioDetail:
    base_price = context_signals.get("base_price", 1.0)
    demand_multiplier = context_signals.get("demand_multiplier", 1.0)
    burn_rate = context_signals.get("burn_rate", 0.01)

    liquidity_depth = max(0.2, community_share / max(insider_share, 1))
    demand_shock = demand_multiplier - burn_rate * 10

    price_path = simulate_price_impact(base_price, demand_shock, liquidity_depth)
    liquidity_levels = simulate_liquidity_shocks(liquidity_depth, "medium")

    sentiment = []
    for price, liquidity in zip(price_path, liquidity_levels):
        sentiment_score = min(1.0, max(0.0, price / base_price * liquidity))
        sentiment.append(round(sentiment_score, 4))

    notes = [
        f"Avg price {np.mean(price_path):.3f}",
        f"Liquidity floor {min(liquidity_levels):.3f}",
        f"Sentiment drift {sentiment[-1] - sentiment[0]:.3f}",
    ]

    return MarketScenarioDetail(
        name="Agent Market Simulation",
        price_path=price_path,
        liquidity_levels=liquidity_levels,
        sentiment_trend=sentiment,
        notes=notes,
    )
