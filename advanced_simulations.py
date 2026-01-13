import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

@dataclass
class BehaviorProfile:
    """Defines the selling behavior for a specific category of stakeholder."""
    alpha: float          # Base sell pressure per month (portion of unlocked)
    description: str = ""

@dataclass
class StakeholderState:
    name: str
    category: str         # Explicit category for behavior mapping
    holdings: float       # H_{g,t}: Tokens currently held (Legal + Liquid)
    unlocked: float       # U_{g,t}: Tokens unlocked and available to sell
    locked: float         # L_{g,t}: Tokens locked in vesting
    cumulative_sold: float = 0.0 # Track cumulative sold tokens

@dataclass
class SimulationOutput:
    months: List[int]
    total_supply_history: List[float]
    circulating_supply_history: List[float]
    gini_history: List[float]
    insider_share_history: List[float]
    governance_concentration_history: List[float]
    price_proxy: List[float]
    stakeholder_states: Dict[str, List[float]] # History of holdings per group

# Category weights for sell pressure (academically defensible heuristics)
CATEGORY_WEIGHTS = {
    "Team": 0.2,        # Founders hold long-term
    "Investors": 1.0,   # VCs take profit
    "Advisors": 0.3,    # Advisory tokens sold slowly
    "Community": 0.5,   # Mixed retail behavior
    "Rewards": 0.6,     # Reward recipients often sell
    "Staking": 0.1,     # Stakers are committed
    "Treasury": 0.0,    # Treasury never sells (strategic only)
    "Reserve": 0.0,     # Reserve never sells
}

def infer_behavior_profile(allocation: dict) -> BehaviorProfile:
    category = allocation.get('category', 'Community')
    vesting_months = allocation.get('vesting_months') or 0
    cliff_months = allocation.get('cliff_months') or 0
    
    # Base rate: 10% monthly if aggressive, unlocked seller
    base_rate = 0.10
    
    # Category weight lookup (default to Community if unknown)
    category_weight = CATEGORY_WEIGHTS.get(category, 0.5)
    
    # Vesting factor: longer vesting = more committed = less selling
    # Formula: 1 / (1 + vesting/24) ranges from 1.0 (no vesting) to ~0.29 (60m vesting)
    vesting_factor = 1.0 / (1.0 + vesting_months / 24.0)
    
    # Cliff bonus: longer cliff = more commitment (reduce by 10% per 6 months cliff)
    cliff_factor = max(0.5, 1.0 - (cliff_months / 60.0))
    
    # Final alpha calculation
    alpha = base_rate * category_weight * vesting_factor * cliff_factor
    
    # Ensure alpha is bounded [0, 0.15] (max 15% monthly sell)
    alpha = max(0.0, min(alpha, 0.15))
    
    description = f"Inferred: {category} (vest={vesting_months}m, cliff={cliff_months}m)"
    return BehaviorProfile(alpha=round(alpha, 4), description=description)


class SimulationEngine:
    
    def __init__(self, initial_supply: float, allocations: List[dict], context_signals: Dict[str, float]):
        self.initial_supply = initial_supply
        self.allocations = allocations
        self.signals = context_signals
        
        # Build dynamic behavior map from allocations
        self.behavior_map: Dict[str, BehaviorProfile] = {}
        for alloc in allocations:
            name = alloc.get('name') or alloc.get('category')
            self.behavior_map[name] = infer_behavior_profile(alloc)
        
        # Initialize stakeholder states
        self.stakeholders: Dict[str, StakeholderState] = {}
        for alloc in allocations:
            name = alloc.get('name') or alloc.get('category')
            percentage = alloc.get('percentage', 0)
            category = alloc.get('category', 'Community')
            cliff = alloc.get('cliff_months') or 0
            vesting = alloc.get('vesting_months') or 0
            
            amount = initial_supply * (percentage / 100.0)
            
            # TGE logic: no cliff and no vesting = fully liquid at launch
            if cliff == 0 and vesting == 0:
                unlocked, locked = amount, 0.0
            else:
                unlocked, locked = 0.0, amount
            
            self.stakeholders[name] = StakeholderState(
                name=name,
                category=category,
                holdings=amount,
                unlocked=unlocked,
                locked=locked,
                cumulative_sold=0.0
            )
        
        self.total_supply = initial_supply
        self.circulating_supply = sum(s.unlocked for s in self.stakeholders.values())
        self.burned_total = 0.0

    def _apply_vesting(self, t: int):

        for name, state in self.stakeholders.items():
            alloc = next(a for a in self.allocations if a['name'] == name)
            cliff = alloc.get('cliff_months') or 0
            vesting = alloc.get('vesting_months') or 1
            
            if t < cliff:
                continue
                
            # Calculate total vested fraction based on time elapsed
            total_alloc = state.holdings + state.cumulative_sold # Reconstruct original total
            
            if vesting <= 0:
                vested_fraction = 1.0 # Immediate unlock
            else:
                progress = min((t - cliff) / vesting, 1.0)
                vested_fraction = progress
            
            # Determine how much SHOULD be unlocked vs how much IS unlocked
            should_be_locked = total_alloc * (1.0 - vested_fraction)
            current_locked = state.locked
            
            unlock_amount = max(0, current_locked - should_be_locked)
            
            # State Transition: Locked -> Unlocked
            state.locked -= unlock_amount
            state.unlocked += unlock_amount

    def _apply_emissions(self, t: int):
        inflation_rate = self.signals.get("emission_rate", 0.05) / 12 # Monthly rate
        new_tokens = self.total_supply * inflation_rate
        
        self.total_supply += new_tokens
        
        # Distribution Rule: 
        # For simplicity, we assume emissions go to Stakers/Community.
        community_nodes = [s for n, s in self.stakeholders.items() if "Community" in n or "Reward" in n]
        if community_nodes:
            share = new_tokens / len(community_nodes)
            for node in community_nodes:
                node.holdings += share
                node.unlocked += share # Rewards are liquid immediately
        else:
            # If no specific bucket, adds to general circulation (dilutes everyone)
            self.circulating_supply += new_tokens

    def _apply_burn(self, t: int):
        """Mechanic 4: Burn (fee burn, buyback burn)"""
        burn_rate = self.signals.get("burn_rate", 0.005) / 12
        burn_amount = self.total_supply * burn_rate
        
        # Burn from circulating supply (fees paid by users)
        # finding distinct community/user stakeholders to burn from
        community_nodes = [s for n, s in self.stakeholders.items() if "Community" in n]
        if community_nodes:
             # distribute burn impact
             per_node = burn_amount / len(community_nodes)
             for node in community_nodes:
                 actual_burn = min(node.unlocked, per_node)
                 node.unlocked -= actual_burn
                 node.holdings -= actual_burn
        else:
            pass # No user balance to burn
            
        self.total_supply -= burn_amount
        self.burned_total += burn_amount

    def _apply_sell_pressure(self, t: int):
        
        # Determine Absorption Sink (Who buys? The public/community)
        market_sinks = [s for n, s in self.stakeholders.items() if "Community" in s.category or "Community" in n]
        market_sink = market_sinks[0] if market_sinks else None
        
        for name, state in self.stakeholders.items():
            if market_sink and state == market_sink:
                continue
                
            # Get inferred behavior profile for this stakeholder
            policy = self.behavior_map.get(name)
            alpha = policy.alpha if policy else 0.0
            
            # 2. Execute Sale
            sell_amount = state.unlocked * alpha
            
            # Safety Check: Cannot sell more than held
            sell_amount = min(sell_amount, state.holdings)
            
            if sell_amount > 0:
                # Seller loses tokens
                state.unlocked -= sell_amount
                state.holdings -= sell_amount
                state.cumulative_sold += sell_amount
                
                # Buyer (Market/Community) gains tokens
                if market_sink:
                    market_sink.unlocked += sell_amount
                    market_sink.holdings += sell_amount

    def _calculate_gini(self) -> float:
        balances = [s.holdings for s in self.stakeholders.values()]
        if not balances or sum(balances) == 0: return 0.0
        n = len(balances)
        sorted_b = sorted(balances)
        cum = sum(sorted_b)
        return (2 * sum((i+1) * v for i, v in enumerate(sorted_b))) / (n * cum) - (n+1)/n

    def _calculate_insider_share(self) -> float:
        # Insiders defined by Category, not just Name
        insider_categories = ['Team', 'Investors', 'Advisors', 'Foundation', 'Partners']
        
        insider_holdings = sum(s.holdings for n, s in self.stakeholders.items() 
                               if s.category in insider_categories or any(tag in n for tag in insider_categories))
                               
        current_supply = sum(s.holdings for s in self.stakeholders.values())
        return insider_holdings / current_supply if current_supply > 0 else 0

    def run(self, months: int = 60) -> SimulationOutput:
        history_S = []
        history_C = []
        history_Gini = []
        history_Insider = []
        history_Gov = []
        history_Price = [] # Proxy
        
        # Initial capture
        history_S.append(self.total_supply)
        self.circulating_supply = sum(s.unlocked for s in self.stakeholders.values())
        history_C.append(self.circulating_supply)
        history_Gini.append(self._calculate_gini())
        history_Insider.append(self._calculate_insider_share())
        history_Gov.append(history_Insider[-1]) # Proxy init
        
        market_demand_acc = 1.0 # Accumulator for price proxy
        
        for t in range(1, months + 1):
            self._apply_vesting(t)
            self._apply_emissions(t)
            self._apply_burn(t)
            self._apply_sell_pressure(t)
            
            # Recalculate System State
            self.circulating_supply = sum(s.unlocked for s in self.stakeholders.values())
            
            # Deterministic Price Proxy: Demand / Supply
            # Demand grows at fixed 2% monthly rate, scaled by scenario multiplier
            demand_growth = self.signals.get("demand_multiplier", 1.0) * 1.02  # 2% monthly growth
            market_demand_acc *= demand_growth  # No randomness - fully deterministic
            
            price_proxy = market_demand_acc / (self.circulating_supply if self.circulating_supply > 0 else 1)
            
            history_S.append(self.total_supply)
            history_C.append(self.circulating_supply)
            history_Gini.append(self._calculate_gini())
            history_Insider.append(self._calculate_insider_share())
            history_Gov.append(history_Insider[-1]) # Proxy for now
            history_Price.append(price_proxy)
            
        return SimulationOutput(
            months=list(range(months + 1)),
            total_supply_history=history_S,
            circulating_supply_history=history_C,
            gini_history=history_Gini,
            insider_share_history=history_Insider,
            governance_concentration_history=history_Gov,
            price_proxy=history_Price,
            stakeholder_states={n: [s.holdings] for n, s in self.stakeholders.items()} # Just final state snapshot or we could track history if needed. 
            # For simplicity in this implementation, I'll return empty history for stakeholders to save memory unless requested.
            # Actually, let's just return final states in a summary format downstream.
        )
