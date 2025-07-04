import json
import os
import re
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics

# Set matplotlib backend for environments that don't support GUI
try:
    import tkinter
    matplotlib.use('TkAgg')  # Interactive backend
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
        system_message = """You are Dr. Tokenomics, a world-renowned blockchain economist and tokenomics architect with over a decade of experience designing sustainable token economies for projects across DeFi, GameFi, infrastructure protocols, DAOs, and beyond—many of which have achieved billions in market capitalization.

        You specialize in extracting key insights from narrative project descriptions and translating them into robust, incentive-aligned token economies. Your expertise includes pattern recognition, stakeholder alignment, and designing mechanisms resilient to market cycles and regulatory shifts.

        You excel at:
        - Designing sustainable token models that balance ecosystem incentives
        - Applying behavioral economics to encourage long-term participation
        - Navigating evolving global regulatory environments with compliant architectures

        Your design philosophy emphasizes:
        - Long-term sustainability over short-term speculation
        - Clear utility-value relationships where tokens have meaningful roles
        - Progressive decentralization that rewards community readiness

        Your communication style is precise, implementation-focused, and analytical. You:
        - Provide actionable, well-reasoned recommendations ready for deployment
        - Use quantitative and precedent-backed arguments to validate mechanisms
        - Present implementation plans in clear phases with measurable outcomes

        All responses must be professional, comprehensive, and directly actionable for both technical and strategic teams. Avoid vague or generic answers. Always deliver clarity, rigor, and practical value in every recommendation."""

    else:
        system_message = """You are a world-class blockchain tokenomics expert with experience designing token economies for top-tier Web3 projects such as Uniswap, Aave, Compound, and Chainlink. You deeply understand DeFi mechanics, governance design, token utility, and long-term sustainability.

        Your goal is to translate project goals and narratives into effective, incentive-aligned token economic models that are sustainable, secure, and growth-oriented. You align stakeholders, ensure regulatory awareness, and design clear utility flows that create value for users and the ecosystem.

        Always provide specific, actionable recommendations with clear reasoning, implementation guidance, and consideration of economic, technical, and governance impacts. Avoid generic responses—every answer must reflect deep domain expertise and strategic precision."""

    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2500,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Failed to call OpenAI API: {e}")
        return "Error generating recommendation."

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

def create_monte_carlo_visualization(mc_results: Dict, token_symbol: str):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Final supply distribution
    ax1.hist(mc_results['final_supplies'], bins=50, color='skyblue', 
             alpha=0.7, edgecolor='black')
    ax1.axvline(mc_results['mean_final_supply'], color='red', linestyle='--', 
                label=f"Mean: {mc_results['mean_final_supply']:,.0f}")
    ax1.axvline(mc_results['median_final_supply'], color='green', linestyle='--', 
                label=f"Median: {mc_results['median_final_supply']:,.0f}")
    ax1.set_title('Final Supply Distribution (12 months)')
    ax1.set_xlabel('Final Supply')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence intervals
    confidence_levels = [5, 25, 50, 75, 95]
    percentiles = [np.percentile(mc_results['final_supplies'], p) for p in confidence_levels]
    ax2.plot(confidence_levels, percentiles, 'o-', linewidth=2, markersize=8)
    ax2.set_title('Supply Confidence Intervals')
    ax2.set_xlabel('Percentile')
    ax2.set_ylabel('Final Supply')
    ax2.grid(True, alpha=0.3)
    
    # 3. Monthly burn simulation (sample)
    sample_burns = mc_results['monthly_burns'][:10]  # Show first 10 simulations
    months = range(1, 13)
    for i, burns in enumerate(sample_burns):
        ax3.plot(months, np.cumsum(burns), alpha=0.6, linewidth=1)
    ax3.set_title('Cumulative Burn Over Time (Sample Simulations)')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Cumulative Burned Tokens')
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk analysis
    risk_metrics = {
        'Mean': mc_results['mean_final_supply'],
        'Median': mc_results['median_final_supply'],
        '5th Percentile': mc_results['percentile_5'],
        '95th Percentile': mc_results['percentile_95']
    }
    
    ax4.bar(range(len(risk_metrics)), list(risk_metrics.values()), 
            color=['blue', 'green', 'red', 'red'], alpha=0.7)
    ax4.set_title('Risk Metrics Summary')
    ax4.set_xticks(range(len(risk_metrics)))
    ax4.set_xticklabels(list(risk_metrics.keys()), rotation=45)
    ax4.set_ylabel('Final Supply')
    
    plt.suptitle(f'Monte Carlo Analysis for {token_symbol.upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Use non-blocking display
    plt.show(block=False)
    plt.pause(0.1)
    input("\nMonte Carlo charts displayed. Press Enter to continue...")

def create_ab_testing_visualization(labels: List[str], values: List[float], ab_results: Dict, token_symbol: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # GPT allocation
    axes[0].pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[0].set_title(f'GPT Recommendation\n(Gini: {ab_results["gpt_gini"]:.3f})')
    
    # Baseline comparisons
    baseline_names = list(ab_results['baselines'].keys())
    for i, baseline_name in enumerate(baseline_names[:3]):
        baseline_data = ab_results['baselines'][baseline_name]
        baseline_labels = list(baseline_data['allocation'].keys())
        baseline_values = list(baseline_data['allocation'].values())
        
        axes[i+1].pie(baseline_values, labels=baseline_labels, autopct='%1.1f%%', startangle=90)
        axes[i+1].set_title(f'{baseline_name} Model\n(Gini: {baseline_data["gini"]:.3f})')
    
    plt.suptitle(f'A/B Testing: Token Allocation Comparison for {token_symbol.upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Use non-blocking display
    plt.show(block=False)
    plt.pause(0.1)
    input("A/B Testing charts displayed. Press Enter to continue...")

# Simulations Choice
def get_user_testing_choice() -> str:
    print("Analysis Options")
    print("1. A/B Testing - Compare your allocation with proven models")
    print("2. Monte Carlo Simulation - Predict token supply scenarios")
    print("3. Both Analysis - Get comprehensive insights")
    print("4. Knowledge Base Analysis - Detailed breakdown of reference projects")
    print("5. Skip Analysis - Just show the recommendation")
    
    while True:
        choice = input("\nSelect your analysis preference (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        print("Invalid choice. Please select 1, 2, 3, 4, or 5.")
        
def extract_allocations_from_knowledge_base(knowledge_base: List[Dict]) -> Dict:
    all_allocations = []
    allocation_keys = Counter()
    project_allocations = {}
    
    for project in knowledge_base:
        project_name = project.get('project', 'Unknown')
        tokenomics = project.get('tokenomics', {})
        allocation = tokenomics.get('allocation', {})
        
        if allocation and isinstance(allocation, dict):
            # Normalize allocation to percentages if they're in decimal format
            normalized_allocation = {}
            for key, value in allocation.items():
                try:
                    # Convert to float and handle both decimal (0.15) and percentage (15) formats
                    float_val = float(value)
                    if float_val <= 1.0:  # Assume decimal format
                        percentage = float_val * 100
                    else:  # Assume percentage format
                        percentage = float_val
                    
                    # Normalize key names for consistency
                    normalized_key = normalize_allocation_key(key)
                    normalized_allocation[normalized_key] = percentage
                    allocation_keys[normalized_key] += 1
                    
                except (ValueError, TypeError):
                    continue
            
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

def ab_testing_baseline(gpt_labels: List[str], gpt_values: List[float], knowledge_base: List[Dict]) -> Dict:
    # Generate dynamic baseline models
    kb_analysis = generate_dynamic_baseline_models(knowledge_base)
    baseline_models = kb_analysis['baseline_models']
    
    # Add original static models as fallback
    static_models = {
        "Static Conservative": {
            "Community": 30, "Team": 15, "Investors": 25, 
            "Advisors": 10, "Marketing": 10, "Reserve": 10
        },
        "Static Community-Focused": {
            "Community": 50, "Team": 20, "Investors": 15, 
            "Advisors": 5, "Marketing": 5, "Reserve": 5
        },
        "Static Balanced": {
            "Community": 40, "Team": 20, "Investors": 20, 
            "Advisors": 10, "Reserve": 10
        }
    }
    
    # Combine dynamic and static models
    all_baseline_models = {**baseline_models, **static_models}
    
    results = {}
    gpt_gini = calculate_gini_coefficient(gpt_values)
    
    for model_name, allocation in all_baseline_models.items():
        # Match GPT categories with baseline categories
        matched_values = []
        for gpt_label in gpt_labels:
            normalized_gpt_label = normalize_allocation_key(gpt_label)
            if normalized_gpt_label in allocation:
                matched_values.append(allocation[normalized_gpt_label])
            else:
                # Find closest match or use average
                closest_match = find_closest_category(normalized_gpt_label, allocation.keys())
                if closest_match:
                    matched_values.append(allocation[closest_match])
                else:
                    # Use average of remaining allocation
                    avg_value = sum(allocation.values()) / len(allocation)
                    matched_values.append(avg_value)
        
        if matched_values:
            baseline_gini = calculate_gini_coefficient(matched_values)
            distribution_score = abs(max(matched_values) - min(matched_values))
            
            results[model_name] = {
                'gini': baseline_gini,
                'distribution_score': distribution_score,
                'allocation': allocation,
                'matched_values': matched_values,
                'better_than_gpt': baseline_gini < gpt_gini
            }
    
    return {
        'gpt_gini': gpt_gini,
        'gpt_distribution_score': abs(max(gpt_values) - min(gpt_values)),
        'baselines': results,
        'kb_analysis': kb_analysis
    }

def find_closest_category(target: str, available_categories: List[str]) -> str:
    target_lower = target.lower()
    # Direct substring match
    for category in available_categories:
        if target_lower in category.lower() or category.lower() in target_lower:
            return category
    
    # Keyword matching
    if 'team' in target_lower or 'founder' in target_lower:
        for category in available_categories:
            if 'team' in category.lower():
                return category
    
    if 'community' in target_lower or 'public' in target_lower:
        for category in available_categories:
            if 'community' in category.lower():
                return category
    
    if 'investor' in target_lower or 'private' in target_lower:
        for category in available_categories:
            if 'investor' in category.lower():
                return category
    
    return None

def kb_analysis_result(kb_analysis: Dict):

    print("Knowledge Base Allocations Analysis")
    
    category_stats = kb_analysis['category_stats']
    allocations_data = kb_analysis['allocations_data']
    
    print(f"\nTotal projects analyzed: {len(allocations_data['all_allocations'])}")
    print(f"Most common allocation categories:")
    
    for category, stats in category_stats.items():
        print(f"\n{category}:")
        print(f"  • Appears in {stats['count']} projects")
        print(f"  • Average: {stats['mean']:.1f}%")
        print(f"  • Median: {stats['median']:.1f}%")
        print(f"  • Range: {stats['min']:.1f}% - {stats['max']:.1f}%")
    
    print(f"\nProject-by-project breakdown:")
    for project_name, allocation in allocations_data['project_allocations'].items():
        print(f"\n{project_name}:")
        for category, percentage in allocation.items():
            print(f"  • {category}: {percentage:.1f}%")
def save_final_supply_histogram(mc_results, token_symbol):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 6))
    plt.hist(mc_results['final_supplies'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    plt.axvline(mc_results['mean_final_supply'], color='red', linestyle='--', label=f"Mean: {mc_results['mean_final_supply']:,.0f}")
    plt.axvline(mc_results['median_final_supply'], color='green', linestyle='--', label=f"Median: {mc_results['median_final_supply']:,.0f}")
    plt.title('Final Supply Distribution (12 Months)')
    plt.xlabel('Final Supply')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"exported_charts/final_supply_{token_symbol}.png", dpi=600)
    plt.close()

def save_confidence_intervals(mc_results, token_symbol):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 6))
    confidence_levels = [5, 25, 50, 75, 95]
    percentiles = [np.percentile(mc_results['final_supplies'], p) for p in confidence_levels]
    plt.plot(confidence_levels, percentiles, 'o-', linewidth=2, markersize=8)
    plt.title('Supply Confidence Intervals')
    plt.xlabel('Percentile')
    plt.ylabel('Final Supply')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"exported_charts/confidence_intervals_{token_symbol}.png", dpi=600)
    plt.close()

def save_cumulative_burn(mc_results, token_symbol):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 6))
    months = range(1, 13)
    for burns in mc_results['monthly_burns'][:10]:
        plt.plot(months, np.cumsum(burns), alpha=0.6, linewidth=1)
    plt.title('Cumulative Burn Over Time (10 Sample Simulations)')
    plt.xlabel('Month')
    plt.ylabel('Cumulative Burned Tokens')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"exported_charts/cumulative_burn_{token_symbol}.png", dpi=600)
    plt.close()

def save_risk_metrics_bar(mc_results, token_symbol):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    risk_metrics = {
        'Mean': mc_results['mean_final_supply'],
        'Median': mc_results['median_final_supply'],
        '5th Percentile': mc_results['percentile_5'],
        '95th Percentile': mc_results['percentile_95']
    }
    plt.bar(risk_metrics.keys(), risk_metrics.values(), color=['blue', 'green', 'red', 'red'], alpha=0.7)
    plt.title('Risk Metrics Summary')
    plt.ylabel('Final Supply')
    plt.tight_layout()
    plt.savefig(f"exported_charts/risk_metrics_{token_symbol}.png", dpi=600)
    plt.close()

def save_ab_testing_charts(labels, values, ab_results, token_symbol):
    import matplotlib.pyplot as plt

    # Save GPT chart
    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'GPT Allocation (Gini: {ab_results["gpt_gini"]:.3f})')
    plt.tight_layout()
    plt.savefig(f"exported_charts/ab_gpt_allocation_{token_symbol}.png", dpi=600)
    plt.close()

    # Save baseline comparison charts
    for i, (baseline_name, baseline_data) in enumerate(ab_results['baselines'].items()):
        plt.figure(figsize=(8, 6))
        baseline_labels = list(baseline_data['allocation'].keys())
        baseline_values = list(baseline_data['allocation'].values())
        plt.pie(baseline_values, labels=baseline_labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'{baseline_name} (Gini: {baseline_data["gini"]:.3f})')
        plt.tight_layout()
        safe_name = baseline_name.replace(" ", "_").replace("/", "_")
        plt.savefig(f"exported_charts/ab_{safe_name.lower()}_{token_symbol}.png", dpi=600)
        plt.close()
        
def save_ab_testing_stacked_bar(labels, values, ab_results, token_symbol):
    import matplotlib.pyplot as plt
    import numpy as np

    models = ['GPT'] + list(ab_results['baselines'].keys())
    all_labels = sorted(set(labels + [lbl for b in ab_results['baselines'].values() for lbl in b['allocation'].keys()]))

    data = []
    for model in models:
        if model == 'GPT':
            model_dict = dict(zip(labels, values))
        else:
            model_dict = ab_results['baselines'][model]['allocation']
        row = [model_dict.get(label, 0) for label in all_labels]
        data.append(row)

    data = np.array(data)
    ind = np.arange(len(models))
    bottom = np.zeros(len(models))

    plt.figure(figsize=(12, 8))
    for i, label in enumerate(all_labels):
        plt.bar(ind, data[:, i], bottom=bottom, label=label)
        bottom += data[:, i]

    plt.xticks(ind, models, rotation=45, ha='right', fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('Allocation (%)', fontsize=14)
    plt.title(f'Token Allocation Comparison\nToken: {token_symbol.upper()}', fontsize=16, fontweight='bold')
    plt.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"exported_charts/ab_comparison_stackedbar_{token_symbol}.png", dpi=600)
    plt.close()

def save_ab_testing_grouped_bar(labels, values, ab_results, token_symbol):
    import matplotlib.pyplot as plt
    import numpy as np

    models = ['GPT'] + list(ab_results['baselines'].keys())
    all_labels = sorted(set(labels + [lbl for b in ab_results['baselines'].values() for lbl in b['allocation'].keys()]))

    width = 0.15
    x = np.arange(len(all_labels))

    plt.figure(figsize=(24, 10))

    for i, model in enumerate(models):
        if model == 'GPT':
            model_dict = dict(zip(labels, values))
        else:
            model_dict = ab_results['baselines'][model]['allocation']
        y = [model_dict.get(label, 0) for label in all_labels]
        plt.bar(x + i * width, y, width=width, label=f'{model} (Gini: {ab_results["gpt_gini"]:.3f})' if model == 'GPT' else f'{model} (Gini: {ab_results["baselines"][model]["gini"]:.3f})')

    plt.xticks(x + width * len(models) / 2, all_labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Allocation (%)', fontsize=14)
    plt.title(f'Token Allocation Comparison by Category\nToken: {token_symbol.upper()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"exported_charts/ab_comparison_groupedbar_{token_symbol}.png", dpi=600)
    plt.close()
        
# Main Functions
def main(): 
    # Get input mode selection
    os.makedirs("exported_charts", exist_ok=True)
    input_mode = get_input_mode()
    
    # Get user input based on selected mode
    if input_mode == '1':
        user_input = get_structured_input()
    else:
        user_input = get_generic_input()
    
    # Generate project summaries
    project_summaries = summarize_all_projects(knowledge_base)
    
    # Create appropriate prompt based on input type
    if user_input.get('input_type') == 'structured':
        prompt = create_structured_prompt(user_input, project_summaries)
        input_type = "structured"
    else:
        prompt = create_generic_prompt(user_input, project_summaries)
        input_type = "generic"
    
    # Generate AI recommendation
    print("\nGenerating token design...")
    
    result = ask_openai_enhanced(prompt, input_type)
    
    print("Tokenomics Design")
    print(result)
    
    # Extract token symbol for visualizations
    token_symbol = user_input.get('token_symbol', user_input.get('project_name', 'TOKEN'))
    
    # Extract allocation for visualization
    labels, values = extract_allocation_enhanced(result)
    
    if len(values) > 1:
        # Always show the pie chart
        create_allocation_pie_chart(labels, values, token_symbol)
        
        # Loop for multiple simulation choices until user chooses to skip
        while True:
            choice = get_user_testing_choice()
            
            if choice == '1':  # A/B Testing only
                print("\nPerforming A/B Testing Analysis...")
                ab_results = perform_ab_testing(labels, values)
                create_ab_testing_visualization(labels, values, ab_results, token_symbol)
                save_ab_testing_charts(labels, values, ab_results, token_symbol)
                save_ab_testing_stacked_bar(labels, values, ab_results, token_symbol)
                save_ab_testing_grouped_bar(labels, values, ab_results, token_symbol)
                def interpret_gini_plain(gini: float) -> str:
                    if gini < 0.1:
                        return f"{gini:.3f} (Very Fair)"
                    elif gini < 0.2:
                        return f"{gini:.3f} (Moderate Fairness)"
                    else:
                        return f"{gini:.3f} (Uneven Distribution)"

                print("\nA/B Testing – Token Allocation Fairness Evaluation")
                print("Gini Coefficient measures inequality in distribution. Lower values indicate a fairer, more balanced allocation.\n")

                gpt_gini = ab_results['gpt_gini']
                print(f"Your Token Design (GPT Recommendation): {interpret_gini_plain(gpt_gini)}\n")

                better_models = []
                worse_models = []

                for model_name, data in ab_results['baselines'].items():
                    interpreted = interpret_gini_plain(data['gini'])
                    if data['better_than_gpt']:
                        better_models.append((model_name, interpreted))
                    else:
                        worse_models.append((model_name, interpreted))

                if better_models:
                    print("Models with more balanced token distribution than your design:")
                    for name, gini_text in better_models:
                        print(f" - {name}: {gini_text}")
                else:
                    print("Your token distribution is more balanced than all tested baseline models.")

                if worse_models:
                    print("\nModels with less balanced token distribution than your design:")
                    for name, gini_text in worse_models:
                        print(f"  - {name}: {gini_text}")

            elif choice == '2':  # Monte Carlo only
                print("\nPerforming Monte Carlo Simulation...")
                total_supply = extract_total_supply(result)
                if total_supply:
                    mc_results = monte_carlo_simulation(total_supply)
                    create_monte_carlo_visualization(mc_results, token_symbol)
                    save_final_supply_histogram(mc_results, token_symbol)
                    save_confidence_intervals(mc_results, token_symbol)
                    save_cumulative_burn(mc_results, token_symbol)
                    save_risk_metrics_bar(mc_results, token_symbol)
                    print(f"\nMonte Carlo Analysis Results (12 months):")
                    print(f"Mean Final Supply: {mc_results['mean_final_supply']:,.0f}")
                    print(f"Standard Deviation: {mc_results['std_final_supply']:,.0f}")
                    print(f"95% Confidence Interval: [{mc_results['percentile_5']:,.0f}, {mc_results['percentile_95']:,.0f}]")
                else:
                    print("Could not extract total supply for Monte Carlo simulation.")
            
            elif choice == '3':  # Both analyses
                print("\nPerforming Comprehensive Analysis...")
                ab_results = perform_ab_testing(labels, values)
                create_ab_testing_visualization(labels, values, ab_results, token_symbol)
                total_supply = extract_total_supply(result)
                if total_supply:
                    mc_results = monte_carlo_simulation(total_supply)
                    create_monte_carlo_visualization(mc_results, token_symbol)
                    print(f"\nComprehensive Analysis Results:")
                    print(f"A/B Testing - GPT Gini: {ab_results['gpt_gini']:.3f}")
                    for model_name, data in ab_results['baselines'].items():
                        status = "Better" if data['better_than_gpt'] else "Worse"
                        print(f"   {model_name}: {data['gini']:.3f} ({status})")
                    print(f"\nMonte Carlo Simulation (12 months):")
                    print(f"   Expected Final Supply: {mc_results['mean_final_supply']:,.0f}")
                    print(f"   Risk Range: [{mc_results['percentile_5']:,.0f}, {mc_results['percentile_95']:,.0f}]")
                else:
                    print("Only A/B Testing results are available.")
            
            elif choice == '4':  # Knowledge Base Analysis
                print("\nPerforming Knowledge Base Analysis...")
                kb_analysis = generate_dynamic_baseline_models(knowledge_base)
                kb_analysis_result(kb_analysis)
                print("Generated Baseline Models")
                for model_name, allocation in kb_analysis['baseline_models'].items():
                    print(f"\n{model_name}:")
                    total = sum(allocation.values())
                    for category, percentage in allocation.items():
                        print(f"  • {category}: {percentage:.1f}%")
                    print(f"  Total: {total:.1f}%")
            
            elif choice == '5':  # Exit loop
                print("Analysis skipped. Recommendation complete.")
                break

    else:
        print("Could not extract proper token allocation from the recommendation.")
        print("Please review the AI response manually.")
    
    project_name = user_input.get('project_name', 'your project')
    print(f"\nTokenomics design for {project_name} completed!")

if __name__ == "__main__":
    main()