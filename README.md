# Tokenomics Design Tool

AI-assisted tokenomics design and analysis tool that generates comprehensive token economic models with advanced analytics.

## Features

- **AI-Powered Design**: Generate tokenomics models using OpenAI GPT-4
- **Two Input Modes**: Structured input for detailed control or generic description
- **Advanced Analytics**: A/B testing, Monte Carlo simulation, and knowledge base analysis
- **Interactive Visualizations**: Pie charts, distribution analysis, and comparison charts
- **Knowledge Base**: Built-in reference from successful Web3 projects

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install openai python-dotenv matplotlib numpy
   ```

2. **Setup Environment**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Add Knowledge Base**
   Ensure `TokenomicsKnowledge.json` is in the project directory

4. **Run the Tool**
   ```bash
   python main.py
   ```

## Usage

1. Choose input mode:
   - **Structured Input**: Detailed project specifications
   - **Generic Input**: Simple project description

2. Provide project details based on selected mode

3. Select analysis options:
   - A/B Testing: Compare with proven models
   - Monte Carlo Simulation: Predict token supply scenarios
   - Knowledge Base Analysis: Detailed reference breakdown
   - Comprehensive Analysis: All analytics combined

4. Review generated tokenomics design and analysis results

## Output

- Comprehensive tokenomics recommendation
- Token allocation visualizations
- Risk analysis and comparison charts
- Exported charts in `exported_charts/` directory

## Requirements

- Python 3.7+
- OpenAI API key
- Required packages: `openai`, `python-dotenv`, `matplotlib`, `numpy`

## File Structure

```
├── main.py   
├── TokenomicsKnowledge.json
├── .env              
├── exported_charts/  
└── README.md
```