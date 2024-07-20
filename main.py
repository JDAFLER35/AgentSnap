import yaml
import asyncio
import logging
from argparse import ArgumentParser
from typing import Dict, Any

from interfaces.api import start_api
from interfaces.cli import start_cli
from core.agent import Agent
from utils.data_processing import DataProcessor
from modules.nlp import NLPModule
from modules.ethics import EthicsModule

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        raise

def setup_logging(config: Dict[str, Any]):
    log_config = config.get('logging', {})
    logging.basicConfig(
        level=log_config.get('level', 'INFO'),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_config.get('file', 'ai_agent.log')
    )

async def run_api(config: Dict[str, Any], agent: Agent):
    await start_api(config, agent)

async def run_cli(config: Dict[str, Any], agent: Agent):
    await start_cli(config, agent)

async def main():
    parser = ArgumentParser(description="AI Agent Application")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--mode", choices=["api", "cli", "both"], default="both", help="Run mode")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    logging.info("Initializing AI Agent components...")
    
    data_processor = DataProcessor()
    nlp_module = NLPModule(config.get('nlp', {}))
    ethics_module = EthicsModule()
    
    agent = Agent("AI Sidekick", config, data_processor, nlp_module, ethics_module)
    
    tasks = []
    if args.mode in ["api", "both"]:
        tasks.append(run_api(config, agent))
    if args.mode in ["cli", "both"]:
        tasks.append(run_cli(config, agent))

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logging.info("Shutting down AI Agent...")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Perform any necessary cleanup
        logging.info("AI Agent shut down complete.")

if __name__ == "__main__":
    asyncio.run(main())