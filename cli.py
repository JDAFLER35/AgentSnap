import argparse
from src.core.agent import Agent

def main():
    parser = argparse.ArgumentParser(description="AI Sidekick CLI")
    parser.add_argument("--name", type=str, help="Name of the agent", default="Sidekick")
    args = parser.parse_args()

    agent = Agent(args.name)
    print(agent)

if __name__ == "__main__":
    main()