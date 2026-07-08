import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from Utils.Agents import (
    Cardiologist,
    Psychologist,
    Pulmonologist,
    MultidisciplinaryTeam,
)

# =====================================================
# Logging Configuration
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv("apikey.env")

# =====================================================
# Configuration
# =====================================================

REPORT_PATH = r"Medical Reports/Medical Report - Michael Johnson - Panic Attack Disorder.txt"

OUTPUT_DIR = "results"

TXT_OUTPUT = os.path.join(OUTPUT_DIR, "final_diagnosis.txt")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "final_diagnosis.json")


# =====================================================
# Utility Functions
# =====================================================

def load_report(path: str) -> str:
    """Load medical report from file."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Medical Report Not Found:\n{path}")

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def run_agent(name, agent):
    """Run a specialist AI agent safely."""

    try:
        logging.info(f"{name} Started")

        response = agent.run()

        logging.info(f"{name} Finished")

        return name, response

    except Exception as e:

        logging.error(f"{name} Failed : {e}")

        return name, f"ERROR : {e}"


def save_results(responses, final_diagnosis):
    """Save TXT and JSON results."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TXT_OUTPUT, "w", encoding="utf-8") as file:

        file.write("=" * 60 + "\n")
        file.write("MEDICAL AI FINAL DIAGNOSIS\n")
        file.write("=" * 60 + "\n\n")

        file.write(final_diagnosis)

    with open(JSON_OUTPUT, "w", encoding="utf-8") as file:

        json.dump(
            {
                "Cardiologist": responses["Cardiologist"],
                "Psychologist": responses["Psychologist"],
                "Pulmonologist": responses["Pulmonologist"],
                "Final Diagnosis": final_diagnosis,
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    logging.info("Results Saved Successfully")


# =====================================================
# Main Program
# =====================================================

def main():

    start = time.perf_counter()

    print("=" * 70)
    print("        AI MULTI-AGENT MEDICAL DIAGNOSIS SYSTEM")
    print("=" * 70)

    logging.info("Loading Medical Report...")

    medical_report = load_report(REPORT_PATH)

    agents = {
        "Cardiologist": Cardiologist(medical_report),
        "Psychologist": Psychologist(medical_report),
        "Pulmonologist": Pulmonologist(medical_report),
    }

    responses = {}

    logging.info("Running Specialist Agents...\n")

    with ThreadPoolExecutor(max_workers=3) as executor:

        futures = {
            executor.submit(run_agent, name, agent): name
            for name, agent in agents.items()
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Running Specialists"
        ):
            name, result = future.result()
            responses[name] = result

    logging.info("Generating Final Diagnosis...\n")

    team = MultidisciplinaryTeam(
        cardiologist_report=responses["Cardiologist"],
        psychologist_report=responses["Psychologist"],
        pulmonologist_report=responses["Pulmonologist"],
    )

    final_diagnosis = team.run()

    save_results(responses, final_diagnosis)

    end = time.perf_counter()

    print("\n" + "=" * 70)
    print("FINAL DIAGNOSIS COMPLETED")
    print("=" * 70)

    print(f"\nTXT Report  : {TXT_OUTPUT}")
    print(f"JSON Report : {JSON_OUTPUT}")
    print(f"\nExecution Time : {end-start:.2f} seconds")

    print("=" * 70)


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    main()