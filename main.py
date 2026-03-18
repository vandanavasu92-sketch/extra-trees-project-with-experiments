# ============================================================
# main.py
# Project entry point
# ============================================================

from runs.run_decicion_tree import run_decision_tree


def main():
    dt_results = run_decision_tree()
    print("\nDecision Tree pipeline completed successfully!")


if __name__ == "__main__":
    main()