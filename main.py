from runs.run_all_models import run_all_models
from runs.generate_tables import generate_tables_from_csv
from runs.generate_plots import generate_all_plots

def main():
    st_results = run_all_models()
    generate_tables_from_csv("results/all_models_results.csv")
    generate_all_plots(
        results_csv="results/all_models_results.csv",
        output_dir="results/plots"
    ) 
    print("\nAll Model pipeline completed successfully!")


if __name__ == "__main__":
    main() 