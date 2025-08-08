"""Command Line Interface for Easy PMF

This module provides command-line tools for PMF analysis.
"""

import argparse
import sys


def main():
    """Main entry point for the easy-pmf command-line tool."""
    parser = argparse.ArgumentParser(
        description="Easy PMF - Command Line Interface for Positive Matrix Factorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  easy-pmf --interactive          # Interactive dataset selection
  easy-pmf --analyze-all          # Analyze all datasets in data/ folder
  easy-pmf --help                 # Show this help message

For more information, visit: https://github.com/easy-pmf/easy-pmf
        """,
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive analysis (default)",
    )

    parser.add_argument(
        "--analyze-all",
        "-a",
        action="store_true",
        help="Analyze all datasets automatically",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing data files (default: data)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files (default: output)",
    )

    parser.add_argument(
        "--factors",
        "-f",
        type=int,
        default=7,
        help="Number of PMF factors (default: 7)",
    )

    parser.add_argument("--version", "-v", action="version", version="Easy PMF 0.1.0")

    args = parser.parse_args()

    # If no specific mode is chosen, default to interactive
    if not args.analyze_all:
        args.interactive = True

    try:
        if args.interactive:
            run_interactive_analysis(args)
        elif args.analyze_all:
            run_batch_analysis(args)
    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_interactive_analysis(args):
    """Run interactive analysis mode."""
    print("🌟 Welcome to Easy PMF Interactive Analysis!")
    print("=" * 50)

    # Try to import and run the quick analysis
    try:
        from . import quick_analysis

        # Set up arguments for quick_analysis
        sys.argv = ["quick_analysis"]  # Reset sys.argv to avoid conflicts
        quick_analysis.main()
    except ImportError:
        print("Error: Could not import analysis modules.")
        print("Please ensure the package is properly installed.")
        sys.exit(1)


def run_batch_analysis(args):
    """Run batch analysis on all datasets."""
    print("🚀 Running batch analysis on all datasets...")
    print("=" * 50)

    try:
        from . import analyze_all_datasets

        # Set up arguments for analyze_all_datasets
        sys.argv = ["analyze_all_datasets"]  # Reset sys.argv to avoid conflicts
        analyze_all_datasets.main()
    except ImportError:
        print("Error: Could not import analysis modules.")
        print("Please ensure the package is properly installed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
