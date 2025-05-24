#!/usr/bin/env python3
"""
Test runner script for LLM OCR package.
Provides convenient commands for running different types of tests.
"""
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for LLM OCR package")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "performance", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run with coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--markers", "-m",
        help="Run tests with specific markers"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path based on type
    if args.type == "unit":
        cmd.append("tests/unit")
    elif args.type == "integration":
        cmd.append("tests/integration")
    elif args.type == "performance":
        cmd.append("tests/performance")
    else:
        cmd.append("tests")
    
    # Add options
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=llm_ocr", "--cov-report=html", "--cov-report=term-missing"])
    
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    # Run tests
    success = run_command(cmd, f"Running {args.type} tests")
    
    if args.coverage and success:
        print("\nCoverage report generated in htmlcov/index.html")
    
    # Additional quality checks
    if args.type == "all":
        print(f"\n{'='*60}")
        print("Running additional quality checks...")
        print('='*60)
        
        # Type checking
        run_command(["python", "-m", "mypy", "llm_ocr"], "Type checking with mypy")
        
        # Code formatting check
        run_command(["python", "-m", "black", "--check", "llm_ocr"], "Code formatting check")
        
        # Import sorting check
        run_command(["python", "-m", "isort", "--check-only", "llm_ocr"], "Import sorting check")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())