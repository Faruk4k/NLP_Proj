"""
Quick start training script - Run all subtasks with minimal configuration
"""

import os
import subprocess
import sys
from pathlib import Path


def run_subtask(subtask_dir: str, config_path: str = "config/config.yaml"):
    """Run training for a subtask."""
    
    print(f"\n{'='*60}")
    print(f"Training {subtask_dir}")
    print(f"{'='*60}\n")
    
    original_dir = os.getcwd()
    
    try:
        os.chdir(subtask_dir)
        
        cmd = [
            sys.executable,
            "src/train.py",
            "--config", config_path,
            "--data-root", "data",
            "--seed", "42",
            "--log-dir", "logs"
        ]
        
        result = subprocess.run(cmd, check=True)
        
        print(f"\n{subtask_dir} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n{subtask_dir} failed with error code {e.returncode}")
        return False
    finally:
        os.chdir(original_dir)


def main():
    """Run all subtasks."""
    
    print("\n" + "="*60)
    print("POLAR: Multilingual Polarization Detection - Quick Start")
    print("="*60)
    
    # Check if we're in the project root
    if not Path("subtask1").exists():
        print("Error: Must run from project root directory")
        print("Usage: python quickstart.py")
        return
    
    # Run subtasks
    results = {}
    
    subtasks = ["subtask1", "subtask2", "subtask3"]
    
    for subtask in subtasks:
        success = run_subtask(subtask)
        results[subtask] = success
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for subtask, success in results.items():
        status = "COMPLETED" if success else "FAILED"
        print(f"{subtask}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\nAll subtasks completed successfully!")
        print("\nNext steps:")
        print("1. Review logs in each subtask's logs/ directory")
        print("2. Check trained models in each subtask's models/ directory")
        print("3. Run inference: python shared_utils/inference.py --help")
        print("4. Submit predictions to: https://www.codabench.org/competitions/10669/")
    else:
        print("\nSome subtasks failed. Check logs for details.")


if __name__ == "__main__":
    main()
