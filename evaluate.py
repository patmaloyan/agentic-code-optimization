
import sys
import importlib.util
import time

def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("module.name", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)

    # Assuming the main function returns the total time
    total_time_ms = module.main() * 1000
    return {"total_time_ms": total_time_ms}

if __name__ == "__main__":
    # Example usage:
    # python evaluate.py <program_to_evaluate.py>
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
        results = evaluate(program_path)
        print(f"Evaluation Results: {results}")
    else:
        print("Usage: python evaluate.py <program_to_evaluate.py>")
