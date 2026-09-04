import os
import sys
import time
import json
import datetime
import unittest

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    start_time = time.time()
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(PROJECT_ROOT, 'tests'))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = round(time.time() - start_time, 3)

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_tests": result.testsRun,
        "passed_tests": result.testsRun - len(result.failures) - len(result.errors),
        "failed_tests": len(result.failures),
        "errored_tests": len(result.errors),
        "status": "OK" if result.wasSuccessful() else "FAILED",
        "elapsed_seconds": elapsed,
        "failures": [str(f[0]) for f in result.failures],
        "errors": [str(e[0]) for e in result.errors]
    }

    out_dir = os.path.join(PROJECT_ROOT, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, 'test_summary.json')

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTest Summary saved to {summary_path}")
    print(f"Status: {summary['status']} | Passed: {summary['passed_tests']}/{summary['total_tests']} in {elapsed}s")

    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()
