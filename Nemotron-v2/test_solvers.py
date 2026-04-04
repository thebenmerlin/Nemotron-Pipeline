"""
Test solvers against train.csv to measure accuracy per family.
Run: python test_solvers.py [path_to_train.csv]
"""

import csv, sys, os
from collections import defaultdict

# Add parent dir if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solvers import solve_puzzle, detect_family

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), '..', 'Nemotron-Pipeline', 'train.csv'
    )
    print(f"Loading {csv_path} ...")

    total = defaultdict(int)
    correct = defaultdict(int)
    solved = defaultdict(int)
    errors = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            prompt = row['prompt']
            gt = str(row['answer']).strip()
            family, answer, cot, was_solved = solve_puzzle(prompt, gt)

            total[family] += 1
            if answer == gt:
                correct[family] += 1
            else:
                if len(errors[family]) < 3:
                    errors[family].append((i, gt, answer))
            if was_solved:
                solved[family] += 1

            if (i + 1) % 500 == 0:
                print(f"  processed {i+1} ...")

    print(f"\n{'='*60}")
    print(f"{'Family':<10} {'Total':>6} {'Solved':>8} {'Correct':>8} {'Acc':>8}")
    print(f"{'-'*60}")
    grand_total = 0
    grand_correct = 0
    grand_solved = 0
    for fam in sorted(total.keys()):
        t, s, c = total[fam], solved[fam], correct[fam]
        grand_total += t
        grand_correct += c
        grand_solved += s
        acc = c / t * 100 if t else 0
        solve_rate = s / t * 100 if t else 0
        print(f"{fam:<10} {t:>6} {s:>7} ({solve_rate:>5.1f}%) {c:>7} ({acc:>5.1f}%)")

    print(f"{'-'*60}")
    overall_acc = grand_correct / grand_total * 100 if grand_total else 0
    overall_solve = grand_solved / grand_total * 100 if grand_total else 0
    print(f"{'TOTAL':<10} {grand_total:>6} {grand_solved:>7} ({overall_solve:>5.1f}%) {grand_correct:>7} ({overall_acc:>5.1f}%)")

    if errors:
        print(f"\nSample errors:")
        for fam, errs in sorted(errors.items()):
            for idx, gt, pred in errs[:2]:
                print(f"  [{fam}] row {idx}: expected={gt!r}, got={pred!r}")

if __name__ == "__main__":
    main()
