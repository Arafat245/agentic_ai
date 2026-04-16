"""
Phase 6 — Save all experiment artifacts into an organized directory.
Creates a clean folder structure for presenting to the course teacher.
"""

import shutil
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
TEMP_EXPS = PROJECT_DIR / 'temp_exps'
OUTPUT_DIR = PROJECT_DIR / 'consensus_experiments'


def main():
    timestamp = datetime.now().strftime('%Y%m%d')

    # Create organized output directory
    dirs = {
        'code': OUTPUT_DIR / 'code',
        'results': OUTPUT_DIR / 'results',
        'logs': OUTPUT_DIR / 'logs',
        'data': OUTPUT_DIR / 'data',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copy all phase6 code files
    code_files = sorted(TEMP_EXPS.glob('phase6_*.py'))
    print("=== CODE FILES ===")
    for f in code_files:
        shutil.copy2(f, dirs['code'] / f.name)
        print(f"  {f.name}")

    # 2. Copy all result CSVs
    result_patterns = [
        'results_api_*.csv',
        'results_local_*.csv',
        'results_lr_top30.csv',
        'results_random_forest_top30.csv',
        'results_xgboost_top30.csv',
        'results_consensus_summary.csv',
        'results_discriminative_summary.csv',
        'readme_llm_section.md',
    ]
    print("\n=== RESULT FILES ===")
    for pattern in result_patterns:
        for f in sorted(TEMP_EXPS.glob(pattern)):
            shutil.copy2(f, dirs['results'] / f.name)
            print(f"  {f.name}")

    # 3. Copy log files
    log_patterns = ['phase6_*.log', 'phase4_output.log']
    print("\n=== LOG FILES ===")
    for pattern in log_patterns:
        for f in sorted(TEMP_EXPS.glob(pattern)):
            shutil.copy2(f, dirs['logs'] / f.name)
            print(f"  {f.name}")

    # 4. Copy intermediate data files
    data_files = [
        'modality_text_features.csv',
        'modality_agent_raw_outputs.pkl',
    ]
    print("\n=== DATA FILES ===")
    for fname in data_files:
        f = TEMP_EXPS / fname
        if f.exists():
            shutil.copy2(f, dirs['data'] / f.name)
            print(f"  {f.name}")

    # 5. Copy the paper for reference
    paper_src = PROJECT_DIR / 'papers' / 'multimodal_agent.pdf'
    if paper_src.exists():
        shutil.copy2(paper_src, OUTPUT_DIR / 'ConSensus_paper.pdf')
        print(f"\n  ConSensus_paper.pdf")

    # 6. Copy README
    readme_src = PROJECT_DIR / 'README.md'
    if readme_src.exists():
        shutil.copy2(readme_src, OUTPUT_DIR / 'README.md')
        print(f"  README.md")

    print(f"\n{'='*60}")
    print(f"All artifacts saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Print directory tree
    print("\nDirectory structure:")
    for p in sorted(OUTPUT_DIR.rglob('*')):
        depth = len(p.relative_to(OUTPUT_DIR).parts)
        indent = '  ' * depth
        if p.is_file():
            size = p.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"{indent}{p.name} ({size_str})")
        else:
            print(f"{indent}{p.name}/")


if __name__ == '__main__':
    main()
