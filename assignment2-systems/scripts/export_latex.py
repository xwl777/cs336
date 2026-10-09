import argparse
from pathlib import Path

import pandas as pd


EXAMPLE_ROWS = [
    {
        "model": "small",
        "mode": "forward",
        "batch_size": 4,
        "context_length": 512,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
        "warmup_steps": 5,
        "measurement_steps": 10,
        "mean_ms": 0.0,
        "std_ms": 0.0,
    },
    {
        "model": "small",
        "mode": "forward_backward",
        "batch_size": 4,
        "context_length": 512,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
        "warmup_steps": 5,
        "measurement_steps": 10,
        "mean_ms": 0.0,
        "std_ms": 0.0,
    },
    {
        "model": "small",
        "mode": "train",
        "batch_size": 4,
        "context_length": 512,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
        "warmup_steps": 5,
        "measurement_steps": 10,
        "mean_ms": 0.0,
        "std_ms": 0.0,
    },
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input format: {path.suffix}. Use .csv, .json, or .jsonl.")


def format_numeric_columns(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        if pd.api.types.is_float_dtype(df[column]):
            df[column] = df[column].map(lambda x: f"{x:.{digits}f}")
    return df


def export_latex(
    df: pd.DataFrame,
    output_path: Path,
    caption: str | None,
    label: str | None,
    digits: int,
    index: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted_df = format_numeric_columns(df, digits)
    latex = formatted_df.to_latex(
        index=index,
        caption=caption,
        label=label,
        escape=True,
    )
    output_path.write_text(latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark results to a LaTeX table with pandas.")
    parser.add_argument("--input", type=Path, help="Input data file: .csv, .json, or .jsonl.")
    parser.add_argument("--output", type=Path, default=Path("benchmark_table.tex"), help="Output .tex path.")
    parser.add_argument("--caption", default="Transformer benchmark results.")
    parser.add_argument("--label", default="tab:benchmark-results")
    parser.add_argument("--digits", type=int, default=3, help="Decimal places for floating-point columns.")
    parser.add_argument("--index", action="store_true", help="Include the DataFrame index in the LaTeX table.")
    parser.add_argument("--example", action="store_true", help="Export an example table instead of reading input.")
    args = parser.parse_args()

    if args.example:
        df = pd.DataFrame(EXAMPLE_ROWS)
    else:
        if args.input is None:
            raise ValueError("Pass --input results.csv/results.json, or use --example.")
        df = read_table(args.input)

    export_latex(
        df=df,
        output_path=args.output,
        caption=args.caption,
        label=args.label,
        digits=args.digits,
        index=args.index,
    )
    print(f"Wrote LaTeX table to {args.output}")


if __name__ == "__main__":
    main()
