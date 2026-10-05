import pandas as pd

FILE = "realdataset.csv"

ALLOWED_PREFIXES = {
    "bazelci/",
    "examples/",
    "scripts/",
    "site/",
    "src/conditions",
    "src/java_tools",
    "src/main",
    "src/test",
    "src/tools",
    "third_party/",
    "tools/",
}

ALLOWED_FILE_TYPES = {
    "JAVA",
    "C/C++",
    "Starlark",
    "python",
    "HTML/CSS/JS",
}

REQUIRED_COLUMNS = ["prefix", "file_type", "cpu_time_ms"]


def main():
    df = pd.read_csv(FILE)

    print(f"Rows found: {len(df)}")

    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"Invalid columns. Expected: {REQUIRED_COLUMNS}"
        )

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    invalid_prefixes = set(df["prefix"]) - ALLOWED_PREFIXES
    if invalid_prefixes:
        raise ValueError(
            f"Invalid prefixes: {sorted(invalid_prefixes)}"
        )

    invalid_types = set(df["file_type"]) - ALLOWED_FILE_TYPES
    if invalid_types:
        raise ValueError(
            f"Invalid file types: {sorted(invalid_types)}"
        )

    df["cpu_time_ms"] = pd.to_numeric(
        df["cpu_time_ms"], errors="coerce"
    )

    if df["cpu_time_ms"].isnull().any():
        raise ValueError("cpu_time_ms contains non-numeric values.")

    if (df["cpu_time_ms"] <= 0).any():
        raise ValueError("cpu_time_ms must be positive.")

    print("Dataset validation passed.")
    print("\nRows by prefix:")
    print(df["prefix"].value_counts())

    print("\nRows by file type:")
    print(df["file_type"].value_counts())


if __name__ == "__main__":
    main()