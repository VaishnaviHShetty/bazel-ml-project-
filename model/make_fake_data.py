import numpy as np
import pandas as pd

np.random.seed(42)

prefixes = ["bazelci/", "examples/", "scripts/", "site/", "src/conditions",
            "src/java_tools", "src/main", "src/test", "src/tools",
            "third_party/", "tools/"]
types = ["JAVA", "C/C++", "Starlark", "python", "HTML/CSS/JS"]

# Fake rule: some prefixes and types are "heavier" than others
prefix_weight = {p: np.random.randint(1000, 9000) for p in prefixes}
type_weight = {t: np.random.randint(500, 6000) for t in types}

rows = []
for _ in range(300):
    p = np.random.choice(prefixes)
    t = np.random.choice(types)
    cpu = prefix_weight[p] + type_weight[t] + np.random.normal(0, 1500)
    rows.append((p, t, max(int(cpu), 500)))

df = pd.DataFrame(rows, columns=["prefix", "file_type", "cpu_time_ms"])
df.to_csv("model/data.csv", index=False)
print(df.head())
print("Saved", len(df), "rows to model/data.csv")