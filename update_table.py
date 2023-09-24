import os
from collections import defaultdict

import psycopg2


def remove_markdown_table(markdown_text):
    text = []
    for line in markdown_text.split("\n"):
        if line.find("|") != -1:
            continue
        text.append(line)
    return "\n".join(text)


def download_data() -> dict[str, list[str | float]]:
    result_table = defaultdict(list)
    runs_metrics = []
    with psycopg2.connect(
        database=os.environ.get("POSTGRES_DB"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
        host=os.environ.get("POSTGRES_HOST"),
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                    SELECT run_uuid, name
                    FROM (
                        SELECT t.*, ROW_NUMBER() OVER(PARTITION BY name ORDER BY end_time) AS rn
                        FROM runs t
                    ) ranked
                    WHERE rn = 1
            """)
            runs = cur.fetchall()
            for run_uuid, name in runs:
                if name.find("iris") != -1:
                    continue
                result_table["name"].append(name)
                cur.execute(
                    """
                    select key, value from metrics where run_uuid = (%s)
                    """,
                    (run_uuid,),
                )
                runs_metrics.append({name: val for name, val in cur.fetchall()})

    metric_names = []
    for run in runs_metrics:
        metric_names.extend(run.keys())
    metric_names = set(metric_names)
    for run_metrics in runs_metrics:
        for metric_name in metric_names:
            result_table[metric_name].append(run_metrics.get(metric_name, None))
    return result_table


def generate_table(data: dict[str, list[str | float]]) -> str:
    keys = data.keys()
    result = ""
    for key in keys:
        result += f"|{key}"
    result += "|\n"
    for _ in range(len(keys)):
        result += "|-"
    result += "|\n"
    n = len(data["name"])
    for i in range(n):
        for key in keys:
            result += f"|{data[key][i]}"
        result += "|\n"
    return result


def main():
    markdown_file_path = "README.md"

    with open(markdown_file_path, "r") as file:
        markdown_content = file.read()

    cleaned_content = remove_markdown_table(markdown_content)
    data = download_data()
    # table = generate_table(data)
    cleaned_content += generate_table(data)
    with open(markdown_file_path, "w") as file:
        file.write(cleaned_content)


if __name__ == "__main__":
    main()
