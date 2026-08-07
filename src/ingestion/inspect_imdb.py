from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

basics_path = PROJECT_ROOT / "data" / "raw" / "imdb" / "title.basics.tsv"


with basics_path.open("r", encoding="utf-8") as file:
    header = file.readline().strip()

    print("Columnas:")
    print(header)

    print("\nPrimeras 5 filas:")
    for _ in range(5):
        print(file.readline().strip())