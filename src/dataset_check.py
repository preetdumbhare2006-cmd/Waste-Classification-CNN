from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/"dataset"
CLASSES=["cardboard","glass","metal","paper","plastic"]
EXT={".jpg",".jpeg",".png",".bmp",".webp"}
total=0
print("\nWaste Dataset Statistics\n"+"-"*40)
for c in CLASSES:
    p=DATASET/c
    if not p.exists():
        print(f"{c:12}: MISSING")
        continue
    n=sum(x.is_file() and x.suffix.lower() in EXT for x in p.iterdir())
    total+=n
    print(f"{c:12}: {n}")
print("-"*40)
print("Total images:",total)
print("Excluded class: trash")
