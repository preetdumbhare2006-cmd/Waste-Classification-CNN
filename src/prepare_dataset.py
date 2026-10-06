from pathlib import Path
import shutil, random
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"dataset"; OUT=ROOT/"data_split"
MAP={"paper":"degradable","cardboard":"degradable","plastic":"non_degradable","glass":"non_degradable","metal":"non_degradable"}
EXT={".jpg",".jpeg",".png",".bmp",".webp"}
if OUT.exists(): shutil.rmtree(OUT)
for split in ["train","val","test"]:
    for label in ["degradable","non_degradable"]:
        (OUT/split/label).mkdir(parents=True)
random.seed(42)
for src_class,target in MAP.items():
    files=[p for p in (SOURCE/src_class).iterdir() if p.is_file() and p.suffix.lower() in EXT]
    random.shuffle(files); n=len(files); a=int(.70*n); b=a+int(.15*n)
    for split,items in [("train",files[:a]),("val",files[a:b]),("test",files[b:])]:
        for i,p in enumerate(items):
            shutil.copy2(p,OUT/split/target/f"{src_class}_{i}_{p.name}")
    print(f"{src_class:10} -> {target:15} train={a}, val={b-a}, test={n-b}")
print("\nCreated:",OUT)
