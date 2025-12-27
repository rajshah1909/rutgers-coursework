# run_all.py
import subprocess, sys

def run(pyfile):
    print("\n=== Running", pyfile)
    try:
        subprocess.run([sys.executable, pyfile], check=True)
    except subprocess.CalledProcessError as e:
        print(f"--- {pyfile} exited with non-zero status ({e.returncode}). Continuing.")

if __name__ == "__main__":
    for f in ["bayes_rule.py","nb_train.py","nb_cls.py","whitening.py","galton_sim.py","map_ml_demo.py"]:
        run(f)
    print("\nAll tasks attempted. Check outputs and logs above.")
