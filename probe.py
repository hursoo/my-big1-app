import os, sys, platform, json, subprocess, io, contextlib
print("=== Python/OS ===")
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Machine:", platform.machine())
print("Processor:", platform.processor())

print("\n=== NumPy/BLAS ===")
try:
    import numpy as np
    print("NumPy:", np.__version__)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    print(buf.getvalue())
except Exception as e:
    print("numpy import/show_config error:", e)

print("\n=== pip list (핵심만) ===")
important = {"tomotopy","numpy","pandas","scikit-learn","matplotlib","janome","mecab-python3","kiwipiepy","konlpy","scipy"}
try:
    pip_list = subprocess.check_output([sys.executable,"-m","pip","list","--format=json"], text=True)
    for p in json.loads(pip_list):
        if p["name"].lower() in important:
            print(f'{p["name"]}=={p["version"]}')
except Exception as e:
    print("pip list error:", e)

print("\n=== tomotopy ===")
try:
    import tomotopy as tp
    print("tomotopy:", tp.__version__)
    if hasattr(tp, "isa"):
        print("tp.isa:", tp.isa)
except Exception as e:
    print("tomotopy import error:", e)

print("\n=== Env (determinism) ===")
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","PYTHONHASHSEED","OPENBLAS_CORETYPE"]:
    print(k, "=", os.environ.get(k))

print("\n=== Locale ===")
for k in ["LANG","LC_ALL","LC_CTYPE"]:
    print(k, "=", os.environ.get(k))

print("\n=== CPU flags (Linux) ===")
try:
    cpu = subprocess.check_output("lscpu | egrep 'Architecture|Model name|Flags'", shell=True, text=True)
    print(cpu)
except Exception as e:
    print("lscpu error:", e)

print("\n=== Freeze to file ===")
try:
    with open("requirements_runtime.txt","w") as f:
        subprocess.check_call([sys.executable,"-m","pip","freeze"], stdout=f)
    print("Saved: requirements_runtime.txt")
except Exception as e:
    print("freeze error:", e)
