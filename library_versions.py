# library_version.py
import sys
import importlib_metadata

def get_library_version(library_name):
    try:
        version = importlib_metadata.version(library_name)
        return version
    except importlib_metadata.PackageNotFoundError:
        return "Not installed"

print("Python version:", sys.version)

libraries = [
    "transformers",
    "torch",
    "scikit-learn",
    "datasets",
    "accelerate",
    "optuna",
    "sentence-transformers",
    "mlflow",
    "spacy",
    "gensim",
    "datasets",
    "pymongo",
    "scipy",
]

print("\nLibrary versions:")
for library in libraries:
    version = get_library_version(library)
    print(f"{library}: {version}")