# Generate requirements.txt file for specific libraries
def generate_requirements():
    required_libraries = ["streamlit", "scikit-learn"]
    with open("requirements.txt", "w") as f:
        for lib in required_libraries:
            result = subprocess.run(["pip", "show", lib], capture_output=True, text=True, check=True)
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    version = line.split()[1]
                    f.write(f"{lib}=={version}\n")
    print("requirements.txt generated successfully.")

generate_requirements()
