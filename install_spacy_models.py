import subprocess

scispacy_url = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz"

def install_spacy_models():
    # Install spaCy models using the spaCy CLI commands
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    subprocess.run(["pip", "install", scispacy_url])

if __name__ == "__main__":
    install_spacy_models()