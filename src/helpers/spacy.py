import os
import spacy; spacy.prefer_gpu()

def install_spacy_model(lang: str="en", pipeline_for: str="sm"):
    """
    Install the spacy model for the specified language and pipeline.

    Args:
        lang (str): Language code (default: "en").
        pipeline_for (str): Pipeline type ("sm" for efficiency, "trf" for accuracy).

    Returns:
        str: The name of the installed spacy model.
    """
    pipeline_name = f"{lang}_core_web_{pipeline_for}"

    try:
        _ = spacy.load(pipeline_name)
    except Exception as e:
        print(f"Error loading spacy model: {e}")
        print("Downloading spacy model...")
        os.system(f"python -m spacy download {pipeline_name}")
        try:
            _ = spacy.load(pipeline_name)
        except Exception as e:
            print(f"Error loading spacy model persists.")
            raise e