"""
Script to download Mozilla Common Voice Spontaneous Speech datasets for all 21 languages.

Downloads data from Mozilla Data Collective and organizes it into the expected format:
    data/mozilla_speech_data/{lang_code}/
        train/
            audios/
            metadata.csv
        validation/
            audios/
            metadata.csv

Usage:
    python -m src.data.download --languages all
    python -m src.data.download --languages aln sco
    python -m src.data.download --languages aln --output-dir /custom/path
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# All 21 languages in the Mozilla Spontaneous Speech dataset
LANGUAGES = {
    # Africa
    "bxk": "Bukusu",
    "cgg": "Chiga",
    "kcn": "Nubi",
    "koo": "Konzo",
    "led": "Lendu",
    "lke": "Kenyi",
    "lth": "Thur",
    "ruc": "Ruuli",
    "rwm": "Amba",
    "ttj": "Rutoro",
    "ukv": "Kuku",
    # Americas
    "hch": "Wixárika",
    "meh": "Southwestern Tlaxiaco Mixtec",
    "mmc": "Michoacán Mazahua",
    "top": "Papantla Totonac",
    "tob": "Toba Qom",
    # Europe
    "aln": "Gheg Albanian",
    "el-CY": "Cypriot Greek",
    "sco": "Scots",
    # Asia
    "bew": "Betawi",
    "pne": "Western Penan",
}

# Mozilla Data Collective dataset identifiers
# Format: mozilla-foundation/mdc-spontaneous-speech-{lang_code}
DATASET_BASE = "mozilla-foundation/mdc-spontaneous-speech"


def get_dataset_name(lang_code: str) -> str:
    """Get the Hugging Face dataset identifier for a language."""
    return f"{DATASET_BASE}-{lang_code}"


def download_and_prepare_language(
    lang_code: str,
    output_dir: Path,
    force_redownload: bool = False,
) -> bool:
    """
    Download and prepare data for a single language.
    
    Args:
        lang_code: ISO 639 language code
        output_dir: Base output directory (data/mozilla_speech_data/)
        force_redownload: If True, redownload even if data exists
        
    Returns:
        True if successful, False otherwise
    """
    lang_dir = output_dir / lang_code
    
    # Check if already downloaded
    if lang_dir.exists() and not force_redownload:
        train_meta = lang_dir / "train" / "metadata.csv"
        val_meta = lang_dir / "validation" / "metadata.csv"
        if train_meta.exists() and val_meta.exists():
            print(f"  [{lang_code}] Already exists, skipping (use --force to redownload)")
            return True
    
    dataset_name = get_dataset_name(lang_code)
    print(f"  [{lang_code}] Downloading from {dataset_name}...")
    
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_name, trust_remote_code=True)
    except Exception as e:
        print(f"  [{lang_code}] ERROR: Failed to download - {e}")
        return False
    
    # Process train and validation splits
    for split in ["train", "validation"]:
        if split not in dataset:
            # Some datasets might use 'dev' or 'test' instead
            if split == "validation" and "dev" in dataset:
                split_data = dataset["dev"]
            elif split == "validation" and "test" in dataset:
                split_data = dataset["test"]
            else:
                print(f"  [{lang_code}] WARNING: No {split} split found")
                continue
        else:
            split_data = dataset[split]
        
        split_dir = lang_dir / split
        audio_dir = split_dir / "audios"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        metadata_rows = []
        
        for idx, example in enumerate(tqdm(split_data, desc=f"    {split}", leave=False)):
            # Extract audio and save
            audio = example.get("audio", {})
            audio_array = audio.get("array")
            sampling_rate = audio.get("sampling_rate", 16000)
            
            # Get transcription
            sentence = example.get("sentence", example.get("text", example.get("transcription", "")))
            
            if audio_array is None or not sentence:
                continue
            
            # Save audio file
            audio_filename = f"{idx:06d}.wav"
            audio_path = audio_dir / audio_filename
            
            # Use soundfile to save audio
            import soundfile as sf
            sf.write(audio_path, audio_array, sampling_rate)
            
            # Add to metadata
            metadata_rows.append({
                "file_name": f"audios/{audio_filename}",
                "sentence": sentence,
            })
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_path = split_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        print(f"    {split}: {len(metadata_rows)} samples saved")
    
    return True


def download_all_languages(
    languages: list[str],
    output_dir: Path,
    force_redownload: bool = False,
) -> dict[str, bool]:
    """
    Download data for multiple languages.
    
    Args:
        languages: List of language codes to download
        output_dir: Base output directory
        force_redownload: If True, redownload even if data exists
        
    Returns:
        Dictionary mapping language codes to success status
    """
    results = {}
    
    print(f"Downloading {len(languages)} languages to {output_dir}")
    print("=" * 60)
    
    for lang_code in languages:
        if lang_code not in LANGUAGES:
            print(f"  [{lang_code}] WARNING: Unknown language code, skipping")
            results[lang_code] = False
            continue
        
        print(f"\n[{lang_code}] {LANGUAGES[lang_code]}")
        results[lang_code] = download_and_prepare_language(
            lang_code, output_dir, force_redownload
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary:")
    successful = sum(results.values())
    print(f"  Successful: {successful}/{len(languages)}")
    
    failed = [lang for lang, success in results.items() if not success]
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download Mozilla Common Voice Spontaneous Speech datasets"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["all"],
        help="Language codes to download (use 'all' for all 21 languages)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mozilla_speech_data"),
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if data exists",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List all available languages and exit",
    )
    
    args = parser.parse_args()
    
    if args.list_languages:
        print("Available languages:")
        for code, name in sorted(LANGUAGES.items()):
            print(f"  {code}: {name}")
        return
    
    # Determine which languages to download
    if "all" in args.languages:
        languages = list(LANGUAGES.keys())
    else:
        languages = args.languages
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    download_all_languages(languages, args.output_dir, args.force)


if __name__ == "__main__":
    main()
