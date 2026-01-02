import argparse
from stt_utils import run_from_cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio and analyze sentiment.")
    parser.add_argument(
        "audio_paths",
        nargs="+",  # multiple files
        help="Paths to audio files"
    )
    args = parser.parse_args()
    
    run_from_cli(args.audio_paths)
