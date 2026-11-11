"""Download the Yambda-50M files from the Hugging Face Hub."""

from huggingface_hub import hf_hub_download

from nexttrack import constants
from nexttrack.config import parse_args


def download(raw_dir, with_embeddings=True):
    """Fetch the flat 50m files, the mappings and optionally the 13.8 GB embeddings."""
    names = list(constants.FLAT_FILES)
    names.append(constants.ALBUM_FILE)
    names.append(constants.ARTIST_FILE)
    if with_embeddings:
        names.append(constants.EMBEDDINGS_FILE)
    paths = []
    for name in names:
        path = hf_hub_download(
            repo_id=constants.HF_REPO_ID,
            repo_type="dataset",
            filename=name,
            local_dir=raw_dir,
        )
        print(path, flush=True)
        paths.append(path)
    return paths


def add_arguments(parser):
    parser.add_argument("--skip-embeddings", action="store_true")


def main():
    config, args = parse_args("Download Yambda-50M", add_arguments)
    download(config["paths"]["raw"], with_embeddings=not args.skip_embeddings)


if __name__ == "__main__":
    main()
