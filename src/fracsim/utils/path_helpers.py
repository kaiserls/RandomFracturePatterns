from pathlib import Path
import logging

REL_PATH_TO_APPS = "PANDAS_coop/src/app/apps/"
APP_TEMPLATE = "2d_fracking_saturated/"
REPOSITORY_NAME = "RandomFracturePatterns"


def get_pandas_app_path():
    def get_repository_path(repo_name: str) -> Path:
        cwd_path = Path.cwd()
        for i, dir in enumerate(cwd_path.parts):
            if dir == repo_name:
                break
            if i == len(cwd_path.parts) - 1:
                raise ValueError(
                    f"Could not find repository {repo_name} in path {cwd_path}"
                )
        repo_path = Path(*cwd_path.parts[0 : i + 1])
        return repo_path
    try:
        with open("repo_path", "r") as f:
            repo_path = f.read()
            repo_path = Path(repo_path)
    except FileNotFoundError:
        logging.error("Could not find file repo_path. Please create it. Trying to guess the path.")
        repo_path = get_repository_path(REPOSITORY_NAME)
    app_path = repo_path / REL_PATH_TO_APPS / APP_TEMPLATE
    return app_path
