from pathlib import Path

REL_PATH_TO_APPS = "PANDAS_coop/src/app/apps/"
APP_TEMPLATE = "2d_fracking_saturated/"
REPOSITORY_NAME = "RandomFracturesRework"


def get_pandas_app_path():
    def get_repository_path(repo_name: str) -> str:
        cwd_path = Path.cwd()
        for i, dir in enumerate(cwd_path.parts):
            if dir == repo_name:
                break
        repo_path = Path(*cwd_path.parts[0 : i + 1])
        return repo_path

    repo_path = get_repository_path(REPOSITORY_NAME)
    app_path = repo_path / REL_PATH_TO_APPS / APP_TEMPLATE
    return app_path
