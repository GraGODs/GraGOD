#  type: ignore
import os

from dotenv import load_dotenv

ROOT_DIR = "GraGOD"

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # noqa
    get_ipython().run_line_magic("autoreload", "2")  # noqa

    get_ipython().run_line_magic(
        "config", 'InlineBackend.figure_format = "jpeg"'
    )  # noqa
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa

    import matplotlib

    matplotlib.pyplot.style.use("default")
    matplotlib.rcParams["figure.dpi"] = 72
except NameError as ex:
    print("Could not load magic extensions:", ex)

base_path = os.getcwd()
idx = base_path.split("/").index(ROOT_DIR)

for _ in base_path.split("/")[idx + 1 :]:
    os.chdir("..")

print("Working on", os.getcwd())


print("Loaded .env file:", load_dotenv(".env"))
