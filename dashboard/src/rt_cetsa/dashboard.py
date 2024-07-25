# pylint: disable=W0621, W1203, W1309, C0116, E0611, E1129
import logging
import os
import shutil
from dataclasses import asdict, dataclass, replace
from functools import partial
from io import BytesIO
from itertools import product
from math import ceil
from pathlib import Path
from typing import IO, Any, Callable, Optional
from zipfile import ZipFile

import filepattern
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import reacton.ipyvuetify as rv
import solara
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from matplotlib.patches import Circle
from polus.images.features.rt_cetsa_intensity_extraction import (
    alphanumeric_row,
    extract_signal,
)
from polus.images.segmentation.rt_cetsa_plate_extraction import extract_plates
from polus.images.segmentation.rt_cetsa_plate_extraction.core import (
    PLATE_DIMS,
    PlateParams,
)
from polus.tabular.regression.rt_cetsa_analysis.preprocess_data import (
    preprocess_data as analysis_preprocess_data,
)
from polus.tabular.regression.rt_cetsa_analysis.run_rscript import run_rscript
from polus.tabular.regression.rt_cetsa_moltenprot import run_moltenprot_fit
from polus.tabular.transforms.rt_cetsa_metadata import (
    preprocess_from_range,
    preprocess_metadata,
)
from pydantic_core import from_json
from solara.components.file_drop import FileInfo
from solara.lab import Ref, task
from tifffile import TiffFile, imread, imwrite

# get env
POLUS_LOG = os.environ.get("POLUS_LOG", logging.INFO)
DATA_DIR = Path(os.environ.get("DATA_DIR", "data")).resolve()


# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("rt_cetsa_analysis")
logger.setLevel(POLUS_LOG)

RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
PREPROCESSED_IMG_DIR = PREPROCESSED_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_IMG_DIR = PROCESSED_DIR / "images"
PROCESSED_MASK_DIR = PROCESSED_DIR / "masks"
PROCESSED_PARAMS_DIR = PROCESSED_DIR / "params"
INTENSITY_EXTRACTION_OUTDIR = DATA_DIR / "intensity_extraction"
MOLTENPROT_OUTDIR = DATA_DIR / "moltenprot"
ANALYSIS_OUTDIR = DATA_DIR / "analysis"
METADATA_DIR = DATA_DIR / "metadata"

PATTERN = "{index:d+}_{temp:f+}.tif"
PROCESSED_IMG_PATTERN = "{index:d+}_{temp:f+}.ome.tiff"

CAMERA_DATA_FILE = PREPROCESSED_DIR / "camera_data.xlsx"
PLATE_PARAMS_FILE = PROCESSED_PARAMS_DIR / "plate.json" # plate_parameters
INTENSITIES_FILE = INTENSITY_EXTRACTION_OUTDIR / "plate.csv" # intensities for a plate
MOLTENPROT_PARAMS_FILE =    MOLTENPROT_OUTDIR / "params.csv"
MOLTENPROT_VALUES_FILE = MOLTENPROT_OUTDIR / "values.csv"
ANALYSIS_SIGNIF_FILE = ANALYSIS_OUTDIR / "signif_df.csv"
METADATA_FILE = METADATA_DIR / "platemap.xlsx"

DATA_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)
PREPROCESSED_DIR.mkdir(exist_ok=True)
PREPROCESSED_IMG_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
PROCESSED_IMG_DIR.mkdir(exist_ok=True)
PROCESSED_MASK_DIR.mkdir(exist_ok=True)
PROCESSED_PARAMS_DIR.mkdir(exist_ok=True)
INTENSITY_EXTRACTION_OUTDIR.mkdir(exist_ok=True)
MOLTENPROT_OUTDIR.mkdir(exist_ok=True)
ANALYSIS_OUTDIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)

@dataclass(frozen=True)
class State:
    """
    Global Application State

    plate_index: index of the image currently selected
    plate_display: selected image + wells detection overlay
    plot_display: for a selected well, plot of intensities vs temperature
    img_files: list of paths to images.
    
    """
    tab_index: int
    raw_images: list[Path]
    camera_data: Path
    preprocessed_img_files: list[Path]
    img_files: list[Path]
    mask_file: Path
    intensities_df: pd.DataFrame | None
    moltenprot_fit_params: dict[str, int]
    range_temp: tuple[float,float]
    use_range_temp: bool
    params_df: pd.DataFrame | None
    values_df: pd.DataFrame | None
    signif_df: pd.DataFrame | None
    platemap: Path | None
    plate_index: int | None
    plate_display: bytes | None
    plot_display: bytes | None 
    params: PlateParams | None
    selected_well: str | None
    status_step1: str
    status_step2: str
    status_step2_1: str
    status_step3: str
    status_step4: str
    status_step5: str 
    upload_progress: float | bool = False
    platemap_upload_progress: float | bool = False
    camera_upload_progress: float | bool = False


@task
async def run_analysis(state: solara.Reactive[State]):
    """Run the RT_Cetsa statistical analysis."""
    state.value = replace(state.value, status_step5=f"Running analysis...")

    if not MOLTENPROT_PARAMS_FILE.exists() or not MOLTENPROT_VALUES_FILE.exists() :
        raise FileNotFoundError("Error: please rerun moltenprot...")
    if not METADATA_FILE.exists():
        raise FileNotFoundError("Error: please reupload platemap...")

    ANALYSIS_INPUT_PATH = analysis_preprocess_data(
        METADATA_FILE.resolve(),
        MOLTENPROT_VALUES_FILE.resolve(),
        MOLTENPROT_PARAMS_FILE.resolve(),
        ANALYSIS_OUTDIR.resolve()
        )

    run_rscript(
        ANALYSIS_INPUT_PATH.resolve(),
        ANALYSIS_OUTDIR.resolve()
    )
    
    if ANALYSIS_SIGNIF_FILE.exists():
        signif_df = pd.read_csv(ANALYSIS_SIGNIF_FILE)
        state.value = replace(
            state.value,
            signif_df=signif_df,
            status_step5=f"Analysis completed."
        )

def update_moltenprot_params(state: solara.Reactive[State] , param: str, value: int):
    params = {**state.value.moltenprot_fit_params, param: value}
    state.value = replace(state.value, moltenprot_fit_params=params)
    logger.info(f" {param} moltenprot params updated to: {value}")

def update_range_min_temp(state: solara.Reactive[State] , value: int):
    updated_range_temp = (value, state.value.range_temp[1])
    state.value = replace(state.value, range_temp = updated_range_temp)
    logger.info(f"updated range min temp to {value}")

def update_range_max_temp(state: solara.Reactive[State] , value: int):
    updated_range_temp = (state.value.range_temp[0], value)
    state.value = replace(state.value, range_temp = updated_range_temp)
    logger.info(f"updated range max temp to {value}")

def update_use_range_temp(state: solara.Reactive[State] , value: bool): 
    state.value = replace(state.value, use_range_temp = value, status_step2_1="Using temp range.")
    logger.info(f"use range temp : {value}")

def update_use_range_temp_false(state: solara.Reactive[State], value: bool): # pylint: disable=unused-argument
    state.value = replace(state.value, use_range_temp = False, status_step2_1="Please drag and drop the camera data file.")
    logger.info(f"use range temp : False")

@task
async def run_moltenprot(state: solara.Reactive[State]):
    """Run the Moltenprot tool."""
    state.value = replace(state.value, status_step4=f"Running moltenprot...")

    if not INTENSITIES_FILE.exists() :
        raise FileNotFoundError("Error: please recompute intensities...")

    fit_params, fit_curves = run_moltenprot_fit(INTENSITIES_FILE, state.value.moltenprot_fit_params)
    fit_params_path = MOLTENPROT_PARAMS_FILE
    fit_curves_path = MOLTENPROT_VALUES_FILE

    # TODO needed to avoid strange behavior from pandas where df loading 
    # from disk is structured differently that the one returned by the tool.    
    fit_params.to_csv(fit_params_path, index=True)
    fit_curves.to_csv(fit_curves_path, index=True)
    params_df = pd.read_csv(fit_params_path)
    values_df = pd.read_csv(fit_curves_path)

    logger.info("Moltenprot fit completed.")

    state.value = replace(
        state.value,
        params_df = params_df,
        values_df = values_df,
        status_step4 = "Moltenprot fit completed."
    )

@task
async def extract_intensities(state: solara.Reactive[State]):
    state.value = replace(state.value, status_step3=f"Computing well intensities...")

    if not PLATE_PARAMS_FILE.exists() or len(state.value.img_files) == 0:
        raise ValueError("Error: please crop and rotate plate images...") 
    
    fps = filepattern.FilePattern(PROCESSED_IMG_DIR, PROCESSED_IMG_PATTERN)
    df = extract_signal(fps, PLATE_PARAMS_FILE)

    # TODO needed to avoid strange behavior from pandas where df loading 
    # from disk is structured differently that the one returned by the tool.
    df.to_csv(INTENSITIES_FILE)
    intensities_df = pd.read_csv(INTENSITIES_FILE)

    logger.info("Well intensities extracted.")

    state.value = replace(
        state.value,
        intensities_df=intensities_df,
        status_step3="Well intensities extracted."
    )

@task
async def preprocess_images(state: solara.Reactive[State]):
    logger.info("preprocessing images...")
    state.value = replace(state.value, status_step2_1=f"Preprocessing images...")
    if not RAW_DIR.exists():
        raise FileNotFoundError("Error: please reupload plate images...") 
    
    if state.value.use_range_temp:
        preprocess_from_range(RAW_DIR, PREPROCESSED_DIR, state.value.range_temp)
    else:
        preprocess_metadata(RAW_DIR, PREPROCESSED_DIR, metadata_file=CAMERA_DATA_FILE)

    state.value = replace(
        state.value,
        preprocessed_img_files = list(PREPROCESSED_IMG_DIR.iterdir()),
        status_step2_1=f"Images preprocessed."
    )

@task
async def extract_plate_params(state: solara.Reactive[State]):
    state.value = replace(state.value, status_step2=f"Extracting plates...")
    
    if not PREPROCESSED_IMG_DIR.exists():
        raise FileNotFoundError("Error: please preprocess images...") 
    
    extract_plates(PREPROCESSED_IMG_DIR, PATTERN, PROCESSED_DIR)

    fp = filepattern.FilePattern(PROCESSED_IMG_DIR, PROCESSED_IMG_PATTERN)
    sorted_fp = sorted(fp, key=lambda f: f[0]["index"])
    img_files: list[Path] = [f[1][0] for f in sorted_fp]

    mask_file = next(PROCESSED_MASK_DIR.iterdir())

    with PLATE_PARAMS_FILE.open("r") as f:
        params = PlateParams(**from_json(f.read()))

    state.value = replace(
        state.value,
        img_files=img_files,
        mask_file=mask_file,
        params=params,
        plate_index = 1,
        status_step2=f"Plate images cropped and rotated."
    )

def model(
    x: float,
    kn: float,
    bn: float,
    ku: float,
    bu: float,
    dhm: float,
    tm: float,
) -> float:
    R = 8.31446261815324
    recip_t_diff = (1 / tm) - (1 / x)
    exp_term = np.exp((dhm / R) * recip_t_diff)
    numerator = kn * x + bn + (ku * x + bu) * exp_term
    denominator = 1 + exp_term
    return numerator / denominator

def  select_moltenprot_well(state: solara.Reactive[State], col: str, row: str):
    """Action of well selection in the moltenprot df."""
    well, plot_display = render_moltenprot_plot(state, seq_position=row)
    state.value = replace(state.value,
                          status_step4=f"Well selected: ({well})",
                          plot_display=plot_display,
                          selected_well=well
                          )

def  select_grid_well(state: solara.Reactive[State], row: str, col: str):
    """Action of well selection in the plate grid."""
    width = len(state.value.params.X)
    well_index = row * width + col
    dims = PLATE_DIMS[state.value.params.size]
    coords =  alphanumeric_row(row=row, col=col, dims=dims)
    well, plot_display = render_moltenprot_plot(state, seq_position=well_index)
    state.value = replace(state.value,
                          status_step4=f"Well selected: ({well})",
                          plot_display=plot_display,
                          selected_well=well
                          )


def render_moltenprot_plot(state: solara.Reactive[State], seq_position: str):
    df = state.value.params_df
    moltprot_row = df.iloc[seq_position]
    param_names = [
        "kN_init", "bN_init", "kU_init", "bU_init", "dHm_init", "Tm_init",
        "kN_fit", "bN_fit", "kU_fit", "bU_fit", "dHm_fit", "Tm_fit",
        "S", "BS_factor", "T_onset", "dCp_component", "dG_std",
    ]
    params = {p: moltprot_row[p] for p in param_names}
    well = moltprot_row["ID"]
    intensities = state.value.intensities_df
    well_intensities = intensities[well]
    temperatures = intensities["Temperature"] + 273.15

    curve = [
        model(t, params["kN_fit"], params["bN_fit"], params["kU_fit"], params["bU_fit"], params["dHm_fit"], params["Tm_fit"])
        for t in temperatures
    ]

    fig, ax = plt.subplots()
    ax.scatter(temperatures, well_intensities)
    ax.plot(temperatures, curve, color="red")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Intensity")
    ax.set_title(well)
    bio = BytesIO()
    plt.savefig(bio, format="png")
    bio.seek(0)
    return well, bio.read()

def build_plate_image(
    index: int,
    state: solara.Reactive[State]
) -> bytes:
    """Display a plate image."""
    bio = BytesIO()
    image_name = state.value.img_files[index].name

    with TiffFile(state.value.img_files[index]) as tiff:
        plt.figure()
        image = tiff.pages[0].asarray()
        plt.imshow(
            image,
            cmap="gray",
            vmin=image.min(),
            vmax=image.max(),
        )
        plt.title(image_name)

        params = state.value.params
        ax = plt.gca()
        for x, y in product(params.X, params.Y):
            circle = Circle(
                (x, y),
                radius=params.radius,
                color="red",
                fill=False,
            )
            ax.add_patch(circle)
        plt.savefig(bio, format="png")
        bio.seek(0)
        plt.close()
    return bio.read()

def show_plate(state: solara.Reactive[State], index: int):
    logger.info(f"show plate {index}")
    state.value = replace(
        state.value,
        plate_index=index-1,
        plate_display=build_plate_image(index-1, state),
    )   

def save_image(filename: str, content : IO[bytes], state: solara.Reactive[State], progress):
    state.value = replace(state.value, status_step1=f"Reading: {filename} [{progress}%]")
    image = imread(content)
    state.value = replace(state.value, status_step1=f"Saving: {filename} [{progress}%]")
    imwrite(RAW_DIR / filename, image)
    state.value = replace(state.value, status_step1=f"Saved: {filename} [{progress}%]")

show_files_step1: solara.Reactive[bool] = solara.reactive(True)
upload_complete_step1: solara.Reactive[bool] = solara.reactive(False)

def upload(files: list[FileInfo], state: solara.Reactive[State], upload_progress: solara.lab.Ref, on_complete: Optional[Callable] = None):
    upload_progress.value = False
    state.value = replace(state.value, status_step1=f"Upload Completed")

    files_count = len(files)
    current_file_index = 0
    save_progress = int(current_file_index * 100 / files_count)

    for file in files:
        if file['name'].endswith(".zip"):
            with ZipFile(file["file_obj"]) as zip_file:
                files_count = files_count - 1 + len(zip_file.namelist())
                for filename in zip_file.namelist():
                    save_progress = int(current_file_index * 100 / files_count)
                    upload_progress.value = save_progress
                    if filename.startswith('__MACOSX/'):
                        current_file_index += 1
                        continue             
                    if filename.endswith(".tif") or filename.endswith(".tiff"):
                        with zip_file.open(filename, "r") as fr:
                            save_image(filename, fr, state, save_progress)
                            current_file_index += 1
        
        elif file['name'].endswith(".tif") or file['name'].endswith(".tiff"):
            save_progress = int(current_file_index * 100 / files_count)
            upload_progress.value = save_progress
            filename = file['name']
            content = BytesIO(file['data'])
            save_image(filename, content, state, save_progress)
            current_file_index += 1

        logger.info(f"Progress {save_progress}%")

    raw_images = list(RAW_DIR.iterdir())
    state.value = replace(state.value,raw_images= raw_images ,status_step1=f"Done uploading images...")
    upload_progress.value = False
    logger.info(f"Done uploading images...")
    if on_complete:
        on_complete()

def upload_plate_params(file: FileInfo, state: solara.Reactive[State], upload_progress: solara.lab.Ref):
    if not file['name'].endswith(".xlsx"):
        state.value = replace(state.value, status_step5="Uploading error. Not a xlsx file.")
        return
    with METADATA_FILE.open("wb") as f:
        f.write(file["data"])
        logger.info("Platemap uploaded successfully.")
        state.value = replace(state.value, platemap= METADATA_FILE.resolve() , status_step5="Platemap uploaded successfully.")

def upload_camera_data(file: FileInfo, state: solara.Reactive[State], upload_progress: solara.lab.Ref):
    if not file['name'].endswith(".xlsx"):
        state.value = replace(state.value, status_step2_1="Uploading error. Not a xlsx file.")
        return
    with CAMERA_DATA_FILE.open("wb") as f:
        f.write(file["data"])
        logger.info("Camera data uploaded successfully.")
        state.value = replace(state.value, camera_data= CAMERA_DATA_FILE.resolve() , status_step2_1="Camera data uploaded successfully.")

def set_tab_index(index: int, state: solara.Reactive[State]):
    state.value = replace(state.value, tab_index=index)

def get_id_index(state: solara.Reactive[State], id_: str):
    return state.value.params_df["ID"].to_list().index(id_)


@solara.component
def DeleteStepData(state: solara.Reactive[State], step_index: int):
    open_delete_confirmation = solara.reactive(False)
    with solara.Row():
        solara.Button(
            icon_name="mdi-delete",
            icon=True,
            on_click=lambda: open_delete_confirmation.set(True),
        )
        solara.lab.ConfirmationDialog(
            open_delete_confirmation,
            ok="Ok, Delete",
            on_ok=partial(delete_all_data, state, step_index),
            content=f"Are you sure you want to delete all data for step {step_index}  and all subsequent steps?",
        )

def delete_all_data(state: solara.Reactive[State], step_index: int):
    logger.info(f"delete all data for step {step_index}  and all subsequent steps...")
    if step_index == 1:
            logger.info(f"delete 1...")
            shutil.rmtree(RAW_DIR)
            RAW_DIR.mkdir()
            state.value = replace(state.value, status_step1=f"All files have been deleted.")
    if step_index <= 2:
            logger.info(f"delete 2...")
            shutil.rmtree(PREPROCESSED_IMG_DIR)
            shutil.rmtree(PREPROCESSED_DIR)
            shutil.rmtree(PROCESSED_IMG_DIR)
            shutil.rmtree(PROCESSED_MASK_DIR)
            shutil.rmtree(PROCESSED_PARAMS_DIR)
            PREPROCESSED_DIR.mkdir()
            PREPROCESSED_IMG_DIR.mkdir()
            PROCESSED_IMG_DIR.mkdir()
            PROCESSED_MASK_DIR.mkdir()
            PROCESSED_PARAMS_DIR.mkdir()
    if step_index <= 3:
            logger.info(f"delete 3...")
            INTENSITIES_FILE.unlink(missing_ok=True)
    if step_index <= 4:
            logger.info(f"delete 4...")
            MOLTENPROT_PARAMS_FILE.unlink(missing_ok=True)
            MOLTENPROT_VALUES_FILE.unlink(missing_ok=True)
    if step_index <= 5:
            logger.info(f"delete 5...")
            METADATA_FILE.unlink(missing_ok=True)
            shutil.rmtree(ANALYSIS_OUTDIR)
            ANALYSIS_OUTDIR.mkdir()
    state.value = replace(state.value, **asdict(init_state()))

def init_state():
    """Set up state when dashboad is loaded."""
    tab_index = 0
    #step1
    raw_images = list(RAW_DIR.iterdir())

    #step2
    # sort = lambda f : f.with_suffix("").stem.split("_")
    fp = filepattern.FilePattern(PREPROCESSED_IMG_DIR, PROCESSED_IMG_PATTERN)
    sorted_fp = sorted(fp, key=lambda f: f[0]["index"])
    preprocessed_img_files: list[Path] = [f[1][0] for f in sorted_fp]

    fp = filepattern.FilePattern(PROCESSED_IMG_DIR, PROCESSED_IMG_PATTERN)
    sorted_fp = sorted(fp, key=lambda f: f[0]["index"])
    img_files: list[Path] = [f[1][0] for f in sorted_fp]
    mask_file = None
    masks = list(PROCESSED_MASK_DIR.iterdir())
    if len(masks) == 1:
        mask_file = masks[0]
    params = None
    if PLATE_PARAMS_FILE.exists():
        with PLATE_PARAMS_FILE.open("r") as f:
            params = PlateParams(**from_json(f.read()))
    selected_well = None
    camera_data = None
    if CAMERA_DATA_FILE.exists():
        camera_data = CAMERA_DATA_FILE.resolve()
    
    range_temp = (37,90)
    use_range_temp = False

    #step3
    intensities_df = None
    if INTENSITIES_FILE.exists():
        intensities_df = pd.read_csv(INTENSITIES_FILE)

    #step4
    moltenprot_fit_params = {
        "savgol": 10,
        "trim_max": 0,
        "trim_min": 0,
        "baseline_fit": 3,
        "baseline_bounds": 3,
    }
    params_df = None
    if MOLTENPROT_PARAMS_FILE.exists():
        params_df = pd.read_csv(MOLTENPROT_PARAMS_FILE)
    values_df = None
    if MOLTENPROT_VALUES_FILE.exists():
        values_df = pd.read_csv(MOLTENPROT_VALUES_FILE)

    #step5
    signif_df = None
    if ANALYSIS_SIGNIF_FILE.exists():
        signif_df = pd.read_csv(ANALYSIS_SIGNIF_FILE)
    platemap = None
    if METADATA_FILE.exists():
        platemap = METADATA_FILE.resolve()

    status_step1: str = "" if raw_images else "Please upload some data..."
    status_step2_1: str = "" if camera_data else "Please drag and drop the camera data file."
    status_step2: str = ""
    status_step3: str = "" 
    status_step4: str = ""
    status_step5: str = "" 

    return State(
        tab_index = tab_index,
        raw_images = raw_images,
        preprocessed_img_files = preprocessed_img_files,
        img_files = img_files,
        mask_file = mask_file,
        plate_index=None,
        plate_display=None,
        plot_display=None,
        params= params,
        selected_well= selected_well,
        intensities_df = intensities_df,
        moltenprot_fit_params = moltenprot_fit_params,
        range_temp = range_temp,
        use_range_temp = use_range_temp,
        params_df = params_df,
        values_df= values_df,
        signif_df = signif_df,
        platemap = platemap,
        camera_data = camera_data,
        status_step1 = status_step1,
        status_step2 = status_step2,
        status_step2_1 = status_step2_1,
        status_step3 = status_step3,
        status_step4 = status_step4,
        status_step5 = status_step5,
    )


state = solara.reactive(init_state())

@solara.component
def Page():

    # NOTE Use that mechanism or update global state directly
    tab_index = Ref(state.fields.tab_index)
    upload_progress = Ref(state.fields.upload_progress)
    platemap_upload_progress = Ref(state.fields.platemap_upload_progress)
    camera_upload_progress = Ref(state.fields.camera_upload_progress)
    plate_display = Ref(state.fields.plate_display)
    selected_well = Ref(state.fields.selected_well)

    show_step1 : bool = True
    show_step2 : bool = len(state.value.raw_images) > 0
    show_step3 : bool = state.value.params is not None and len(state.value.img_files) > 0
    show_step4 : bool = state.value.intensities_df is not None
    show_step5 : bool = state.value.params_df is not None and state.value.values_df is not None
    show_run_analysis: bool = show_step4 and state.value.platemap is not None

    with solara.Column(align="center"):
        with solara.lab.Tabs(value=tab_index):
            """Step 1
            
            Upload images.
            output : img_files
            """
            with solara.lab.Tab("1/ Upload", disabled=not show_step1, tab_children=[DeleteStepData(state=state, step_index=1)
] if tab_index.value == 0 else []):
                with solara.Card(title="Step 1: Upload data"):
                    solara.Markdown(r'''
                                    
                        * Drag and drop one or multiple tif images or zip files.
                        * The list of images already uploaded is shown at the bottom.
                        * You can delete all uploaded images by clicking on the trash icon.

                        ''')
                    def _set_progress(*args):
                        state.value = replace(state.value,status_step1=f"Analyzing files...")
                        upload_progress.set(True)

                    def on_upload_complete():
                        upload_complete_step1.set(True)
                        show_files_step1.set(False)

                    if show_files_step1.value:
                        solara.FileDropMultiple(
                            on_file=partial(upload, state=state, upload_progress=upload_progress, on_complete=on_upload_complete),
                            lazy=False,
                            on_total_progress=_set_progress,
                        )

                    if state.value.status_step1 == "Done uploading images...":
                        solara.Success(state.value.status_step1)
                    else:
                        solara.Markdown(state.value.status_step1)
                    solara.ProgressLinear(upload_progress.value)
                    if state.value.status_step1 == "Done uploading images...":
                        show_files_step1.set(True)
                    
                    with solara.CardActions():
                        with solara.Row():
                            solara.Markdown(f"{len(state.value.raw_images)} images uploaded.")
                
                if len(state.value.raw_images) > 0:
                    with solara.Card():
                        raw_images = state.value.raw_images
                        if len(raw_images) > 0:
                                num_col = 8
                                batch_len = ceil(len(raw_images) / num_col)
                                with solara.GridFixed(columns=num_col):
                                    for col in range(0,num_col):
                                        with solara.Column():
                                            for file in raw_images[col * batch_len : min((col+1)*batch_len, len(raw_images))]:
                                                solara.HTML(unsafe_innerHTML=f"{file.name}")
                    solara.Button(label="Next Step", on_click=partial(set_tab_index, 1, state),
                                    style={"margin": "8px", "background-color": "#6aa7f7"})
                                

            """Step 2"""
            with solara.lab.Tab("2/ Preprocess Plates", disabled=not show_step2, tab_children=[DeleteStepData(state=state, step_index=2)
] if tab_index.value == 1 else []):
                solara.Markdown(r'''
                    * In this step, we preprocess plate images:        
                        * Crop and rotate plate image.
                        * Detect plate wells.
                    * NOTE : The tool currently expects all image names to be a numerically ordered sequence (ex: 1.tif, 2.tif, 11.tif etc...)
                    * Image and detected wells are displayed for sanity check.
                    ''')
                with solara.Card(title="Step 2.1: Preprocess data"):
                    with solara.Columns():
                        with solara.Column(style={"color": "#e3e3e3" if state.value.use_range_temp else "black",
                                                  "background-color": "white" if state.value.use_range_temp else "#fafafa"}):
                            solara.Checkbox(label="Upload Camera Data File", value=not state.value.use_range_temp , on_value=partial(update_use_range_temp_false,state))
                            solara.FileDrop(
                                label="Drag and drop camera data file.",
                                on_file=partial(upload_camera_data, state=state, upload_progress=camera_upload_progress),
                                lazy=False,
                                on_total_progress=camera_upload_progress.set
                            )
                        with solara.Column(style={"background-color": "#fafafa" if state.value.use_range_temp else "white"}):
                            solara.Checkbox(label="Use Temp Range", value=state.value.use_range_temp , on_value=partial(update_use_range_temp,state))
                            with solara.Row(gap="2px", margin="2px", justify="center",
                                            style={"background-color": "#fafafa" if state.value.use_range_temp else "white"}):
                                solara.InputFloat(
                                    label="min_temp",
                                    disabled=not state.value.use_range_temp,
                                    value=state.value.range_temp[0],
                                    on_value=partial(update_range_min_temp, state))
                                solara.InputFloat(
                                    label="max_temp",
                                    disabled=not state.value.use_range_temp,
                                    value=state.value.range_temp[1],
                                    on_value=partial(update_range_max_temp, state))
                            solara.Row(justify="center",
                                        style={"background-color": "#fafafa" if state.value.use_range_temp else "white",
                                               "height": "53px"})
                    if not state.value.use_range_temp:
                        if not preprocess_images.error:
                            if state.value.status_step2_1 == "Images preprocessed.":
                                solara.Success(state.value.status_step2_1)
                            else:
                                solara.Markdown(state.value.status_step2_1)
                    else:
                        if not preprocess_images.error and state.value.status_step2_1 != "Please drag and drop the camera data file.":
                            if state.value.status_step2_1 == "Images preprocessed.":
                                solara.Success(state.value.status_step2_1)
                            else:
                                solara.Markdown(state.value.status_step2_1)
                    solara.ProgressLinear(camera_upload_progress.value)
                    solara.ProgressLinear(preprocess_images.pending)
                    
                    with solara.CardActions():
                        with solara.Row(gap="2px", margin="2px", justify="center"):
                            solara.Button(
                                label="Preprocess images",
                                on_click=partial(preprocess_images,state),
                            )

                    if preprocess_images.error:
                        state.value = replace(state.value, status_step2_1=str(preprocess_images.exception))


                with solara.Card(title="Step 2: Preview data"):
                    
                    if extract_plate_params.error:
                        solara.Error(state.value.status_step2)
                    elif state.value.status_step2 == "Plate images cropped and rotated.":
                        solara.Success(state.value.status_step2)
                    else:
                        solara.Markdown(state.value.status_step2)
                    solara.ProgressLinear(extract_plate_params.pending)

                    with solara.CardActions():
                        with solara.Row(gap="2px", margin="2px", justify="center"):
                            solara.Button(
                                label="Extract Plate Params",
                                on_click=partial(extract_plate_params,state),
                                disabled= not state.value.status_step2_1 == "Images preprocessed."
                            )

                    if extract_plate_params.error:
                        state.value = replace(state.value, status_step2=str(extract_plate_params.exception))
                    
                if state.value.status_step2 == "Plate images cropped and rotated.":
                    with solara.Card():
                        if len(state.value.img_files) > 0 :
                            solara.SliderInt(
                                "Select image",
                                min=1,
                                max=len(state.value.img_files),
                                on_value=partial(show_plate, state)
                            )

                        if plate_display.value is not None:
                            solara.Image(plate_display.value)
                    solara.Button(label="Next Step", on_click=partial(set_tab_index, 2, state),
                                    style={"margin": "8px", "background-color": "#6aa7f7"})

            """Step 3"""
            with solara.lab.Tab("3/ Extract well intensities", disabled=not show_step3, tab_children=[DeleteStepData(state=state, step_index=3)
] if tab_index.value == 2 else []):
                with solara.Card(title="Step 3: Extract well intensities"):
                    solara.Markdown(r'''
                        * Extract plate intensities.
                        * Once completed, a dataframe collecting the well intensities will be displayed.
                        ''')
                    
                    if not extract_intensities.error and state.value.status_step3 == "Well intensities extracted.":
                        solara.Success(state.value.status_step3)
                    elif extract_intensities.error:
                        state.value = replace(state.value, status_step3=str(extract_intensities.exception))
                        solara.Error(state.value.status_step3)
                    else:
                        solara.Markdown(state.value.status_step3)
                    
                    solara.ProgressLinear(extract_intensities.pending)

                    with solara.CardActions():
                        with solara.Row():
                            solara.Button(
                                label="Extract Intensities",
                                on_click=partial(extract_intensities,state),
                            )                        

                if state.value.intensities_df is not None:
                    with solara.Card():
                        def return_df_headers_step3():
                            headers_ = [{"text": col, "value": col, "align": "left"} for col in state.value.intensities_df.columns.to_list()]
                            return headers_
                        rv.DataTable(items=state.value.intensities_df.to_dict("records"),
                                    headers=return_df_headers_step3(),
                                    items_per_page=-1,
                                    hide_default_footer=True,
                                    height="calc(100vh - 300px)",
                                    fixed_header=True,
                                    dense=True,
                        )
                    solara.Button(label="Next Step", on_click=partial(set_tab_index, 3, state),
                                    style={"margin": "8px", "background-color": "#6aa7f7"})

            """Step 4"""
            with solara.lab.Tab("4/ Run Moltenprot", disabled=not show_step4, tab_children=[DeleteStepData(state=state, step_index=4)
] if tab_index.value == 3 else []):
                with solara.Card(title="Step 4: Run Moltenprot"):
                    solara.Markdown(r'''
                        Run moltenprot and extract model parameters.
                        ''')
                    with solara.Row(gap="2px", margin="2px", justify="center"):
                        solara.InputInt(
                            label="trim_min",
                            value=state.value.moltenprot_fit_params['trim_min'],
                            on_value=partial(update_moltenprot_params, state, "trim_min")
                        )
                        solara.InputInt(
                            label="trim_max",
                            value=state.value.moltenprot_fit_params['trim_max'],
                            on_value=partial(update_moltenprot_params, state, "trim_max")
                        )
                        solara.InputInt(
                            label="savgol",
                            value=state.value.moltenprot_fit_params['savgol'],
                            on_value=partial(update_moltenprot_params, state, "savgol")
                        )
                        solara.InputInt(
                            label="baseline_fit",
                            value=state.value.moltenprot_fit_params['baseline_fit'],
                            on_value=partial(update_moltenprot_params, state, "baseline_fit")
                        )
                        solara.InputInt(
                            label="baseline_bounds",
                            value=state.value.moltenprot_fit_params['baseline_bounds'],
                            on_value=partial(update_moltenprot_params, state, "baseline_bounds")
                        )
                    
                    if not run_moltenprot.error and state.value.status_step4 == "Moltenprot fit completed.":
                        solara.Success(state.value.status_step4)
                    elif run_moltenprot.error:
                        state.value = replace(state.value, status_step4=str(run_moltenprot.exception))
                        solara.Error(state.value.status_step4)
                    else:
                        solara.Markdown(state.value.status_step4)
                    if run_moltenprot.error:
                        state.value = replace(state.value, status_step4=str(run_moltenprot.exception))
                    solara.ProgressLinear(run_moltenprot.pending)

                    with solara.CardActions():
                        with solara.Row(gap="2px", margin="2px", justify="center"):
                            solara.Button("Run Moltenprot", on_click=partial(run_moltenprot,state))

                if state.value.params_df is not None: 
                    def get_bs_factor(coords: str):
                        return state.value.params_df[state.value.params_df["ID"] == coords]["BS_factor"].to_numpy()[0]
                    
                    def color_mapping(x: float) -> float:
                        if x <= 0.8:
                            return 0.0
                        if x <= 1.0:
                            return (x - 0.8) / 0.2
                        else:
                            return 1.0

                    def get_color(colors: LinearSegmentedColormap, coords: str):
                        bs_factor = get_bs_factor(coords)
                        return rgb2hex(colors(color_mapping(bs_factor)))

                    with solara.Card("Well Plate"):
                        with solara.Row():
                            if state.value.plot_display is not None:
                                solara.Image(state.value.plot_display)  

                            dims = PLATE_DIMS[state.value.params.size]
                            with solara.Div(style={"padding": "1.6%"}):
                                colors = LinearSegmentedColormap.from_list('colors', [
                                    (0.0, '#fa0000'),
                                    (0.5, '#008000'),
                                    (1.0,'#00fa00')
                                    ], N=512)

                                with solara.GridFixed(columns=len(state.value.params.X), align_items="self-start", justify_items="center"):
                                    for row in range(len(state.value.params.Y)):
                                        for col in range(len(state.value.params.X)):
                                            coords =  alphanumeric_row(row=row, col=col, dims=dims)
                                            solara.Button(
                                                coords,
                                                outlined=False,
                                                text=True,
                                                on_click=partial(select_grid_well, state, row, col),
                                                style={"font-size": "8px", "width" :"fit-content", "height":"fit-content", "padding": "0px",
                                                    "background-color": get_color(colors, coords), "margin": "0px", "display": "inline-block",
                                                    "border-radius": "0", "border": "none", "justify-content": "center"}
                                            )

                        def return_df_headers_step4_1():
                            headers_ = [{"text": col, "value": col, "align": "left"} for col in state.value.params_df.columns.to_list()]
                            headers_[0] = {**headers_[0], "width": "70px"} # ID
                            dhm_init_index = state.value.params_df.columns.to_list().index("dHm_init") 
                            headers_[dhm_init_index] = {**headers_[dhm_init_index], "width": "100px"} # BS_factor
                            return headers_

                        if state.value.plot_display is not None:
                            items_ = [x for x in state.value.params_df.to_dict("records") if x["ID"] == selected_well.value]
                            rv.DataTable(
                                items=items_,
                                headers=return_df_headers_step4_1(),
                                hide_default_footer=True,
                                disable_sort=True,
                            )
                    # NOTE: this is informative but Solara is too slow at rendering this
                    # and it is not a good user experience
                    # color_display_html = colors._repr_html_().replace("under", "0.8").replace("over", "1.0").replace("colors", "").replace('<div style="margin: 0 auto; display: inline-block;">bad <div title="#00000000" style="display: inline-block; width: 1em; height: 1em; margin: 0; vertical-align: middle; border: 1px solid #555; background-color: #00000000;"></div></div>', "")
                    # heatmap_colors = [solara.Markdown("## BS Factor Color Mapping"), solara.HTML(unsafe_innerHTML=color_display_html)]
                    # solara.Details(summary="See Color Mapping", children=heatmap_colors, expand=False)

                    with solara.Card("Fit Params"):
                        def select_row(x: Any, y: Any, z: dict): # pylint: disable=unused-argument
                            id_ = z["ID"]
                            row_num = get_id_index(state, id_)
                            select_moltenprot_well(state, "0", row_num)
                        multi_sort, set_multi_sort = solara.use_state(False)
                        with solara.Row(justify="end"):
                            solara.Switch(label="Multi Sort", value=multi_sort, on_value=set_multi_sort)
                        data_table = rv.DataTable(items=state.value.params_df.to_dict("records"),
                                    headers=return_df_headers_step4_1(),
                                    items_per_page=-1,
                                    multi_sort=multi_sort,
                                    hide_default_footer=True,
                                    height="calc(100vh - 300px)",
                                    fixed_header=True,
                                    dense=True,
                        )
                        rv.use_event(data_table, "click:row", select_row)
                        

                    def return_df_headers_step4_2():
                        headers_ = [{"text": col, "value": col, "align": "left"} for col in state.value.values_df.columns.to_list()]
                        headers_[0] = {**headers_[0], "width": "125px"} # Temperature
                        return headers_
                    with solara.Card("Baseline Corrected"):
                        rv.DataTable(items=state.value.values_df.to_dict("records"),
                                    headers=return_df_headers_step4_2(),
                                    items_per_page=-1,
                                    hide_default_footer=True,
                                    height="calc(100vh - 300px)",
                                    fixed_header=True,
                                    dense=True,
                        )
                    solara.Button(label="Next Step", on_click=partial(set_tab_index, 4, state),
                                    style={"margin": "8px", "background-color": "#6aa7f7"})

                    
            """Step 5"""
            with solara.lab.Tab("5/ Run Analysis", disabled=not show_step5, tab_children=[DeleteStepData(state=state, step_index=5)
] if tab_index.value == 4 else []):
                solara.Markdown(r'''
                    ## Run statistical analysis :
                    * A platemap needs to be uploaded first.
                    * When completed, a result dataframe will be displayed.
                    ''',
                    style={"margin": "8px"})
                with solara.Card(title="Step 5: Run Analysis"):

                    solara.Markdown(state.value.status_step5)
                    solara.ProgressLinear(run_analysis.pending)
                    if run_analysis.error:
                        state.value = replace(state.value, status_step5=str(run_analysis.exception))

                    if state.value.platemap:
                        solara.Markdown(f"Platemap {state.value.platemap.name} file found.")

                    solara.FileDrop(
                        label="Drag and drop platemap file.",
                        on_file=partial(upload_plate_params, state=state, upload_progress=platemap_upload_progress),
                        lazy=False,
                        on_total_progress=platemap_upload_progress.set,
                    )
                    
                    with solara.CardActions():
                        with solara.Row(gap="2px", margin="2px", justify="center"):
                            if show_run_analysis:
                                solara.Button("Analyze", on_click=partial(run_analysis, state))
                
                if state.value.signif_df is not None:     
                    with solara.Card():
                        solara.DataFrame(state.value.signif_df, scrollable=True)
