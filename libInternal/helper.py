import os
from datetime import datetime
import pandas as pd

def variableDump(var, name="var", indent=0, show_id=False):
    spacing = "  " * indent
    type_name = type(var).__name__

    if show_id:
        id_str = f" @id={id(var)}"
    else:
        id_str = ""

    if isinstance(var, dict):
        print(f"{spacing}{name} (dict, {len(var)} items){id_str}:")
        for k, v in var.items():
            variableDump(v, f"[{repr(k)}]", indent + 1, show_id)
    elif isinstance(var, (list, tuple, set)):
        typename = type(var).__name__
        print(f"{spacing}{name} ({typename}, {len(var)} items){id_str}:")
        for i, v in enumerate(var):
            variableDump(v, f"[{i}]", indent + 1, show_id)
    else:
        print(f"{spacing}{name} ({type_name}){id_str}: {repr(var)}")

def setFileLocation():
    fileTimeStamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    output_dir = os.path.join(project_root, "assets/outputs")
    os.makedirs(output_dir, exist_ok=True)

    return fileTimeStamp, output_dir

def setExportDataLocation():
    fileTimeStamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    output_dir = os.path.join(project_root, "assets/outputsData")
    os.makedirs(output_dir, exist_ok=True)

    return fileTimeStamp, output_dir

def cleanYear(val):
    if pd.isna(val):
        return None
    val = str(val)
    if "/" in val:
        return int(val.split("/")[1])
    return int(val)