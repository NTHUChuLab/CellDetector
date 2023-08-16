import numpy as np
import pandas as pd
from brainrender import settings

settings.SHOW_AXES = False
settings.WHOLE_SCREEN = True

from rich import print
from myterial import orange
from pathlib import Path

print(f"[{orange}]Running example: {Path(__file__).name}")

screenshot_folder = "/fig_render"
scene = Scene(inset=False, screenshots_folder=screenshot_folder)

brain_region_df = pd.read_excel("/brain_region_sub.xlsx")
for i in range(len(brain_region_df["brain_region"])):
    region = scene.add_brain_region(brain_region_df["brain_region"][i], alpha=0.5, silhouette=True, color=[0, 0.5, 0.8])

scene.render(interactive=False)
scale = 3

scene.screenshot(name="brainrender_shot_sub", scale=scale)
scene.close()
