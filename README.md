# 🔬 Next Experiment Data Driven (NEDD)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### ✨ LLM Hackathon for Materials & Chemistry 2025

> **Smarter experiments, fewer wasted steps.**

---

## 🔎 What is NEDD?

NEDD is a **UI + AI Agent** designed to help scientists plan their **next experiments** in a **data-driven** way.

When experiments are run, the parameter space is often explored **non-uniformly**. This leads to:

- ⚠️ **Skewed data distributions**
- ⚠️ **Regions of ignorance** (parameters left unexplored)
- ⚠️ **Poor ML model training** on biased datasets

NEDD helps prevent this by **visualizing parameter space**, highlighting **gaps in exploration**, and guiding **next step suggestions** with ML + LLM support.

---

## 📊 Features

- **Load & explore data**: Import your experimental dataset easily.
- **Visualize parameter space**: Interactive pairplots show skewness and regions of ignorance.
- **ML-ready datasets**: Ensure balanced coverage for better predictive modeling.
- **Active Learning loop**: Suggest new experiments that maximize knowledge gain.
- **LLM assistant**: Query your data in natural language and get experiment guidance.

---

## 🌟 Why does it matter?

Machine learning models thrive on **diverse and representative data**.  
If experiments are clustered in one corner of the parameter space, the model can’t generalize well.  
NEDD ensures:

- ✅ **Uniform coverage of experimental space**
- ✅ **Reduced bias in datasets**
- ✅ **Fewer experiments needed** for strong ML predictions
- ✅ **Smarter, faster materials discovery**

---

## 🚀 Future Vision

We imagine NEDD as a **LLM-assisted experiment planner**, where you can:

- Drop in your dataset
- Chat with the AI about trends and missing regions
- Get **optimized experiment suggestions** powered by ML, not randomness
- Minimize experimental effort while maximizing discovery potential

---

## 💡 Example Workflow

1. Load your dataset.
2. Visualize with a pairplot → spot skewness.
3. Train ML → identify uncertain regions.
4. Use NEDD’s Active Learning agent → propose next 4–5 most informative experiments.
5. Iterate until your model converges.

---

## ▶️ How to Run

1. `streamlit run main.py`
2. Open `http://localhost:8501/`

---

## 🌸 Hackathon Spirit

This project is being developed during the **LLM Hackathon for Materials & Chemistry 2025**.  
It’s a prototype with big dreams: **bringing AI + ML + LLMs together to accelerate experimental science**.

## ⚙️ If you have Windows

<details>
<summary><b>Recommended setup for corporate or restricted machines where software installers are blocked by IT security policies.</b></summary>

**Install Git**

1. Download and install Git for Windows from
   👉[https://git-scm.com/downloads/win](https://git-scm.com/downloads/win)
2. You can follow this YouTube tutorial for guidance:
   - 🎥[Git Installation on Windows](https://www.youtube.com/watch?v=Av7lcVIbEBY)
     When prompted during installation:
   - Choose Nano as default editor.
   - Leave all other settings at their default values.
   - Check the option to Launch Git Bash at the end.
3. A terminal window titled MINGW64 should open automatically.
   This is your Git Bash terminal, where you will run all commands.

**Get NEDD (one-time setup)**

1. Create a folder for the project.
   For example, open your Downloads folder and create a new folder called _proj_.
   > Avoid using folders synchronized with cloud storage (e.g., OneDrive), because the final project size is approximately 1.4 GB.
2. In the Git Bash (MINGW64) terminal, navigate to that folder:
   `cd ~/Downloads/proj`
3. Clone the NEDD repository:
   `git clone https://github.com/ViktoriiaBaib/NEDD.git`
   After it finishes, confirm that a new folder named NEDD appears inside _proj_.
4. Download the pre-built Conda environment:
   👉[nedd_env.zip (GitHub Actions Artifact)](https://github.com/ViktoriiaBaib/NEDD/actions/runs/18231289768/artifacts/4178307048)
5. Place the downloaded file here:
   `~/Downloads/proj/NEDD/env/nedd_env.zip`
6. Unzip _nedd_env.zip_ by double-clicking it.
   This will create a new folder called _nedd_env_ inseide the _env/_ directory.
7. Update internal paths inside the environment (one-time only).
   In Git Bash (MINGW64), navigate to NEDD and run:
   `cd ~/Downloads/proj/NEDD`
   `./env/nedd_env/Scripts/conda-unpack.exe`
   Wait until it completes (this may take a minute).

**Run the app (every time)**

- Double-click `run_NEDD.cmd` inside the _NEDD_ folder.
  A command window will open and display log messages.
  When finished, close the window to stop the app.

✅ That's it!
The app will open in your default browser (recommended: Chrome) at
[http://localhost:8501](http://localhost:8501)

</details>
