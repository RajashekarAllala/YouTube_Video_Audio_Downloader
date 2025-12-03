# 📥 **YouTube Video & Audio Downloader**  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9--3.12-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/yt--dlp-latest-red?style=for-the-badge&logo=youtube&logoColor=white" />
  <img src="https://img.shields.io/badge/FFmpeg-required-green?style=for-the-badge&logo=ffmpeg&logoColor=white" />
  <img src="https://img.shields.io/badge/Build-PyInstaller-orange?style=for-the-badge&logo=windows&logoColor=white" />
  <img src="https://img.shields.io/github/license/RajashekarAllala/YouTube_Video_Audio_Downloader?style=for-the-badge" />
</p>

# 📌 **Overview**

A powerful, user-friendly downloader for YouTube videos & playlists with:

- A clean **Tkinter GUI**
- Full **yt-dlp** backend for stability
- **FFmpeg** support for merging & audio conversion
- Persistent **download queue** with resume support
- **Light & Dark themes**
- Cross-platform (Windows, macOS, Linux)
- Fully packaged EXE support using **PyInstaller**

> **⚠️ Important Note:**  
>
> This application **cannot download** videos that are:
> - Premium / YouTube Premium  
> - Private videos  
> - Member-only videos (Channel Members Only)  
> - Age-restricted videos requiring login  
> - yt-dlp (and YouTube API) block downloading restricted content **without proper authentication**, which this app does not support by design.

---

# 🚀 **Features**

### 🎞️ **Video Downloading**
- Download the **highest quality** automatically  
- Choose resolution (1080p / 720p / 480p…)  
- Automatic fallback when resolution unavailable  

### 🎧 **Audio Mode**
- Extract audio in:  
  - MP3  
  - AAC  
  - Opus  
- Configurable bitrate (default: 192k)

### 📂 **Queue Management**
- Add items to queue  
- Move items **Up / Down**  
- **Retry** failed downloads  
- **Cancel** active downloads
- **Clear** queue
- **Remove** selected URLs in the queued state
- Queue persists even after closing the app  

### 📑 **Playlist Support**
- Expands playlist into individual tasks  
- Saves inside a dedicated folder  

### 🎨 **Themes & Settings**
- Light / Dark theme  
- Custom FFmpeg path  
- Save window geometry  
- Auto-detection of bundled FFmpeg  

---

# **📸 Screenshots**

### Main Window  
<img width="2880" height="1698" alt="image" src="https://github.com/user-attachments/assets/071e45b8-1580-4f43-a943-4bd28d1225b1" />


### Settings Window
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/24453aa1-5262-410f-aa88-3f82213013e2" />


### Download Queue
<img width="2854" height="1106" alt="image" src="https://github.com/user-attachments/assets/52a42752-d3aa-498c-a249-be8c76792725" />

---

# 📦 Installation

## ✅ Prerequisites
- Python **3.9 – 3.12**
  https://www.python.org/downloads/
- Dependencies:
  ```bash
  pip install yt-dlp
  pip install pyinstaller
  pip install tkinter  # Linux only

## ✅ **FFmpeg Setup**

### **Windows (Recommended)**

Download FFmpeg:  
https://www.gyan.dev/ffmpeg/builds/

Extract to: 
- C:\ffmpeg\

Executable path: 
- C:\ffmpeg\bin\ffmpeg.exe

✔ FFmpeg installed Required for:

- Video + audio merging
- Audio conversion (mp3, aac, opus)
- Post-processing

If FFmpeg is missing, yt-dlp will show:
- “ERROR: ffmpeg is not installed. Aborting due to --abort-on-error”

- You can fix it by setting FFmpeg in the app Settings.

### **Bundled FFmpeg (Optional)**

Place FFmpeg inside your project where the python script is placed:

-     ffmpeg/
         |-- bin/
               |--ffmpeg.exe


The app will automatically detect it:

-     <script_path>/ffmpeg/bin/ffmpeg.exe

---

## 🎨 **Icon Requirements**

To ensure the EXE shows the correct icon:

### ✅ **Place the icon file next to your main script**
- Ensure that **icon.ico** or **icon.png** is present in the **same directory as the main script** before building.  
- PyInstaller requires the icon file to exist **physically at build time**:

Your project structure must look like this:
-     project/
         |-- yt_downloader.py
         |-- icon.ico
         |-- ffmpeg/
         |-- config/
---

## ▶️ **Running the App - Python Script**

-     python YouTube_Video_Audio_Downloader.py

---

## **🏗️ Building an EXE (PyInstaller)**

**🟦 Option A — OneDir build (recommended)**

Includes folders (best for FFmpeg bundling):
**Note:** You can use icon.ico if available, otherwise, use icon.png.

-     pyinstaller --noconfirm --clean --onedir \
      --name YouTube_Downloader \
      --windowed \
      --add-data "icon.png;." \
      --icon=icon.png \
      --add-binary "ffmpeg/bin/ffmpeg.exe;ffmpeg/bin" \
      YouTube_Video_Audio_Downloader.py

**Produces:**

-     dist/
        |-- YouTube_Downloader/
                    |-- YouTube_Downloader.exe
                    |-- ffmpeg/bin/ffmpeg.exe

**🟥 Option B — OneFile build**

⚠ FFmpeg must be external or inside the bundle folder.

-     pyinstaller --noconfirm --clean --onefile \
      --name YouTube_Downloader \
      --windowed \
      --icon icon.png \
      --add-binary "ffmpeg/bin/ffmpeg.exe;ffmpeg/bin" \
      YouTube_Video_Audio_Downloader.py

**Produces:**
-     dist/YouTube_Downloader.exe

⚠ Note: OneFile temporarily extracts files to a runtime folder — FFmpeg must be accessed via:
Path(sys._MEIPASS) / "ffmpeg/bin/ffmpeg.exe"

- Current code already handles this automatically.

---

## **🔷 OneDir vs OneFile — Which Should You Use?**

| Mode        | Pros                                                                       | Cons                                                                                |
| ----------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **OneDir**  | ✔ Easiest FFmpeg bundling<br>✔ Faster startup<br>✔ More stable with yt-dlp | Folder is larger                                                                    |
| **OneFile** | ✔ Single EXE<br>✔ Portable                                                 | ❗ Slower startup<br>❗ FFmpeg must be extracted<br>❗ Runtime folder can cause issues |


Recommendation: OneDir build (more stable for yt-dlp + FFmpeg)

---

## **🧰 Troubleshooting**
#### ❗ FFmpeg is not installed. Aborting...

Solution:
- Go to Settings → FFmpeg path → Browse
- Select: ffmpeg/bin/ffmpeg.exe from the ffmpeg installation path

#### ❗ FFmpeg not bundled in EXE
 - Ensure to add while creating the exe installer:
-     --add-binary "ffmpeg/bin/ffmpeg.exe;ffmpeg/bin"

#### ❗ Long startup time in OneFile mode

- Normal — PyInstaller extracts all files on each launch. Use OneDir for instant startup.

#### ❗ "Resolution not available"

- The app will automatically download the best available format.

#### ❗ **GUI Freezes**

- Avoid downloading **very large playlists**.  
  Extracting hundreds or thousands of video URLs takes time, and building the task queue can temporarily freeze the UI.

- Avoid running **heavy operations on the Tkinter main thread**.  
  All downloads already run in background threads, but extremely large metadata operations may still cause short UI delays.

---

## **📄 License**
MIT License – free to use, modify, and distribute.
© 2025 Rajashekar Allala
