## Tracker compare

Software for comparing object tracking algorithms and preparing a custom dataset for comparision.

### Dependency install
- clone repo with recursive option
- install dependencies with by running dependencies.sh
- build opencv 4.9.0 with contrib modules from source, according to the [docs](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) and set OPENCV_DIR variable in main CMakeLists.txt

### Changing logging verbosity by env
Set env to desired logging level, eg.:
`SPDLOG_LEVEL=debug`