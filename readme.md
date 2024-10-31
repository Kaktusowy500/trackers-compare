## Tracker Compare

Software for analyzing and comparing object tracking algorithms.

### Dependency Installation

1. Clone the repository with the recursive option:

    ```bash
    git clone git@github.com:Kaktusowy500/trackers-compare.git --recursive
    ```

2. Install dependencies by running:

    ```bash
    sudo ./dependencies.sh
    ```

3. Build OpenCV 4.9.0 with contrib modules from source according to the [official documentation](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) and set the `OPENCV_DIR` variable in the main `CMakeLists.txt` to the build directory.

4. For Python dependencies, run the following command (it is recommended to use a virtual environment):

    ```bash
    pip install -r requirements.txt
    ```

### Building the Project

Run the following commands in the main repository directory:

```bash
mkdir build && cd build
cmake ..
make -j10
```



### Changing logging verbosity by enviroment variable:
Set env to desired logging level, eg.:
```bash
export SPDLOG_LEVEL=debug
```
Avalaible levels: trace, debug, info, warn, error, critical

### Run 
To run the app in evaluation mode:
```
./build/tracker_compare <path_to_dataset>
```
Results will be saved in the `runs/date-time` directory.

To create plots and tables with a summary: 
```
python python-utils/summary_table.py <path_to_results_directory>
```
Generated tables and plots will be saved in the results directory, under the `plots` subdirectory
