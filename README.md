AnimalAccM is an open-source graphical user interface for automated behavior analytics of individual animals using triaxial accelerometers and machine learning.

Python packages required: _pandas, pathlib, glob, matplotlib, seaborn, sklearn, train_test_split, joblib, math, imblearn_


An example for the operation procedures:

**Step 1:** Run the ‘MainWindow.py’ file and launch the interface. ‘Home’ page shows up.

**Step 2:** Click button ‘Manage Project’ to switch the page. If a new project will be created, go to step 3. If the project is existing, go to step 4.

**Step 3:** Click button ‘Project name’ and input a project name; click button ‘Experimenter name’ and input an experimenter name; click button ‘Date (MMDDYY)’ and input a project date; click button ‘Row data directory’ and input a path of data for model development. Finally click button ‘Create the project’ to create a new project.

**Step 4:** Click button ‘Browse’ and select a configuration file in CSV format. Click button ‘Load the project’ to load an existing project.

**Step 5:** Click button ‘Preprocess Data’ to switch the page. Click button ‘Sampling rate’ and input a sampling rate with the unit of Hz; click button ‘Window size’ and input a window size with the unit of second; and click button ‘Step size’ and input a step size with the unit of second. Click button ‘Start the extraction’ to extract features.

**Step 6:** Click button ‘Check data balance’ and watch a pop-up plot for distribution of number of data points for each class. Click button ‘Number of combined classes’ and input class name with comma separators and without spaces. Click button ‘Start the cleaning’ to clean unqualified data points. If users want to retain the minority of interest and importance, they can leave the ‘Number of combined classes’ blank and click button ‘Start the cleaning’ to make the program move forward.  

**Step 7:** Click button ‘Develop Models’ page to switch the page. Click button ‘Select a model’ and select a model; and click button ‘Validation ratio’ to input a validation ratio. Click button ‘Start the model training’ to train a model. Click button ‘Start the evaluation’ to display a confusion matrix, precision, recall, F1 score, and accuracy on the interface. 

**Step 8:** Click button ‘Analyze Behavior’ to switch the page. Click button ‘Select the trained model’ to select a trained model. Click button ‘Select the file for analysis’ to select a file to be analyzed. Click button ‘Load key parameters for analysis’ to load key parameters, such as sampling rate, window size, and step size. Click button ‘Start the analysis’ to launch the behavior analysis. Click button ‘Visualized example results’ to display time budgets of detected behaviors. 
