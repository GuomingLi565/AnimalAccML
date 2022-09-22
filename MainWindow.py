import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from Ui_MainWindow import Ui_MainWindow
import os
import shutil
import pandas as pd
from HelperFunction import *

class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()

        ########## default parameters ##########
        self.ProjectName = str()
        self.ExperimenterName = str()
        self.ProjectDate = str()
        self.RowDataDirectory = str()
        self.MainDirectory = str()
        self.ModelPath = str()
        self.DataPath = str()
        self.PlotsPath = str()
        self.MetricsPath = str()
        self.ResultPath = str()
        self.FeaturePath = str()
        self.SamplingRate = str()
        self.WindowSize = str()
        self.StepSize = str()
        self.ClassNames = str()
        self.ModelName = str()
        self.TrainValRatio = str()
        self.TrainedModelPath = str()
        self.FilePath_BehaviorAnalysis = str()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)

        self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)

        ########## Switching pages ##########
        self.ui.home_btn.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_home))
        self.ui.ManageProject_btn.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_ManageProject))
        self.ui.DataPreprocessing_btn.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_DataPreprocessing))
        self.ui.ModelDevelopment_btn.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_ModelDevelopment))
        self.ui.BehaviorAnalysis_btn.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_BehaviorAnalysis))

        ########## Manage Project page operations ##########
        self.ui.PushButton_ProjectName.clicked.connect(self.getProjectName)
        self.ui.PushButton_ExperimenterName.clicked.connect(self.getExperimenterName)
        self.ui.PushButton_Date.clicked.connect(self.getProjectDate)
        self.ui.PushButton_RowDataDirectory.clicked.connect(self.getRowDataDirectory)
        self.ui.PushButton_Create.clicked.connect(self.CreateTheProject)
        self.ui.PushButton_Browse.clicked.connect(self.getConfigPath)
        self.ui.PushButton_Loading.clicked.connect(self.LoadExistingProject)

        ########## Data Preprocessing page operations ##########
        self.ui.PushButton_SamplingRate.clicked.connect(self.getSamplingRate)
        self.ui.PushButton_WindowSize.clicked.connect(self.getWindowSize)
        self.ui.PushButton_StepSize.clicked.connect(self.getStepSize)
        self.ui.PushButton_StartExtraction.clicked.connect(lambda: FeatureExtraction(self.MainDirectory, self.SamplingRate, self.WindowSize, self.StepSize, self.DataPath, self.FeaturePath))
        self.ui.PushButton_CheckDataBalance.clicked.connect(lambda: CheckDataBalance(self.PlotsPath, self.FeaturePath))
        self.ui.PushButton_ClassNames.clicked.connect(self.getClassNames)
        self.ui.PushButton_StartCleaning.clicked.connect(lambda: ClassCombination(self.FeaturePath, self.ClassNames))

        ########## Model Development page operations ##########
        self.ui.PushButton_ModelSelection.clicked.connect(self.getModelNames)
        self.ui.PushButton_TrainValRatio.clicked.connect(self.getTrainValRatio)
        self.ui.PushButton_StartTraining.clicked.connect(lambda: ModelTraining(self.MetricsPath, self.ModelName, self.FeaturePath, self.ModelPath, self.PlotsPath, self.TrainValRatio))
        self.ui.PushButton_StartEvaluation.clicked.connect(self.displayEvaluationResults)

        ########## Behavior Analysis page operations ##########
        self.ui.PushButton_SelectTrainedModel.clicked.connect(self.getTrainedModelPath)
        self.ui.PushButton_SelectAnalyzedFile.clicked.connect(self.getAnalyzedFile)
        self.ui.PushButton_LoadKeyParameters.clicked.connect(self.getKeyParameters)
        self.ui.PushButton_StartAnalysis.clicked.connect(lambda: BehaviorAnalysis(self.SamplingRate, self.WindowSize, self.StepSize, self.TrainedModelPath, self.FilePath_BehaviorAnalysis, self.ResultPath, self.PlotsPath))
        self.ui.PushButton_VisualizeExampleResults.clicked.connect(self.visualizeBehaviorResults)
        self.main_win.show()

    def getProjectName(self):
        self.ProjectName, ok = QInputDialog.getText(self.ui.page_ManageProject, 'Text Input', 'Enter the project name:')
        if ok:
            self.ui.LineEdit_ProjectName.setText(str(self.ProjectName))
        return self.ProjectName

    def getExperimenterName(self):
        self.ExperimenterName, ok = QInputDialog.getText(self.ui.page_ManageProject, 'Text Input', 'Enter the experimenter name:')
        if ok:
            self.ui.LineEdit_ExperimenterName.setText(str(self.ExperimenterName))
        return self.ExperimenterName

    def getProjectDate(self):
        self.ProjectDate, ok = QInputDialog.getText(self.ui.page_ManageProject, 'Text Input', 'Enter the project date:')
        if ok:
            self.ui.LineEdit_Date.setText(str(self.ProjectDate))
        return self.ProjectDate

    def getRowDataDirectory(self):
        self.RowDataDirectory, ok = QInputDialog.getText(self.ui.page_ManageProject, 'Text Input', 'Enter the row data directory:')
        if ok:
            self.ui.LineEdit_RowDataDirectory.setText(str(self.RowDataDirectory))
        return self.RowDataDirectory

    def getConfigPath(self):
        filename = QFileDialog.getOpenFileName(self.ui.page_ManageProject)
        self.ConfigPath = filename[0]
        return self.ConfigPath

    def getSamplingRate(self):
        self.SamplingRate, ok = QInputDialog.getText(self.ui.page_DataPreprocessing, 'Text Input', 'Enter the sampling rate (Hz):')
        if ok:
            self.ui.LineEdit_SamplingRate.setText(str(self.SamplingRate))
        return self.SamplingRate

    def getWindowSize(self):
        self.WindowSize, ok = QInputDialog.getText(self.ui.page_DataPreprocessing, 'Text Input', 'Enter the window size (second):')
        if ok:
            self.ui.LineEdit_WindowSize.setText(str(self.WindowSize))
        return self.WindowSize

    def getStepSize(self):
        self.StepSize, ok = QInputDialog.getText(self.ui.page_DataPreprocessing, 'Text Input', 'Enter the step size (second):')
        if ok:
            self.ui.LineEdit_StepSize.setText(str(self.StepSize))
        return self.StepSize

    def getClassNames(self):
        self.ClassNames, ok = QInputDialog.getText(self.ui.page_DataPreprocessing, 'Text Input', 'Enter the names of combined classes (separate the classes with comma):')
        ClassNames = self.ClassNames
        if ok:
            self.ui.LineEdit_ClassNames.setText(str(ClassNames))
        self.ClassNames = self.ClassNames.split(",")
        print(self.ClassNames)
        return self.ClassNames

    def getModelNames(self):
        ModelNames = ("AdaBoostClassifier", "BaggingClassifier", "BernoulliNB", "DecisionTreeClassifier", "ExtraTreeClassifier", "ExtraTreesClassifier", "GaussianNB",
                      "GaussianProcessClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier", "LinearDiscriminantAnalysis", "LinearRidge",
                      "QuadraticDiscriminantAnalysis", "RandomForestClassifier", "SGDClassifier", "SVM")
        self.ModelName, ok = QInputDialog.getItem(self.ui.page_ModelDevelopment, "Select input dialog", "List of models", ModelNames, 0, False)
        if ok:
            self.ui.LineEdit_ModelSelection.setText(str(self.ModelName))
        return self.ModelName

    def getTrainValRatio(self):
        self.TrainValRatio, ok = QInputDialog.getText(self.ui.page_ModelDevelopment, 'Text Input', 'Validation Ratio:')
        if ok:
            self.ui.LineEdit_TrainValRatio.setText(str(self.TrainValRatio))
        return self.TrainValRatio

    def getTrainedModelPath(self):
        filename = QFileDialog.getOpenFileName(self.ui.page_BehaviorAnalysis)
        self.TrainedModelPath = filename[0]
        print("Loading the trianed model successfully!")
        return self.TrainedModelPath

    def getAnalyzedFile(self):
        filename = QFileDialog.getOpenFileName(self.ui.page_BehaviorAnalysis)
        self.FilePath_BehaviorAnalysis = filename[0]
        print("Loading the file successfully!")
        return self.FilePath_BehaviorAnalysis

    def getKeyParameters(self):
        filename = QFileDialog.getOpenFileName(self.ui.page_BehaviorAnalysis)
        ParameterPath = filename[0]
        Parameters = pd.DataFrame(pd.read_csv(ParameterPath, index_col=0))
        self.SamplingRate = Parameters.loc["SamplingRate", "KeyParameters"]
        self.WindowSize = Parameters.loc['WindowSize', 'KeyParameters']
        self.StepSize = Parameters.loc['StepSize', 'KeyParameters']
        print('Loading the parameters successfully!')
        return self.SamplingRate, self.WindowSize, self.StepSize

    def LoadExistingProject(self):
        df_config = pd.DataFrame(pd.read_csv(self.ConfigPath, header=None))
        self.MainDirectory = df_config[0][0]
        self.ModelPath = df_config[0][1]
        self.DataPath = df_config[0][2]
        self.PlotsPath = df_config[0][3]
        self.ResultPath = df_config[0][4]
        self.FeaturePath = df_config[0][5]
        self.MetricsPath = df_config[0][6]
        print("Loading was completed successfully!")
        return self.MainDirectory, self.ModelPath, self.DataPath, self.PlotsPath, self.ResultPath, self.FeaturePath, self.MetricsPath


    def CreateTheProject(self):
        ProjectName = self.ProjectName.replace(" ", "")
        ExperimenterName = self.ExperimenterName.replace(" ", "")
        ProjectDate = self.ProjectDate.replace(" ", "")
        self.MainDirectory = str(ProjectName) + "_" + str(ExperimenterName) + "_" + str(ProjectDate)
        self.ModelPath = self.MainDirectory+'/Models'
        self.DataPath = self.MainDirectory+'/RowData'
        self.PlotsPath = self.MainDirectory+'/Plots'
        self.ResultPath = self.MainDirectory+'/BehaviorResults'
        self.FeaturePath = self.MainDirectory+'/Features'
        self.MetricsPath = self.MainDirectory+'/EvaluationMetrics'

        DirectoryList = [self.ModelPath, self.DataPath, self.PlotsPath, self.ResultPath, self.FeaturePath, self.MetricsPath]

        os.makedirs(self.MainDirectory, exist_ok=True)

        for directory in DirectoryList:
            os.makedirs(directory, exist_ok=True)

        src_dir = self.RowDataDirectory
        src_dir = src_dir.replace(os.sep, '/')

        dest_dir = os.getcwd()
        dest_dir = os.path.join(dest_dir, self.DataPath)
        dest_dir = dest_dir.replace(os.sep, '/')
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

        SaveDirectoryList = pd.DataFrame([self.MainDirectory, self.ModelPath, self.DataPath,
                                          self.PlotsPath, self.ResultPath, self.FeaturePath, self.MetricsPath])
        config_name = str(self.MainDirectory) + "/" + str(self.MainDirectory) + "_config.csv"
        SaveDirectoryList.to_csv(config_name, index=False, header=False)

        print("Creating the project successfully!")
        return self.MainDirectory, self.ModelPath, self.DataPath, self.PlotsPath, self.ResultPath, self.FeaturePath, self.MetricsPath

    def displayEvaluationResults(self):
        MetricsPath = self.MetricsPath + "/Metrics_" + str(self.ModelName) + '_' + str(self.TrainValRatio) + ".csv"
        data = pd.DataFrame(pd.read_csv(MetricsPath, index_col=0))
        av_precision = round(data.loc["macro avg", "precision"] * 100, 1)
        av_recall = round(data.loc["macro avg", "recall"] * 100, 1)
        av_f1_score = round(data.loc["macro avg", "f1-score"] * 100, 1)
        weighted_precision = round(data.loc["weighted avg", "precision"] * 100, 1)
        weighted_recall = round(data.loc["weighted avg", "recall"] * 100, 1)
        weighted_f1_score = round(data.loc["weighted avg", "f1-score"] * 100, 1)
        overall_accuracy = round(data.loc["accuracy", "support"] * 100, 1)
        self.ui.LineEdit_av_precision.setText(str(av_precision))
        self.ui.LineEdit_av_recall.setText(str(av_recall))
        self.ui.LineEdit_av_f1_score.setText(str(av_f1_score))
        self.ui.LineEdit_weighted_precision.setText(str(weighted_precision))
        self.ui.LineEdit_weighted_recall.setText(str(weighted_recall))
        self.ui.LineEdit_weighted_f1_score.setText(str(weighted_f1_score))
        self.ui.LineEdit_overall_accuracy.setText(str(overall_accuracy))
        figurepath = self.PlotsPath + "/ConfusionMatrix_" + str(self.ModelName) + '_' + str(self.TrainValRatio) + ".png"
        self.ui.image_ConfusionMatrix.setPixmap(QPixmap(figurepath))
        print("Evaluation was completed")

    def visualizeBehaviorResults(self):
        ResultPath = self.ResultPath + "/TimeBudget.txt"
        data = pd.DataFrame(pd.read_csv(ResultPath, index_col=0))
        data_index = data.index
        self.ui.QLabel_Top_1_Name.setText(str(data_index[0] + " (second)"))
        self.ui.QLabel_Top_2_Name.setText(str(data_index[1] + " (second)"))
        self.ui.QLabel_Top_3_Name.setText(str(data_index[2] + " (second)"))
        self.ui.LineEdit_Top_1_Behavior.setText(str(data.iloc[0][0]))
        self.ui.LineEdit_Top_2_Behavior.setText(str(data.iloc[1][0]))
        self.ui.LineEdit_Top_3_Behavior.setText(str(data.iloc[2][0]))
        figurepath = self.PlotsPath + "/BehaviorResults.png"
        self.ui.image_BehaviorBudget.setPixmap(QPixmap(figurepath))
        print('Visualization was completed!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    sys.exit(app.exec_())