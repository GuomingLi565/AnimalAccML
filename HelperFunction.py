import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import *
from sklearn.metrics import classification_report, plot_confusion_matrix
import joblib
import warnings
from math import *
from imblearn.over_sampling import RandomOverSampler

def FeatureExtraction(MainDirectory, SamplingRate, WindowSize, StepSize, DataPath, FeaturePath):
    warnings.filterwarnings("ignore")
    filepath = str(DataPath)+"/*.csv"
    WindowLength = int(int(SamplingRate) * float(WindowSize))
    StepLength = int(int(SamplingRate) * float(StepSize))
    data_combined = pd.DataFrame()

    lst = [SamplingRate, WindowSize, StepSize, WindowLength, StepLength]
    lst_df = pd.DataFrame(lst, index=['SamplingRate', 'WindowSize', 'StepSize', 'WindowLength', 'StepLength'], columns=['KeyParameters'])
    lst_save_name = MainDirectory + "/KeyParameters_DataProcessing.csv"
    lst_df.to_csv(lst_save_name)

    for file in glob.glob(filepath):
        test_object = Path(file).stem
        data = pd.DataFrame(pd.read_csv(file))
        labels = data['label']
        data_formatted = data.drop(['label'], axis=1).copy()
        range_end = len(data_formatted) - WindowLength
        data_temp = pd.DataFrame()

        for i in range(0, range_end, StepLength):
            data_selected = data_formatted[i: i + WindowLength]

            # features
            data_kurt = data_selected.kurtosis()
            data_mad = data_selected.mad()
            data_mean = data_selected.mean()
            data_median = data_selected.median()
            data_min = data_selected.min()
            data_max = data_selected.max()
            data_quan_25 = data_selected.quantile(q=0.25)
            data_quan_50 = data_selected.quantile(q=0.5)
            data_quan_75 = data_selected.quantile(q=0.75)
            data_skew = data_selected.skew()
            data_sum = data_selected.sum()
            data_std = data_selected.std()
            data_var = data_selected.var()
            SignalMagnitudeArea = abs(data_formatted.loc[i][0]) + abs(data_formatted.loc[i][1]) + abs(data_formatted.loc[i][2])
            VectorMagnitude = sqrt(abs(data_formatted.loc[i][0] + data_formatted.loc[i][1] + data_formatted.loc[i][2]))
            MovementVariation = abs(data_formatted.loc[i][0]-data_formatted.loc[i+1][0]) + abs(data_formatted.loc[i][1]-data_formatted.loc[i+1][1]) + abs(data_formatted.loc[i][2]-data_formatted.loc[i+1][2])
            Energy = ((data_formatted.loc[i][0])**2 + (data_formatted.loc[i][1])**2 + (data_formatted.loc[i][2])**2)**2
            Entropy = (1+(data_formatted.loc[i][0] + data_formatted.loc[i][1] + data_formatted.loc[i][2]))**2 * log(abs(1+data_formatted.loc[i][0] + data_formatted.loc[i][1] + (data_formatted.loc[i][2])**2))

            data_combined_hor = pd.concat(
                [pd.Series(test_object), pd.Series(labels.loc[i]), pd.Series(data_formatted.loc[i]),
                 data_kurt, data_mad, data_mean, data_median, data_min, data_max, data_quan_25,
                 data_quan_50, data_quan_75, data_skew, data_sum, data_std, data_var,
                 pd.Series(SignalMagnitudeArea), pd.Series(VectorMagnitude), pd.Series(MovementVariation), pd.Series(Energy), pd.Series(Entropy)], axis=0, ignore_index=True)

            data_temp = data_temp.append(data_combined_hor, ignore_index=True)

        data_combined = data_combined.append(data_temp, ignore_index=True)


    data_combined.columns = ['test_object', 'label', 'AccX', 'AccY', 'AccZ', 'AccX_kurt', 'AccY_kurt', 'AccZ_kurt', 'AccX_mad',
                             'AccY_mad', 'AccZ_mad', 'AccX_mean', 'AccY_mean', 'AccZ_mean', 'AccX_median', 'AccY_median',
                             'AccZ_median', 'AccX_min', 'AccY_min', 'AccZ_min', 'AccX_max', 'AccY_max', 'AccZ_max',
                             'AccX_quan_25', 'AccY_quan_25', 'AccZ_quan_25', 'AccX_quan_50', 'AccY_quan_50', 'AccZ_quan_50',
                             'AccX_quan_75', 'AccY_quan_75', 'AccZ_quan_75', 'AccX_skew', 'AccY_skew', 'AccZ_skew', 'AccX_sum',
                             'AccY_sum', 'AccZ_sum', 'AccX_std', 'AccY_std', 'AccZ_std', 'AccX_var', 'AccY_var', 'AccZ_var',
                             'SignalMagnitudeArea', 'VectorMagnitude', 'MovementVariation', 'Energy', 'Entropy']
    print(len(data_combined))
    filename = FeaturePath + "/features.txt"
    data_combined.to_csv(filename, sep=',', index=False)
    print("Feature extraction was completed!")

def CheckDataBalance(PlotsPath, FeaturePath):
    sns.set()
    filename = FeaturePath + "/features.txt"
    data = pd.DataFrame(pd.read_csv(filename))
    data = data.dropna(axis=0)
    prob = data['label'].value_counts()
    print(prob)
    prob.plot(kind='bar')
    plt.xticks(rotation=25)
    plt.xlabel('Class names')
    plt.ylabel('Counts of data points')
    plt.tight_layout()
    figurename = PlotsPath + "/ClassCountsPlot.png"
    plt.savefig(figurename)
    plt.show()

def ClassCombination(FeaturePath, ClassNames):
    filename = FeaturePath + "/features.txt"
    data = pd.DataFrame(pd.read_csv(filename))
    data.dropna(axis=0, inplace=True)
    # data.reset_index(drop=True, inplace=True)
    # label = data['label'].tolist()
    # label_set = set(label)
    # unique_list = list(label_set)
    # print(data['label'].unique())
    # print(data.head(50))
    # print(unique_list)
    print(len(data))
    combination_list = ClassNames
    # for i in range(len(data)):
    #     if data.iat[i, 1] in combination_list:
    #         data.iat[i, 1] = 'others'
    for combined_label in combination_list:

        # print(combined_label)
        data.loc[(data['label'] == combined_label), 'label'] = 'Others'
        # data = data.replace({'label': {combined_label:'others'}})
    # print(data['label'].unique())
    filename_combined = FeaturePath + "/features_combined.txt"
    print(len(data))
    data.to_csv(filename_combined, sep=',', index=False)
    print("Data cleaning is completed!")

def ModelTraining(MetricsPath, ModelName, FeaturePath, ModelPath, PlotsPath, TrainValRatio):
    #sns.set()
    warnings.filterwarnings("ignore")
    TrainValRatio = float(TrainValRatio)
    filename = FeaturePath+"/features_combined.txt"
    data = pd.DataFrame(pd.read_csv(filename))
    labels = data["label"]
    features = data.drop(["test_object", "label"], axis=1).copy()
    oversample = RandomOverSampler(sampling_strategy='minority')
    features,labels = oversample.fit_resample(features,labels)

    X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=TrainValRatio, random_state=12)
    ModelName = ModelName

    if ModelName == "LinearRidge":
        model = linear_model.RidgeClassifier()
    elif ModelName == "LinearDiscriminantAnalysis":
        model = discriminant_analysis.LinearDiscriminantAnalysis()
    elif ModelName == "QuadraticDiscriminantAnalysis":
        model = discriminant_analysis.QuadraticDiscriminantAnalysis()
    elif ModelName == "SGDClassifier":
        model = linear_model.SGDClassifier()
    elif ModelName == "GaussianProcessClassifier":
        model = gaussian_process.GaussianProcessClassifier()
    elif ModelName == "GaussianNB":
        model = naive_bayes.GaussianNB()
    elif ModelName == "BernoulliNB":
        model = naive_bayes.BernoulliNB()
    elif ModelName == "DecisionTreeClassifier":
        model = tree.DecisionTreeClassifier()
    elif ModelName == "ExtraTreeClassifier":
        model = tree.ExtraTreeClassifier()
    elif ModelName == "BaggingClassifier":
        model = ensemble.BaggingClassifier()
    elif ModelName == "ExtraTreesClassifier":
        model = ensemble.ExtraTreesClassifier()
    elif ModelName == "AdaBoostClassifier":
        model = ensemble.AdaBoostClassifier()
    elif ModelName == "GradientBoostingClassifier":
        model = ensemble.GradientBoostingClassifier()
    elif ModelName == "HistGradientBoostingClassifier":
        model = ensemble.HistGradientBoostingClassifier()
    elif ModelName == "RandomForestClassifier":
        model = ensemble.RandomForestClassifier()
    elif ModelName == "SVM":
        model = svm.SVC()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    EvaluationMetrics = classification_report(y_test, y_pred, output_dict=True)
    EvaluationMetrics = pd.DataFrame(EvaluationMetrics).transpose()

    Metricsname = MetricsPath + "/Metrics_" + str(ModelName) + '_' + str(TrainValRatio) + '.csv'
    EvaluationMetrics.to_csv(Metricsname)

    modelname = ModelPath + "/" + str(ModelName) + '_' + str(TrainValRatio) + ".sav"
    joblib.dump(model, modelname)


    # plot_confusion_matrix(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test, normalize='true', values_format='0.3f')
    # model.plot_confusion_matrix(X_test, y_test)
    plt.grid(False)
    figurename = PlotsPath + "/ConfusionMatrix_" + str(ModelName) + '_' + str(TrainValRatio) + ".png"
    plt.tight_layout()
    plt.savefig(figurename)
    #plt.show()

    print("Model training was completed!")

def BehaviorAnalysis(SamplingRate, WindowSize, StepSize, TrainedModelPath, FilePath_BehaviorAnalysis, ResultPath, PlotsPath):
    sns.set()
    loaded_model = joblib.load(TrainedModelPath)
    data = pd.DataFrame(pd.read_csv(FilePath_BehaviorAnalysis))
    data_formatted = data.dropna(axis=0).copy()
    data_combined = pd.DataFrame()

    WindowLength = int(int(SamplingRate) * float(WindowSize))
    StepLength = int(int(SamplingRate) * float(StepSize))
    range_end = len(data_formatted) - WindowLength

    for i in range(0, range_end, StepLength):
        data_selected = data_formatted[i: i + WindowLength]

        # features
        data_kurt = data_selected.kurtosis()
        data_mad = data_selected.mad()
        data_mean = data_selected.mean()
        data_median = data_selected.median()
        data_min = data_selected.min()
        data_max = data_selected.max()
        data_quan_25 = data_selected.quantile(q=0.25)
        data_quan_50 = data_selected.quantile(q=0.5)
        data_quan_75 = data_selected.quantile(q=0.75)
        data_skew = data_selected.skew()
        data_sum = data_selected.sum()
        data_std = data_selected.std()
        data_var = data_selected.var()

        SignalMagnitudeArea = abs(data_formatted.loc[i][0]) + abs(data_formatted.loc[i][1]) + abs(data_formatted.loc[i][2])
        VectorMagnitude = sqrt(abs(data_formatted.loc[i][0] + data_formatted.loc[i][1] + data_formatted.loc[i][2]))
        MovementVariation = abs(data_formatted.loc[i][0] - data_formatted.loc[i + 1][0]) + abs(data_formatted.loc[i][1] - data_formatted.loc[i + 1][1]) + abs(data_formatted.loc[i][2] - data_formatted.loc[i + 1][2])
        Energy = ((data_formatted.loc[i][0]) ** 2 + (data_formatted.loc[i][1]) ** 2 + (data_formatted.loc[i][2]) ** 2) ** 2
        Entropy = (1 + (data_formatted.loc[i][0] + data_formatted.loc[i][1] + data_formatted.loc[i][2])) ** 2 * log(abs(1 + data_formatted.loc[i][0] + data_formatted.loc[i][1] + (data_formatted.loc[i][2]) ** 2))

        data_combined_hor = pd.concat([pd.Series(data_formatted.loc[i]), data_kurt, data_mad,
                                       data_mean, data_median, data_min, data_max, data_quan_25, data_quan_50,
                                       data_quan_75, data_skew, data_sum, data_std, data_var,
                                       pd.Series(SignalMagnitudeArea), pd.Series(VectorMagnitude), pd.Series(MovementVariation), pd.Series(Energy), pd.Series(Entropy)], axis=0,
                                      ignore_index=True)

        data_combined = data_combined.append(data_combined_hor, ignore_index=True)

    data_combined.columns = ['AccX', 'AccY', 'AccZ', 'AccX_kurt', 'AccY_kurt', 'AccZ_kurt', 'AccX_mad', 'AccY_mad',
                             'AccZ_mad', 'AccX_mean', 'AccY_mean', 'AccZ_mean', 'AccX_median', 'AccY_median', 'AccZ_median',
                             'AccX_min', 'AccY_min', 'AccZ_min', 'AccX_max', 'AccY_max', 'AccZ_max', 'AccX_quan_25',
                             'AccY_quan_25', 'AccZ_quan_25', 'AccX_quan_50', 'AccY_quan_50', 'AccZ_quan_50', 'AccX_quan_75',
                             'AccY_quan_75', 'AccZ_quan_75', 'AccX_skew', 'AccY_skew', 'AccZ_skew', 'AccX_sum', 'AccY_sum',
                             'AccZ_sum', 'AccX_std', 'AccY_std', 'AccZ_std', 'AccX_var', 'AccY_var', 'AccZ_var',
                             'SignalMagnitudeArea', 'VectorMagnitude', 'MovementVariation', 'Energy', 'Entropy']
    y_pred = loaded_model.predict(data_combined)
    data_combined['y_pred'] = y_pred
    data_save_path = ResultPath + "/Predicted_Results.txt"
    data_combined.to_csv(data_save_path)
    y_pred = pd.DataFrame(y_pred)
    y_pred_counts = y_pred.value_counts(sort=True).copy()
    y_pred_counts = y_pred_counts * float(StepSize)
    TimeBudgetPath = ResultPath + '/TimeBudget.txt'
    y_pred_counts.to_csv(TimeBudgetPath)
    y_pred_counts = pd.DataFrame(pd.read_csv(TimeBudgetPath, index_col=0))
    y_pred_counts.plot(kind='bar', legend=None)
    plt.xlabel('Behavior')
    plt.ylabel('Time budget (second)')
    plt.tight_layout()
    figurename = PlotsPath + "/BehaviorResults.png"
    plt.savefig(figurename)
    # plt.show()

    filename = ResultPath + "/Predicted_Results.txt"
    y_pred = pd.DataFrame(pd.read_csv(filename))
    y_pred = y_pred['y_pred']
    Duration_distribution = pd.DataFrame()
    count = 1
    for i in range(1, len(y_pred)):
        if y_pred[i] == y_pred[i - 1]:
            count += 1
        else:
            data = pd.DataFrame([y_pred[i - 1], count]).T
            Duration_distribution = pd.concat([Duration_distribution, data])
            count = 1

    Duration_distribution.columns = ['Class', 'Duration']
    Duration_distribution['Duration'] = Duration_distribution['Duration'] * StepSize
    Duration_distribution_path = ResultPath + '/Duration_distribution.csv'
    Duration_distribution.to_csv(Duration_distribution_path)
    Duration_statistics = Duration_distribution.groupby('Class').agg(
        {'Duration': ['mean', 'min', 'max', 'median', 'std']}).copy()
    Duration_statistics_path = ResultPath + '/Duration_statistics.csv'
    Duration_statistics.to_csv(Duration_statistics_path)

    Behavior_Index = Duration_distribution['Class'].unique()
    Behavior_Index_df = pd.DataFrame()
    for i in Behavior_Index:
        for j in Behavior_Index:
            if i != j:
                data = pd.DataFrame([i, j]).T
                Behavior_Index_df = pd.concat([Behavior_Index_df, data])

    Behavior_Index_df.columns = ['First', 'Second']
    Behavior_Index_df['Frequency'] = 0
    frequency = Behavior_Index_df['Frequency'].tolist()

    for x in range(len(Duration_distribution) - 1):
        for i in range(len(Behavior_Index_df)):
            if Duration_distribution.iloc[x][0] == Behavior_Index_df.iloc[i][0] and Duration_distribution.iloc[x + 1][
                0] == Behavior_Index_df.iloc[i][1]:
                frequency[i] += 1
    Behavior_Index_df['Frequency_acc'] = frequency
    Behavior_Index_df.drop(['Frequency'], axis=1, inplace=True)
    Behavior_Index_df_path = ResultPath + '/Behavior_sequence.csv'
    Behavior_Index_df.to_csv(Behavior_Index_df_path)

    print("Behavior analysis was completed!")



