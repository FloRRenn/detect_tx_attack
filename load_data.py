import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

from typing import Any

# pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', None)

class LoadDataset:
    def __init__(self, object_file : Any, columns : list[str] = None, modify_datatype : bool = False):
        if isinstance(object_file, str):
            self._df = self._read_file(object_file, columns)
        elif isinstance(object_file, pd.DataFrame):
            if columns:
                object_file = object_file[columns]
            self._df = object_file.copy()
            
        if modify_datatype:
            self._df = self._change_datatype(self._df)
            
    @property       
    def DataFrame(self):
        return self._df

    @DataFrame.setter
    def DataFrame(self, new_df):
        self._df = new_df
    
    def __getattr__(self, attr):
        if hasattr(pd.DataFrame, attr):
            return getattr(self._df, attr)
        else:
            raise AttributeError(f"'LoadDataset' object has no attribute '{attr}'")
    
    def __getitem__(self, key : Any):
        return self._df[key]
    
    def __setitem__(self, key, value):
        self._df[key] = value
    
    def get_column(self, columns : list[str], inplace : bool = False):
        if inplace:
            self._df = self._df[columns]
            return self
        return LoadDataset(self._df[columns])
    
    def my_apply(self, columns : list[str], func, inplace : bool = False, *args, **kwargs):
        if inplace:
            for col in columns:
                self._df[col] = self._df[col].apply(func, *args, **kwargs)
            return self
        
        else:
            res = self._df.copy()
            for col in columns:
                res[col] = res[col].apply(func, *args, **kwargs)
            return LoadDataset(res)
    
    ########################################## SPLIT DATASET ####################################################################################
    
    def splitTo_X_Y(self, feature_columns : list[str] = None, target_columns : list[str] = None):
        if target_columns is None:
            raise Exception("Please provide column(s) as label(s)")

        if feature_columns is None:
            feature_columns = list(set(self._df.columns) - set(target_columns))

        X = self._df[feature_columns]
        Y = self._df[target_columns]
        
        return X, Y
        
    def trainTestSplit(self, target_column: str, test_size: float = 0.2, random_state: int = None):
        if target_column not in self._df.columns:
            raise ValueError(f"The specified target column '{target_column}' does not exist in the DataFrame.")

        X = self._df.drop(columns = [target_column])
        y = self._df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

        return X_train, X_test, y_train, y_test
    
    ##################################################################################################################################################
    ##################################################################################################################################################
    
    def encodeCategoricalColumns(self, columns_to_encode: list[str], method = 'label'):
        if method not in ['label', 'one-hot']:
            raise ValueError("Method must be 'label' or 'one-hot'")

        for col in columns_to_encode:
            if col not in self._df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

        if method == 'label':
            for col in columns_to_encode:
                le = LabelEncoder()
                self._df[col] = le.fit_transform(self._df[col])
            print(f"Label encoding applied to columns: {columns_to_encode}")

        elif method == 'one-hot':
            self._df = pd.get_dummies(self._df, columns = columns_to_encode, prefix = columns_to_encode)
            print(f"One-hot encoding applied to columns: {columns_to_encode}")
            
    def scalingFeatures(self, X, scaler = None, scale_type = 'fit_transform'):
        if not scaler:
            scaler = StandardScaler()
        
        if scale_type == 'fit_transform':
            X_scaled = scaler.fit_transform(X)
        elif scale_type == 'transform':
            X_scaled = scaler.transform(X)
        else:
            raise ValueError("Invalid scale_type. Use 'fit_transform' or 'transform'.")

        return X_scaled, scaler
    
    ##############################################################################################################################################
    ########################################## SORT BY COLUMNS ####################################################################################
    
    def sort_by_column(self, column: str, ascending: bool = True, inplace: bool = False):
        sorted_df = self._df.sort_values(by = column, ascending = ascending)

        if inplace:
            self._df = sorted_df
            return self
        else:
            return LoadDataset(sorted_df)
        
    def sort_by_many_columns(self, columns: list[str], ascending: list[bool] = None, inplace: bool = False):
        if ascending is None:
            ascending = [True] * len(columns)

        sorted_df = self._df.sort_values(by = columns, ascending = ascending)

        if inplace:
            self._df = sorted_df
            return self
        else:
            return LoadDataset(sorted_df)
        
    def convert_to_datetime(self, column: str, format_: str = None, inplace: bool = False):
        converted_df = self._df.copy()

        if format:
            converted_df[column] = pd.to_datetime(self._df[column], format = format_)
        else:
            converted_df[column] = pd.to_datetime(self._df[column])

        if inplace:
            self._df = converted_df
        else:
            return LoadDataset(converted_df)
    
    ##################################################################################################################################################
    ########################################## DETECT OUTLIER ####################################################################################
    
    def detect_outliers_isolation_forest(self, column_name : str, contamination : float = 0.05):
        if column_name not in self._df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        model = IsolationForest(contamination = contamination)
        outliers = model.fit_predict(self[[column_name]])
        outliers_mask = outliers == -1
        num_outliers = outliers_mask.sum()
        print(f"Number of outliers in column '{column_name}' (Isolation Forest): {num_outliers}")
        # self_df[column_name][outliers_mask] = np.nan

    def detect_outliers_one_class_svm(self, column_name : str, nu : float = 0.05):
        if column_name not in self._df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        model = OneClassSVM(nu=nu)
        outliers = model.fit_predict(self[[column_name]])
        outliers_mask = outliers == -1
        num_outliers = outliers_mask.sum()
        print(f"Number of outliers in column '{column_name}' (One-Class SVM): {num_outliers}")

    def detect_outliers_elliptic_envelope(self, column_name : str, contamination : float = 0.05):
        if column_name not in self._df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        model = EllipticEnvelope(contamination=contamination)
        outliers = model.fit_predict(self[[column_name]])
        outliers_mask = outliers == -1
        num_outliers = outliers_mask.sum()
        print(f"Number of outliers in column '{column_name}' (Elliptic Envelope): {num_outliers}")
    
    ##############################################################################################################################
    ########################################## PLOT CHART ########################################################################
    
    def distributionPercentOfValueInColumn(self, column : str):
        if column not in self._df.columns:
            raise ValueError(f"The specified column '{column}' does not exist in the DataFrame.")

        distribution = self._df[column].value_counts()
        sorted_distribution = np.argsort(-distribution.values)

        print(f"Distribution of each unique value in column {column.upper()}:")
        for i in sorted_distribution:
            print(f">> Value `{distribution.index[i]}`: {distribution.values[i]} ({np.round((distribution.values[i] / self._df.shape[0] * 100), 3)} %)")
    
    
    def plotChart(self, column: str, 
                        xlabel: str = "X axis", ylabel: str = "Y axis", 
                        title: str = None,
                        figsize: tuple = (15, 10),
                        top_n: int = None,
                        type: str = 'vertical'):
        if column not in self._df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        
        plt.figure(figsize=figsize)
        
        if top_n is not None:
            top_values = self._df[column].value_counts().nlargest(top_n).index
            filtered_df = self._df[self._df[column].isin(top_values)]
            distribution = filtered_df[column].value_counts()
        else:
            distribution = self._df[column].value_counts()
        
        # Use either bar or barh based on the 'type' parameter
        if type == 'vertical':
            ax = distribution.plot(kind='bar')
            
            # Add value counts at the top of each bar
            for i, v in enumerate(distribution):
                ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
                
            ax.set_xticks(range(len(distribution)))
            ax.set_xticklabels(distribution.index, rotation=0, ha='center')  # Adjust rotation and alignment as needed
        elif type == 'horizontal':
            ax = distribution.plot(kind='barh')
            
            # Add value counts at the left of each bar (horizontal alignment set to 'left')
            for i, v in enumerate(distribution):
                ax.text(v + 0.1, i, str(v), ha='left', va='center')
                
            ax.set_yticks(range(len(distribution)))
            ax.set_yticklabels(distribution.index)  # Adjust alignment as needed
        else:
            raise ValueError("Invalid value for 'type'. Use 'vertical' or 'horizontal'.")
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        
        plt.show()

        
    def plotPie(self, column : str, 
                    title : str = "Plot Pie By Column ",
                    figsize : tuple = (15, 10), top_n: int = None):

        if column not in self._df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        plt.figure(figsize=figsize)

        if top_n is not None:
            top_values = self._df[column].value_counts().nlargest(top_n).index
            filtered_df = self._df[self._df[column].isin(top_values)]
            distribution = filtered_df[column].value_counts()
        else:
            distribution = self._df[column].value_counts()

        unique_vals = distribution.index

        if title == "Plot Pie By Column ":
            title += column 
            title += f" - Top {top_n}" if top_n is not None else ""
        
        plt.pie(distribution, labels = unique_vals, startangle = 90,
                pctdistance = 0.8, autopct = '%1.1f%%')
        plt.title(title)
        plt.legend()
        plt.show()
        
    def plotPair(self, columns_to_plot : list[str], hue = None, diag_kind = 'auto', height :float = 2.5):
        numerical_columns = self._df.select_dtypes(include=[np.number])
        columns_to_plot = list(set(columns_to_plot) & set(numerical_columns.columns))

        if not columns_to_plot:
            raise ValueError("No valid numerical columns to plot.")

        if hue is None:
            sns.pairplot(self._df, vars = columns_to_plot, diag_kind = diag_kind, height = height)
        else:
            sns.pairplot(self._df, vars = columns_to_plot, diag_kind = diag_kind, 
                         hue = hue, height = height)

        plt.show()
        
    def plotScatter(self, x_column_name : str, y_column_name : str,
                    xlabel : str = None, ylabel : str = None, 
                    title : str = None, figsize : tuple = (8, 6)):
        if x_column_name not in self._df.columns or y_column_name not in self._df.columns:
            raise ValueError("One or both of the specified columns do not exist in the DataFrame.")
        
        plt.figure(figsize = figsize)
        plt.scatter(self._df[x_column_name], self._df[y_column_name], 
                    label = 'Data Points', alpha = 0.5)
        
        plt.xlabel(xlabel if xlabel else x_column_name)
        plt.ylabel(ylabel if ylabel else y_column_name)
        plt.title(title if title else f'{x_column_name} vs {y_column_name}')

        plt.grid('both')
        plt.show()
        
    def plotLineChart(self, x_column : str, y_column : str, xlabel="X axis", ylabel="Y axis", title="Line Chart", figsize=(15, 10)):
        if x_column not in self._df.columns or y_column not in self._df.columns:
            raise ValueError(f"One or both of the specified columns do not exist in the DataFrame.")

        plt.figure(figsize=figsize)
        plt.plot(self._df[x_column], self._df[y_column], marker='o', linestyle='-', color='b', label='Line')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plotBox(self, column : str, title = None, figsize = (8, 6)):
        if column not in self._df.columns:
            raise ValueError(f"The specified column '{column}' does not exist in the DataFrame.")

        plt.figure(figsize = figsize)
        plt.boxplot(self._df[column], vert=False)
        plt.title(title if title else column)
        plt.xlabel(column)
        plt.grid()
        plt.show()
        
    def histogram(self, column : str, bins : int = 10, 
                  xlabel : str = None, ylabel : str = None, 
                  title : str = None, figsize : tuple = (8, 6)):
        if column not in self._df.columns:
            raise ValueError(f"The specified column '{column}' does not exist in the DataFrame.")

        plt.figure(figsize = figsize)
        plt.hist(self._df[column], bins = bins, alpha = 0.75)
        plt.title(title if title else f'Histogram of {column}')
        plt.xlabel(xlabel if xlabel else column)
        plt.ylabel(ylabel if ylabel else 'Frequency')
        plt.grid()
        plt.show()
        
    def correlationMatrix(self, figsize : tuple = (20, 20), columns : list[str] = None,
                          cmap : str = "coolwarm"):
        if columns is None:
            corr = self._df.corr()
        else:
            corr = self._df[columns].corr()
            
        sns.set(rc = {'figure.figsize': figsize})
        sns.heatmap(corr, cmap = cmap, annot = True, fmt = ".2f")
        
    ##############################################################################################################################
    ##############################################################################################################################
        
    def _change_datatype(self, df : pd.DataFrame):
        int_types = [np.int8, np.int16, np.int32]
        float_types = [np.float16, np.float32]

        for col in df.columns:
            col_dtype = df[col].dtype
            
            if col_dtype == np.int64 or col_dtype == np.int32:
                for int_type in int_types:
                    if np.iinfo(int_type).min <= df[col].min() and df[col].max() <= np.iinfo(int_type).max:
                        df[col] = df[col].astype(int_type)
                        break
                    
            elif col_dtype == np.float64:
                for float_type in float_types:
                    if np.finfo(float_type).min <= df[col].min() and df[col].max() <= np.finfo(float_type).max:
                        df[col] = df[col].astype(float_type)
                        break
        return df
    
    def _read_file(self, file_name : str, columns : list[str]):
        file_extension = file_name.split('.')[-1].lower()
        func = None
        
        if file_extension == 'csv':
            func = pd.read_csv
        elif file_extension in ['xls', 'xlsx']:
            func = pd.read_excel
        elif file_extension == 'json':
            func = pd.read_json
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return func(file_name, low_memory = True, usecols = columns)