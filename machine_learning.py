#-*- coding:utf-8 -*-

import time
import os
import os.path
import re
import sys
import datetime

from tkinter import Tk
from tkinter import Frame
from tkinter import ttk  
from tkinter import scrolledtext  
from tkinter import Menu  
from tkinter import Spinbox 
from tkinter import messagebox
from tkinter import filedialog
from tkinter import BooleanVar
from tkinter import Checkbutton
from tkinter import StringVar
from tkinter import Button
from tkinter import Label
from tkinter import Entry
from tkinter import IntVar
from tkinter.simpledialog import askstring, askinteger, askfloat

from pandas import read_excel
from pandas import read_csv
from pandas import ExcelFile
from pandas import ExcelWriter
from pandas import DataFrame
from pandas import Series

from numpy.random import randint
from numpy import around 
from numpy import nan

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 

from sklearn import preprocessing 
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

from collections import Mapping
from collections import Counter
from itertools import product
from copy import deepcopy


def log(*args, **kwargs):
    ''' 运行记录 '''
    localtime = time.asctime(time.localtime(time.time()))
    print('log:', localtime, '\n', *args, **kwargs)


def read(file_address, sheet_name=0, sep=','):
    directory_name, file_name = os.path.split(file_address)
    fileName, suffix = os.path.splitext(file_name)
    if suffix == '.csv':
        df = read_csv(file_address,
            sep = sep,
            header = 0,
            encoding = 'gbk',
            engine = 'python',
            )
    elif suffix == '.xlsx':
        df = read_excel(file_address,
            sheetname = sheet_name,
            index = None,
            header = 0, 
            encoding = 'utf-8',
            )
    elif suffix == '.xls':
        xls_file = ExcelFile(file_address)
        df = read_excel(xls_file,
            sheetname = sheet_name,
            index = None,
            header = 0,
            encoding = 'utf-8',
            )
    else:
        raise TypeError('文件类型错误，请输入xls/xlsx/csv文件')
    log(df.head())

    return df

def objFloat(obj):
	try:
		return float(obj)
	except:
		return nan

def dropna(data):
	data = data.applymap(objFloat)
	data = data.dropna(axis=1, how='all', thresh=None, subset=None, inplace=False)
	data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

	return data

def writeFile(df, file_address, sheet_name=0, sep=','):
    directory_name, fileName = os.path.split(file_address)
    name, suffix = os.path.splitext(fileName)
    if suffix == '.csv':
        df.to_csv(file_address,
            sep = sep,
            header = 1,
            encoding = 'gbk',
            index = None,
            )
        log('保存完成')
        mess = fileName + '保存完成'
        messagebox.showinfo('提示', mess)

    elif suffix == '.xlsx':
        writer = ExcelWriter(file_address)
        df.to_excel(writer,
            index = None,
            header = 1, 
            encoding = 'utf-8',
            )
        writer.save()
        log('保存完成')
        mess = fileName + '保存完成'
        messagebox.showinfo('提示', mess)

    else:
       messagebox.showwarning("警告",'警告：文件类型无法保存，请输入xlsx/csv文件！')

    return 


class ParameterGrid(object):

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
            self.param_grid = param_grid

    def __iter__(self):
        for p in self.param_grid:
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params


class Data(object):

    def __init__(self, df):
        df = dropna(df)
        self.columns = df.columns.values.tolist()
        self.label = df.pop(self.columns[len(self.columns) - 1])
        self.values = df.values

        self.stand()
        self.data_split()

    def stand(self):
        min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
        self.scaler_X = min_max_scaler.fit_transform(self.values)

    def data_split(self):
        self.train_X, self.test_X, self.train_y, self.test_y = \
        train_test_split(self.scaler_X,
                        self.label,
                        test_size = 0.3,
                        random_state = 0)


class PredictData(object):
    def __init__(self, df):
        df = dropna(df)
        self.columns = df.columns.values.tolist()
        self.values = df.values
        self.stand()

    def stand(self):
        min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
        self.scaler_X = min_max_scaler.fit_transform(self.values)


class Rfc(Data):
    def __init__(self, df, predict_df):
        super().__init__(df)
        self.data_split()
        self.predict_df = predict_df
        self.parameters = { 'n_estimators': [100, 500, 1000],
                            'criterion': ['entropy'],
                            'min_samples_leaf':[10, 20, 30, 40], 
                            'max_features': [None],
                          }
        self.predict()

    def learnModel(self, 
                n_estimators=10, 
                criterion='gini', 
                min_samples_leaf=1, 
                max_features='auto'):
        random_forest = RandomForestClassifier(n_estimators = n_estimators,
                                            criterion = criterion, 
                                            min_samples_leaf = min_samples_leaf,
                                            max_features = max_features)
        random_forest.fit(self.train_X, self.train_y)
        accuracy = random_forest.score(self.test_X, self.test_y)
        return [n_estimators, criterion, min_samples_leaf, max_features, accuracy]

    def selectpar(self):
        df_record_list = DataFrame(columns=[
                                            'n_estimators',
                                            'criterion',
                                            'min_samples_leaf', 
                                            'max_features', 
                                            'accuracy',
                                            ])
        for par in ParameterGrid(self.parameters):
            record = self.learnModel(n_estimators=par['n_estimators'],
                                criterion=par['criterion'],
                                min_samples_leaf=par['min_samples_leaf'],
                                max_features=par['max_features'])
            df_record_list.loc[df_record_list.shape[0]] = record
        newdf = df_record_list.sort_values(by='accuracy', ascending=False)
        par_dict = dict(zip(newdf.columns.values.tolist(), newdf.iloc[0]))
        return par_dict

    def predict(self):
        par_dict = self.selectpar()
        random_forest = RandomForestClassifier(n_estimators = par_dict['n_estimators'],
                                            criterion = par_dict['criterion'], 
                                            min_samples_leaf = par_dict['min_samples_leaf'],
                                            max_features = par_dict['max_features'])
        # random_forest.fit(self.scaler_X, self.label)
        # random_forest = GridSearchCV(
        #     RandomForestClassifier(
        #         n_estimators=10,
        #         criterion='gini',
        #         min_samples_leaf=1,
        #         max_features='auto'
        #         ),
        #     param_grid = self.parameters,
        #     cv = 2,
        #     )
        random_forest = random_forest.fit(self.scaler_X, self.label)

        data = PredictData(self.predict_df)
        self.label_y = random_forest.predict(data.scaler_X)


class Svc(Data):
    def __init__(self, df, predict_df):
        super().__init__(df)
        self.data_split()
        self.predict_df = predict_df
        #max_features = randint(df.shape[1], size=(3))
        self.parameters = {'C': [0.5, 0.1, 0.5, 1.0], #
                    'kernel': ["linear", "poly", "rbf", "sigmoid",], #"linear", "poly", "rbf", "sigmoid", "precomputed"
                    'degree': [3, 2],
                    'gamma': ['auto']} #"poly", "rbf", "sigmoid"

        self.predict()

    def learnModel(self, 
                C = 0.1, 
                kernel = 'poly', 
                degree = 3, 
                gamma = 'auto'):
        svc_clf = SVC(C = C,
                    kernel = kernel, 
                    degree = degree,
                    gamma = gamma)
        log('self.train_X:', self.train_X)
        log('self.train_y:', self.train_y)

        svc_clf.fit(self.train_X, self.train_y)
        accuracy = svc_clf.score(self.test_X, self.test_y)
        return [C, kernel, degree, gamma, accuracy]

    def selectpar(self):
        df_record_list = DataFrame(columns=[
                                            'C',
                                            'kernel',
                                            'degree', 
                                            'gamma', 
                                            'accuracy',
                                            ])
        for par in ParameterGrid(self.parameters):
            record = self.learnModel(C=par['C'],
                                kernel=par['kernel'],
                                degree=par['degree'],
                                gamma=par['gamma'])
            df_record_list.loc[df_record_list.shape[0]] = record
        newdf = df_record_list.sort_values(by='accuracy', ascending=False)
        par_dict = dict(zip(newdf.columns.values.tolist(), newdf.iloc[0]))

        return par_dict

    def predict(self):
        svc_clf = GridSearchCV(
            SVC(
                C = 0.1,
                kernel = 'poly',
                degree = 3,
                gamma = 'auto',
                class_weight='balanced',
                ),
            param_grid = self.parameters,
            )
        svc_clf = svc_clf.fit(self.scaler_X, self.label)

        # par_dict = self.selectpar()
        # log('predict,par_dict', par_dict)
        # svc_clf = SVC(C = par_dict['C'],
        #                     kernel = par_dict['kernel'], 
        #                     degree = par_dict['degree'],
        #                     gamma = par_dict['gamma'])
        # svc_clf.fit(self.scaler_X, self.label)

        data = PredictData(self.predict_df)
        self.label_y = svc_clf.predict(data.scaler_X)
        log('label_y',self.label_y)

class FeatureWeight(Data):

    def __init__(self, df):
        super().__init__(df)
        self.parameters = { 'n_estimators': [100, 500, 1000],
                            'criterion': ['entropy'],
                            'min_samples_leaf':[10, 20, 30, 40], 
                            'max_features': [None],
                          }

    def learnModel(self, 
                n_estimators=10, 
                criterion='gini', 
                min_samples_leaf=1, 
                max_features='auto'):
        random_forest = RandomForestClassifier(n_estimators = n_estimators,
                                            criterion = criterion, 
                                            min_samples_leaf = min_samples_leaf,
                                            max_features = max_features)
        random_forest.fit(self.train_X, self.train_y)
        accuracy = random_forest.score(self.test_X, self.test_y)
        return [n_estimators, criterion, min_samples_leaf, max_features, accuracy]

    def selectpar(self):
        df_record_list = DataFrame(columns=[
                                            'n_estimators',
                                            'criterion',
                                            'min_samples_leaf', 
                                            'max_features', 
                                            'accuracy',
                                            ])
        for par in ParameterGrid(self.parameters):
            record = self.learnModel(
                n_estimators = par['n_estimators'],
                criterion = par['criterion'],
                min_samples_leaf = par['min_samples_leaf'],
                max_features = par['max_features'],
                )
            df_record_list.loc[df_record_list.shape[0]] = record
        newdf = df_record_list.sort_values(by='accuracy', ascending=False)
        par_dict = dict(zip(newdf.columns.values.tolist(), newdf.iloc[0]))
        return par_dict

    def featureWeight(self):
        par_dict = self.selectpar()
        random_forest = RandomForestClassifier(n_estimators = par_dict['n_estimators'],
                                            criterion = par_dict['criterion'], 
                                            min_samples_leaf = par_dict['min_samples_leaf'],
                                            max_features = par_dict['max_features'])

        random_forest = random_forest.fit(self.scaler_X, self.label)
        self.clf_weight = random_forest.feature_importances_

    def weightPlt(self):
        self.featureWeight()
        log(self.columns, self.clf_weight)
        fig = plt.figure(figsize=(5, 7))
        left, bottom, width, height = 0.12, 0.3, 0.8, 0.6
        fig.add_axes([left,bottom,width,height])
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("机器学习特征权重柱形图")
        plt.xlabel("权重占比")
        plt.yticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
        plt.ylabel("特征名称")
        plt.xticks(rotation= 270)
        font = {
            'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
        plt.bar(self.columns[:-1], self.clf_weight)
        plt.ion()
        try:
            plt.pause(10)
        except:
            plt.close()
        
        return self.clf_weight

    def pearsonPlt(self):
        label = Series(self.label)
        df = DataFrame(self.values)
        corrLs = []
        for c in range(df.shape[1]):
            corr = label.corr(df[c], method='pearson', min_periods=None)
            corrLs.append(abs(corr))
        fig = plt.figure(figsize=(6, 8))
        left, bottom, width, height = 0.12, 0.3, 0.8, 0.6
        fig.add_axes([left,bottom,width,height])
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("皮尔逊相关系数柱形图")
        plt.xlabel("特征名称")
        plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
        plt.ylabel("相关系数")
        plt.xticks(rotation= 270)
        font = {
            'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
        plt.bar(self.columns[:-1], corrLs, facecolor = 'blue', edgecolor = 'white')
        plt.ion()
        try:
            plt.pause(10)
        except:
            plt.close()

        return corrLs


class KmearnsClustering(PredictData):
    def __init__(self, df):
        super().__init__(df)
        self.predict()

    def predict(self):
        self.y_pred = KMeans(n_clusters=8, random_state=None).fit_predict(self.values)
            

class IndexFrame(Frame):
    '''首页 + 控制页面'''
    def __init__(self, master=None):

        Frame.__init__(self, master)
        self.root = master
        self.itemName = StringVar()
        self.createPage()

    def createPage(self):
        #Label(self, text='首页').grid(row=0,column=0)
        classiFication = Button(self, text='分类算法', width=30, pady=7, command=self.classiFication_frame)
        classiFication.grid(row=0, column=0, pady=6)

        clustering = Button(self, text='聚类算法', width=30, pady=7, command=self.clustering_frame)
        clustering.grid(row=1, column=0, pady=6)

        feature = Button(self, text='特征分析', width=30, pady=7, command=self.feature_frame)
        feature.grid(row=2,column=0, pady=6)

    def classiFication_frame(self):
        self.destroy()
        mainpage = MainPage(root)
        return mainpage.classiFicationData()

    def clustering_frame(self):
        self.destroy()
        mainpage = MainPage(root)
        return mainpage.clusteringData()

    def feature_frame(self):
        self.destroy()
        mainpage = MainPage(root)
        return mainpage.featureData()


class ClassiFicationFrame(Frame):
    '''分类算法'''
    def __init__(self, master=None):  
        Frame.__init__(self, master)  
        self.root = master  
        self.createPage() 
  
    def createPage(self):
        Label(self, text='分类算法').grid(row=0,column=0)
        #设置窗口背景
        style = ttk.Style() 
        style.configure("BW.TLabel", font=("微软雅黑", "12"))  #默认字体

        self.learn_pathdb = StringVar()
        Entry(self, textvariable = self.learn_pathdb, width=50).grid(row=1, column=0)
        Button(self, text='导入学习文件', command=self.selectLearningPath).grid(row=1,column=1)

        self.predict_pathdb = StringVar()
        Entry(self, textvariable = self.predict_pathdb, width=50).grid(row=2, column=0)
        Button(self, text='导入预测文件', command=self.selectPreDictPath).grid(row=2,column=1)

        Button(self, text='运行', width=10, command=self.run).grid(row=3,column=1)

        algo_frame = Frame(self)
        algo_frame.grid(row=3)

        self.algorithm1 = BooleanVar()
        Checkbutton1 = Checkbutton(algo_frame, text="支持向量机", variable= self.algorithm1, command=self.func)
        Checkbutton1.grid(row=0, column= 0, sticky='W')

        self.algorithm2 = BooleanVar()
        Checkbutton2 = Checkbutton(algo_frame, text="随机森林", variable= self.algorithm2, command=self.func)
        Checkbutton2.grid(row=0, column = 1, sticky='W')
        
    def selectLearningPath(self):
        path = filedialog.askopenfilename()
        self.learn_pathdb.set(path)
        self.fitPath = path 

    def selectPreDictPath(self):
        path = filedialog.askopenfilename()
        self.predict_pathdb.set(path)
        self.predictPath = path 

    def callCheckbutton(self):
        print(self.variables)

    def func(self):
        get_ls = [self.algorithm1.get(), self.algorithm2.get()]
        checks = ['支持向量机', '随机森林']
        new_ls = []
        for index, get in enumerate(get_ls):
            if get:
                new_ls.append(checks[index])
        self.new_ls = new_ls
  
    def run(self):
        messagebox.showinfo('提示', '点击确定开始运行\n因为调整参数您可能要稍等几分钟才能得到结果')
        try:
            print(self.new_ls)
        except:
            messagebox.showwarning("警告",'警告：您没有选取算法！')

        try:
            fitDf = read(self.fitPath)
            predictDf = read(self.predictPath)
            resultDf = deepcopy(predictDf)
        except AttributeError:
            messagebox.showwarning("警告",'警告：你没有导入文件！')

        if '随机森林' in self.new_ls:
            try:
                rfc = Rfc(fitDf, predictDf)
                series_label = Series(rfc.label_y)
                log('series_label:',series_label)
                resultDf['随机森林预测结果'] = series_label
            except:
                messagebox.showwarning('警告','分类算法时label为整数')
                raise TypeError('数据错误程序挂起')

        if '支持向量机' in self.new_ls:
            try:
                svc_clf = Svc(fitDf, predictDf)
                series_label = Series(svc_clf.label_y)
                log('series_label:',series_label)
                resultDf['支持向量机预测结果'] = series_label
            except:
                messagebox.showwarning('警告','分类算法时label为整数')
                raise TypeError('数据错误程序挂起')
            
        #res = askstring("文字输入", "请输入你要写入的文件名")
        save_path = filedialog.asksaveasfilename()
        log(save_path)
        writeFile(resultDf, save_path)


class ClusteringFrame(Frame):
    '''聚类算法'''
    def __init__(self, master=None):  
        Frame.__init__(self, master)  
        self.root = master  
        self.createPage() 
  
    def createPage(self):
        Label(self, text='聚类算法').grid(row=0,column=0)
        #设置窗口背景
        style = ttk.Style() 
        style.configure("BW.TLabel", font=("微软雅黑", "12"))  #默认字体

        self.learn_pathdb = StringVar()
        Entry(self, textvariable = self.learn_pathdb, width=50).grid(row=1, column=0)
        Button(self, text='导入学习文件', command=self.selectLearningPath).grid(row=1,column=1)
        Button(self, text='运行', width=20, command=self.run).grid(row=2, column=1, columnspan=2)

    def selectLearningPath(self):
        path = filedialog.askopenfilename()
        self.learn_pathdb.set(path)
        self.filePath = path

    def run(self):
        fitDf = read(self.filePath)
        resultDf = deepcopy(fitDf)
        kmearns = KmearnsClustering(fitDf)
        resultDf['类别'] = kmearns.y_pred

        save_path = filedialog.asksaveasfilename()
        log(save_path)
        writeFile(resultDf, save_path)


class FeatureFrame(Frame):
    '''特征分析'''
    def __init__(self, master=None):
        Frame.__init__(self, master)  
        self.root = master  
        self.createPage() 
  
    def createPage(self):
        Label(self, text='特征分析').grid(row=0,column=0)
        #设置窗口背景
        style = ttk.Style() 
        style.configure("BW.TLabel", font=("微软雅黑", "12"))  #默认字体

        self.learn_pathdb = StringVar()
        Entry(self, textvariable = self.learn_pathdb, width=50).grid(row=1, column=0)
        Button(self, text='导入学习文件', command=self.selectLearningPath).grid(row=1,column=1)
        Button(self, text='运行', width=10, command=self.run).grid(row=3,column=1)

        algo_frame = Frame(self)
        algo_frame.grid(row=3)

        self.weight_check = BooleanVar()
        Checkbutton1 = Checkbutton(algo_frame, text="特征权重", variable=self.weight_check, command=self.func)
        Checkbutton1.grid(row=0, column= 0,sticky='W')
        self.per_check = BooleanVar()
        Checkbutton2 = Checkbutton(algo_frame, text="相关系数", variable=self.per_check, command=self.func)
        Checkbutton2.grid(row=0,column = 1, sticky='W')
   
    def selectLearningPath(self):
        path = filedialog.askopenfilename()
        self.learn_pathdb.set(path)
        self.filePath = path
    
    def func(self):
        get_ls = [self.weight_check.get(), self.per_check.get()]
        checks = ['特征权重', '相关系数']
        new_ls = []
        for index, get in enumerate(get_ls):
            if get:
                new_ls.append(checks[index])
        self.new_ls = new_ls

    def run(self):
        try:
            print(self.new_ls)
        except:
            messagebox.showwarning("警告",'警告：您没有选取算法！')
        try:
            df = read(self.filePath)
            fw = FeatureWeight(df)
            resultDf = DataFrame({'特征':fw.columns[:-1]})
            log(resultDf)
        except AttributeError:
            messagebox.showwarning("警告",'警告：你没有导入文件！')

        if '特征权重' in self.new_ls:
            try:
                weight = fw.weightPlt()
                resultDf['特征权重'] = weight
                log(resultDf)            
            except:
                messagebox.showwarning('警告','运用分类算法时label必须为整数')
                raise TypeError('数据错误程序挂起')

        if '相关系数' in self.new_ls:
            corrLs = fw.pearsonPlt()
            resultDf['相关系数'] = corrLs
            log(resultDf)
        log('look')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        save_path = filedialog.asksaveasfilename()
        log(save_path)
        writeFile(resultDf, save_path)


#---------------------------------------------------------------
#控制台
#菜单栏对应的各个子页面 
class MainPage(object):
    def __init__(self, master=None):  
        self.root = master 
        self.root.geometry('700x500')
        self.root.iconbitmap('hw.ico')
        self.createPage() 
    def createPage(self):
        '''页面'''
        self.indexPage = IndexFrame(self.root)
        self.classiFicationPage = ClassiFicationFrame(self.root)
        self.clusteringPage = ClusteringFrame(self.root)
        self.featurePage = FeatureFrame(self.root)

        #默认显示首页界面
        self.indexPage.pack()
        #菜单
        menubar = Menu(self.root)
        menubar.add_command(label='首页', command = self.indexData)
        menubar.add_command(label='分类算法', command = self.classiFicationData)
        menubar.add_command(label='聚类算法', command = self.clusteringData)
        menubar.add_command(label='特征分析', command = self.featureData)
   
        menubar.add_command(label='退出', command = self.root.quit)
        self.root['menu'] = menubar
    
    def indexData(self):
        self.indexPage.pack()
        self.classiFicationPage.pack_forget()
        self.clusteringPage.pack_forget()
        self.featurePage.pack_forget()
 
    def classiFicationData(self):
        self.indexPage.pack_forget() 
        self.classiFicationPage.pack()
        self.clusteringPage.pack_forget()
        self.featurePage.pack_forget()

    def clusteringData(self):
        self.indexPage.pack_forget() 
        self.classiFicationPage.pack_forget() 
        self.clusteringPage.pack()
        self.featurePage.pack_forget()

    def featureData(self):
        self.indexPage.pack_forget() 
        self.classiFicationPage.pack_forget() 
        self.clusteringPage.pack_forget()
        self.featurePage.pack()


if __name__ == '__main__':
    root = Tk()
    root.title("机器学习系统")
    root.resizable(False, False) #不允许改变窗口大小
    style = ttk.Style() #设置窗口背景 
    style.configure("BW.TLabel", font=("微软雅黑", "12"))  #默认字体 
    MainPage(root)
    root.mainloop()
