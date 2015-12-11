#! /usr/bin/python
import sys,getopt,subprocess,re,math,commands,time,copy,random
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pylab as Py
from operator import itemgetter, attrgetter, methodcaller

from sklearn import datasets, linear_model , cross_validation, metrics, tree, svm
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier

from pylab import savefig
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import mixture

GlobalNumModels=11
GlobalNumClassModels=5
GlobalNumClusteringModels=6
GlobalNumClusters=10

ChoseRegModel=8
ChoseClassModel=4
ChoseClusterModel=0
GlobalNumFolds=5

GLNecessaryCombisNotDone=True
GLLeaveOneOut=False
#GLLeaveOneOut=True 
GLCheckAllModels=False 
GLCheckAllModels= True
GLModelDiagnostics=False 

"""
Regression:

0    LinearRegression
1    RidgeRegression
2    Lasso
3    ElasticNet
4    LassoLars
5    DecisionTreeRegressor
6    SVMRegressor
7    SGDRegressor
8    RandomForestRegressor
9    GradientBoostingRegressor
10    AdaBoostRegressor

Classification:

0    LogisticRegression
1    SVMClassification
2    SGDClassifier
3    DecisionTreeClassifier
4    GradientBoostingClassifier
5    RandomForestClassifier

Clustering 
0    KMeans
1    AffinityPropogation
2    MeanShift
3    Spectral Clustering
//4    Ward Hierarchial Clustering
4    Agglomerating Clustering
5    DBSCAN
6    Gaussian mixtures
"""

def usage():
    print "\t Usage: Learn.py -x <Training file name> -r <0:Regression 1:Classification 2: Clustering>\n\t Optional: \n\t\t -o <Output file name> \n\t\t -y <Test-file-name> \n\t\t -c <correlation-flag> 0: No correlaiton 1: IpStats correlation"
    sys.exit()

def MapColVals(df,colArray=[]):
    for colName in colArray:
        if( not(colName in df.columns) ):
            print "\t Column: %s is not found in data-frame "
            return
        uniqVals = df[colName].unique()
        matchVals = {}
        for Idx,currMatch in enumerate(uniqVals):
            matchVals[currMatch] = Idx
        df[colName] = df[colName].map(matchVals)
    return df

def shuffle(df):
    #credits: Jerome Zhao : http://stackoverflow.com/questions/15772009/shuffling-permutation-a-dataframe-in-pandas
    index = list(df.index)
    random.shuffle(index)
    df = df.ix[index]
    df.reset_index()
    return df
    
def Normalize(x,y,z):
    return (float(x)+float(y))/float(z)
    
def IsNumber(s):
    # Credits: StackExchange: DanielGoldberg: http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-in-python
    try:
        float(s)
        return True
    except ValueError:
        return False

def RestofCols(ListOfCols,Column):
    return filter(lambda x: x!=Column, ListOfCols)

def ChoseLearnModel(UseModel,RegressionFlag=[],NumModels=GlobalNumModels):
    
    if(RegressionFlag[0]):    
        if(UseModel==0):
            LearnModel= linear_model.LinearRegression()
            print " \t Linear regression!! "
            Method='LinearRegression'
        elif(UseModel==1):
            alpha=0.5
            LearnModel = linear_model.RidgeCV(alphas=[0.1,0.15,0.2,0.25])#,0.01,0.7,0.75,0.80,0.90,0.93])
            print " \t Ridge-regression-- alpha: "+str(alpha)
            Method='RidgeRegression'
        elif(UseModel==2):
            alpha=0.01
            LearnModel= linear_model.LassoCV(cv=10) #(alphas=[0.0001,0.001,0.01,0.04,]) #Lasso(alpha=alpha) #alpha=alpha)
            print " \t Lasso-- alpha: "+str(alpha)
            Method='Lasso'
        elif(UseModel==3):
            alpha=0.001
            LearnModel= linear_model.ElasticNet(alpha=alpha) #alpha=alpha)
            print " \t ElasticNet-- alpha: "+str(alpha)
            Method='ElasticNet'
        elif(UseModel==4):
            alpha=0.4
            LearnModel= linear_model.LassoLars(alpha=alpha)#,normalize=False) #alpha=alpha)
            print " \t LassoLars-- alpha: "+str(alpha)
            Method='LassoLars'
        elif(UseModel==5):
            Method='DecisionTreeRegressor'
            LearnModel=tree.DecisionTreeRegressor(max_depth=3)
            print "\t Decision tree regressor!! "
        elif(UseModel==6):
            Method='SVMRegressor'
            LearnModel=svm.SVR() #kernel='rbf', C=1e3, gamma=0.05)
            print "\t SVM regressor!! "
        elif(UseModel==7):
            Method='SGDRegressor'
            LearnModel=SGDRegressor(loss="huber") #"epsilon_insensitive")
            print "\t SGD regressor!! "
        elif(UseModel==8):
            Method='RandomForestRegressor'
            LearnModel=RandomForestRegressor(n_estimators=10) # (loss="huber") #"epsilon_insensitive")
            print "\t Random Forest Regressor!! "
        elif(UseModel==9):
            Method='GradientBoostingRegressor'
            LearnModel=GradientBoostingRegressor(n_estimators=5) # (loss="huber") #"epsilon_insensitive")
            print "\t Gradient Boosting Regressor!! "
        elif(UseModel==10):    
            Method='AdaBoostRegressor'
            rng = np.random.RandomState(1)
            LearnModel=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=100, random_state=rng) # (loss="huber") #"epsilon_insensitive")
            print "\t AdaBoost Regressor!! "
    elif(RegressionFlag[1]):
        if(UseModel==0):
            LearnModel= linear_model.LogisticRegression(penalty='l1',tol=0.1)
            print " \t Classification: LogisticRegression !! "
            Method='LogisticRegression'
        elif(UseModel==1):
            alpha=0.5
            LearnModel = svm.SVC() 
            print " \t SVMClassification "
            Method='SVMClassification'
        elif(UseModel==2):
            alpha=0.01
            LearnModel= SGDClassifier(loss="hinge", penalty="l2")
            print " \t SGDClassifier "
            Method='SGDClassifier'
        elif(UseModel==3):
            alpha=0.001
            LearnModel= tree.DecisionTreeClassifier()
            print " \t DecisionTreeClassifier "
            Method='DecisionTreeClassifier'
        elif(UseModel==4):
            alpha=0.4
            LearnModel= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
            print " \t GradientBoostingClassifier "
            Method='GradientBoostingClassifier'
        elif(UseModel==5):
            LearnModel = RandomForestClassifier()
            print "\t RandomForestClassifier "
            Method = 'RandomForestClassifier'
    elif(RegressionFlag[2]):
        if(UseModel==0):
            #LearnModel= KMeans(init='k-means++',n_clusters= GlobalNumClusters)
            LearnModel= KMeans(init='k-means++',n_clusters= GlobalNumClusters)
            print "\t Kmeans-Clustering "
            Method='KMeans-Clustering'
        elif(UseModel==1):
            LearnModel=AffinityPropagation(preference=-25)
            print "\t AffinityPropogation "
            Method='AffinityPropogation'
        elif(UseModel==2):
            LearnModel=MeanShift(bin_seeding=True)    
            print "\t MeanShift "
            Method='MeanShift'
        elif(UseModel==4):
            LearnModel=AgglomerativeClustering(n_clusters=GlobalNumClusters,affinity='euclidean')
            print "\t AgglomerativeClustering "
            Method='AgglomerativeClustering'
        elif(UseModel==5):
            LearnModel=DBSCAN(eps=0.3)
            print "\t DBSCAN "
            Method='DBSCAN'
        elif(UseModel==6):
            LearnModel=mixture.GMM(    n_components=GlobalNumClusters,covariance_type='spherical')
            print "\t GMM mixture"
            Method='GMM_Mixture'
        elif(UseModel==3):
            LearnModel= KMeans(init='k-means++',n_clusters= GlobalNumClusters)
            #LearnModel=spectral_clustering(X,n_clusters=GlobalNumClusters,eigen_solver='arpack')
            print "\t AgainKMeans" 
            Method='AgainKMeans'    
                
        """elif(UseModel==3):
            LearnModel=spectral_clustering(X,n_clusters=GlobalNumClusters,eigen_solver='arpack')
            print "\t SpectralClustering" 
            Method='SpectralClustering' """            

    return (LearnModel,Method)    
       
def CrossValidation(IpStats,XParams,YParams,RegressionFlag,UseModel=0,NumFolds=GlobalNumFolds,NumModels=GlobalNumModels):

 AverageErrors={}
 AverageR2={}
 NecessaryCombisNotDone=GLNecessaryCombisNotDone
 LeaveOneOut=GLLeaveOneOut
 CheckAllModels=GLCheckAllModels
 ModelDiagnostics=GLModelDiagnostics
 
 if(RegressionFlag[0]):
     NumModels=GlobalNumModels
 elif(RegressionFlag[1]):
     NumModels=GlobalNumClassModels
 elif(RegressionFlag[2]):
     NumModels=GlobalNumClusteringModels
         
 for Idx,CurrCol in enumerate(XParams):    
  if(NecessaryCombisNotDone): 
    AverageErrors[CurrCol]={}
    AverageR2[CurrCol]={}
    
    RestOfColumns=XParams
    if(LeaveOneOut):
        print "\t Calling RestOfColumns "
        RestOfColumns=RestofCols(XParams,CurrCol)
        TmpX=IpStats[RestOfColumns]
        if(Idx==(len(XParams)-1)):
            NecessaryCombisNotDone=False    
    else:
        TmpX=IpStats[XParams]
        NecessaryCombisNotDone=False
    TmpY=IpStats[YParams]
    print "\n\t Param: "+str(CurrCol)+"\t Rest of Columns: "+str(TmpX.columns)
    
    X=TmpX.as_matrix() 
    Y=TmpY.as_matrix() 

    IpStatsLen=len(IpStats)
    kf = cross_validation.KFold(IpStatsLen, n_folds=NumFolds)
    print "\t X-columns: "+str(TmpX.columns)
    print "\t X shape: "+str(X.shape)+" Y shape "+str(Y.shape)+"\t Len(K-folds): "+str(len(kf))
    LearnModel= linear_model.LinearRegression()
    alpha=0.0001
 
    if(CheckAllModels==False):
        NumModels = 1
    for Idx in range(NumModels):
     Idx=NumModels-1
     AllModelsNotChecked=True
     for Idx in range(NumModels):
      if(CheckAllModels):
         print "\n\t CurrentModel index "+str(Idx)+"\t NumModels "+str(NumModels)
      else:
         Idx=UseModel
      if(AllModelsNotChecked):
        (LearnModel,Method)=ChoseLearnModel(Idx,RegressionFlag) #=True)         
        AvgRelativeErrorAcrossFolds=0.0    
        AvgRelativeR2AcrossFolds=0.0
        if ModelDiagnostics:
            ModelDiagnosticsOutput=open('ModelDiagnosticsOutput.dat','w')
            ModelDiagnosticsOutput.write("\tIndex\tOutputY\tTestY\tAbsError\n\n")
            FoldIdx=-1
        for train_index, test_index in kf:
            
            if(RegressionFlag[2]):
                TrainX, TestX = X[train_index], X[test_index]
                LearnModel.fit(TrainX) 
                OutputY=LearnModel.fit_predict(TestX)    
                ExplainedVarianceR2=0 
                AvgRelativeR2AcrossFolds+=(ExplainedVarianceR2)
                AvgRelativeErrorAcrossFolds+=0;
            else:
                #print("TRAIN:", train_index, "TEST:", test_index)
                TrainX, TestX = X[train_index], X[test_index]
                TrainY, TestY = Y[train_index], Y[test_index]
                LearnModel.fit(TrainX,TrainY.ravel())
                OutputY=LearnModel.predict(TestX)
                                
                if ModelDiagnostics:
                    FoldIdx+=1
                    FoldIdxScaled=int(FoldIdx*IpStatsLen/NumFolds)
                    for i in range(len(TestY)):
                        AbsError=abs(TestY[i][0]-OutputY[i])/TestY[i]
                        ModelDiagnosticsOutput.write("\t"+str(FoldIdxScaled+i)+"\t"+str(round(OutputY[i],6))+"\t"+str(round(TestY[i][0],6))+"\t"+str(round(AbsError,6)))                

                ExplainedVarianceR2=( LearnModel.score(TestX,TestY) )    # This is actually R2
                MeanRelativeError=( np.mean(( abs(LearnModel.predict(TestX)- TestY)/TestY) ** 1) );  #MeanRelativeError=( np.mean(( (LearnModel.predict(TestX)- TestY)) ** 2) )
                AbsError=0.0
                if RegressionFlag[0]:
                    for i in range(len(TestY)):
                        Temp=abs(TestY[i][0]-OutputY[i])/TestY[i]
                        AbsError+=Temp
                    AbsError/=len(TestY)
                elif RegressionFlag[1]:
                    AbsError = 0.0
                    for i in range(len(TestY)):
                        if( TestY[i][0] != OutputY[i]):
                            AbsError +=1
                    AbsError/=len(TestY)
                            
                AvgRelativeErrorAcrossFolds+=(AbsError) #(MeanRelativeError) (MeanRelativeError)
                AvgRelativeR2AcrossFolds+=(ExplainedVarianceR2)
                print "\t MeanRelativeError: "+str(AbsError)+"\t R2/Explained variance: "+str(ExplainedVarianceR2)
                if hasattr(LearnModel, 'feature_importances_'): 
                    FImp=(LearnModel.feature_importances_)
                    FeatureImp=[]
                    for Idx,CurrParam in enumerate(RestOfColumns):
                        Temp=[]
                        Temp.append(round(FImp[Idx],5))
                        Temp.append(CurrParam)
                        FeatureImp.append(Temp)
                    SortedFeatureImp=sorted(FeatureImp, key=itemgetter(0))
                    for Idx,CurrParam in enumerate(SortedFeatureImp):
                        print "\t CurrParam: "+str(CurrParam)    
            
        AverageErrors[CurrCol][Method]=    float(AvgRelativeErrorAcrossFolds/NumFolds)
        AverageR2[CurrCol][Method]=float(AvgRelativeR2AcrossFolds/NumFolds)
        if ModelDiagnostics:
            ModelDiagnosticsOutput.close()
        print "\t AvgRelativeErrorAcrossFolds: "+str(AverageErrors[CurrCol][Method])+" CumulRelativeErrorAcrossFolds "+str(AvgRelativeErrorAcrossFolds)
        print "\t AverageR2/ExplainedVariance: "+str(AverageR2[CurrCol][Method])+" CumulativeRelativeErrorAcrossFolds "+str(AvgRelativeR2AcrossFolds)
        if(not(CheckAllModels)):
            AllModelsNotChecked=False

 print "\n\n\t ----- Summary -------- \n\n"
 print "\t Parameter \t LearnModel \t\t\t\t\t Avg.Error \t R2 "
 if(LeaveOneOut):        
  for CurrCol in XParams:
   if( (CurrCol in AverageErrors) and (CurrCol in AverageR2) ):
    print "\t CurrParam: "+str(CurrCol)
    for CurrMethod in AverageErrors[CurrCol]:
          print "\t "+str(CurrCol)+"\t "+str(CurrMethod)+"\t\t\t\t\t "+str(round(AverageErrors[CurrCol][CurrMethod],6))+"\t "+str(round(AverageR2[CurrCol][CurrMethod],6))    
 else:
  CurrParam=XParams[0]
  for CurrMethod in AverageErrors[CurrParam]:
      if(CurrParam in AverageR2):
          print "\t "+str(CurrParam)+"\t "+str(CurrMethod)+"\t\t\t\t\t "+str(round(AverageErrors[CurrParam][CurrMethod],6))+"\t "+str(round(AverageR2[CurrParam][CurrMethod],6))

def CheckTolerance(Num,Tolerance,List):
    return next( (x for x in List if( ( x > ( Num*(1-Tolerance) ) ) & ( x < ( Num*(1+Tolerance) ) ) ) ) , False)

def BinParam(Data,Param,Bins):
    BinnedData={}
    for CurrBin in Bins:
        BinnedData[CurrBin[0]]=Data[ ( Data[Param] > CurrBin[0] ) & ( Data[Param] <= CurrBin[1] ) ]
        print "\t Bin: "+str(CurrBin)+" length "+str(BinnedData[CurrBin[0]].shape)
    return BinnedData

def FindInDF(Prop1,Prop2,Prop1Val,Prop2Val,DF):
    for idx,CurrRowData in DF.iterrows():
        if( ( CurrRowData[Prop1]==Prop1Val ) and ( CurrRowData[Prop2]==Prop2Val ) ):
            return CurrRowData
    print "\t Couldn't find Prop1: "+str(Prop1Val)+"\t Prop2: "+str(Prop2Val)
    sys.exit()

def OverlapData(TestX,TrainX,XParams,TestValueTolerance=0.1):
        FilteredTestX=TestX
        for currCol in XParams:
            if (~( (currCol in TestX.columns)  and (currCol in TrainX.columns) ) ):
                print "\t currCol is not found in Test/Train data"
                sys.exit()

        for Idx,CurrCol in enumerate(XParams):
            TempTestX=pd.DataFrame(columns=TestX.columns)
            print "\t Curr-param: "+str(CurrCol)+"\t FilteredTestX-shape: "+str(FilteredTestX.shape)
            for index,CurrRow in FilteredTestX.iterrows():
                TestVal=CurrRow[CurrCol]
                CheckRangeResult=CheckTolerance(TestVal,TestValueTolerance,TrainX[CurrCol])            
                if(CheckRangeResult):
                    df=CurrRow.copy()
                    TempTestX=TempTestX.append(df,ignore_index=True)
                    
            FilteredTestX=TempTestX    
        return FilteredTestX

def ErrorBinning(TempTestSet,AllParams):
    AbsErrorBins=[(0.0,0.049),(0.05,0.099),(0.1,0.199),(0.2,0.24999),(0.25,2)]
    BinnedTestData=BinParam(TempTestSet,"AbsError",AbsErrorBins)
    AvgBinnedTestData={}
    VarBinnedTestData={}

    for CurrKey in BinnedTestData:
        AvgBinnedTestData[CurrKey]={}
        VarBinnedTestData[CurrKey]={}
        print "\t CurrKey: "+str(CurrKey)+" shape: "+str(BinnedTestData[CurrKey].shape)
        print "\t Average\t Variance\t Minimum\t Maximum "
        for CurrCol in AllParams:
        #print "\t CurrCol: "+str(CurrCol)
            AvgBinnedTestData[CurrKey][CurrCol]=BinnedTestData[CurrKey][CurrCol].mean()
            VarBinnedTestData[CurrKey][CurrCol]=BinnedTestData[CurrKey][CurrCol].var()

            print "\t\t"+str(CurrCol)+"\t"+str(AvgBinnedTestData[CurrKey][CurrCol])+"\t"+str(VarBinnedTestData[CurrKey][CurrCol])+"\t"+str(BinnedTestData[CurrKey][CurrCol].min())+"\t"+str(BinnedTestData[CurrKey][CurrCol].max())

def BuildModels(argv):
    """
    Testing how help works in python.
    print "\t Usage: Learn.py -x <Training file name> \n\t Optional: \n\t\t -o <Output file name> \n\t\t -y <Test-file-name> \n\t\t -c <correlation-flag> 0: No correlaiton 1: IpStats correlation"
    """
    InputFileName=''
    OutputFileName=''
    IpRegressionFlag=''
    IpClassificationFlag=''
    IpClusteringFlag=''
    IpCheckAllModelsFlag=''
    TestFileName=''
    CorrelationFlag=''
    TestResProvided = ''
    verbose=False 
    try:
       opts, args = getopt.getopt(sys.argv[1:],"x:o:r:i:y:c:t:h:v:",["training=","output=","regflag=","checkallflag=","test=","correlation","testy","help","verbose"])
    except getopt.GetoptError:
        #print str(err) # will print something like "option -a not recognized"
       usage()
       sys.exit(2)
      
    for opt, arg in opts:
        print "\t Opt: "+str(opt)+" argument "+str(arg)    
        if opt == '-h':
            usage()        
        elif opt in ("-x", "--training"):
            InputFileName= arg.strip()
            print "\t\t Input file is "+str(InputFileName)
        elif opt in ("-y", "--test"):
            TestFileName= arg.strip()
            print "\t\t Test file is "+str(TestFileName)
        elif opt in ("-r", "--regflag"):
            TempArg=int( arg.strip())
            if(TempArg==0):
                IpRegressionFlag=True; IpClassificationFlag=False;IpClusteringFlag=False
            elif(TempArg==1):
                IpRegressionFlag=False; IpClassificationFlag=True;IpClusteringFlag=False
            elif(TempArg==2):
                IpRegressionFlag=False; IpClassificationFlag=False;IpClusteringFlag=True

            else:
                print "\t ERROR: Illegal value provided for \"-r\" flag.     "
                usage()    
            print "\t Regr: "+str(IpRegressionFlag)+"\t Class: "+str(IpClassificationFlag)+"\t Clustering: "+str(IpClusteringFlag)
        elif opt in ("-i", "--checkallflag"):
            TempArg=int(arg.strip())
            if(TempArg>0):
                IpCheckAllModelsFlag=True
            else:
                IpCheckAllModelsFlag=False
            
            print "\t IpCheckAllModelsFlag is "+str(IpCheckAllModelsFlag)+"\n";                                        
        elif opt in ("-c", "--correlation"):
            CorrelationFlag= arg.strip()
            print "\t Correlation flag is "+str(CorrelationFlag)+"\n";    
        elif opt in ("-t", "--testy"):
            Temp = int(arg.strip())
            if(Temp>0):
                TestResProvided = True
            else:
                TestResProvided = False
            print "\t TestResProvided flag is "+str(TestResProvided)+"\n";                      
        elif opt in ("-o", "--output"):
            OutputFileName=arg.strip()
            print "\t Source file is "+str(OutputFileName)+"\n";            
        else:
               usage()

    if(len(opts)==0):
        usage()

    if((InputFileName=='') or (TestResProvided=='')):
        usage()
    if(OutputFileName==''):
        OutputFileName='DefaultOutputFile.log'
        print "\t INFO: Using default output file name: "+str(OutputFileName)
    if(CorrelationFlag==''):
        CorrelationFlag=0
        print "\t\t INFO: Using default correlation flag: "+str(CorrelationFlag)    
    
    print "\t "
    IpStats=pd.read_csv(InputFileName,sep=',',header=0) #sep='\t' sep=','
    IpStats = IpStats.fillna('0') #Not sure whether this can be used as a standard practice.
    print "\t WARNING: NaN across training data is substituted with 0"
    IpStats=shuffle(IpStats);IpStats=shuffle(IpStats);IpStats=shuffle(IpStats)
    print "\t IpStats.shape: "+str(IpStats.shape)#+"\n\t columns "+str(IpStats.columns)
    
    YParams= ["Survived"] 
    XYParams=["PassengerId","Survived","Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
    XParams=["PassengerId","Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]    

    XParams = ["Sex","PassengerId","Age","Fare","Ticket"]
    AllParams= XParams
    
    IpStats = MapColVals(IpStats,['Sex',"Ticket","Cabin","Embarked"])
    for currParam in IpStats.columns:
        print "\t Param: %s type: %s "%(currParam,type(IpStats[currParam][0]))
    #sys.exit()

    for CurrParam in YParams:
        XYParams.append(CurrParam)
    for CurrParam in XParams:
        XYParams.append(CurrParam)

    print "\t len(XParams): "+str(len(XParams))+"\t YParams.shape: "+str(len(YParams))
    TempX=IpStats[XParams]

    ChoseModel=''
    if(IpRegressionFlag):
       ChoseModel=ChoseRegModel
    elif(IpClassificationFlag):
       ChoseModel=ChoseClassModel
    elif(IpClusteringFlag):
       ChoseModel=ChoseClusterModel

    CrossValidation(IpStats,XParams,YParams,RegressionFlag=[IpRegressionFlag,IpClassificationFlag,IpClusteringFlag],UseModel=ChoseModel,NumFolds=GlobalNumFolds,NumModels=GlobalNumModels);
    sys.exit()

    if(CorrelationFlag==1):
        CorrOutput=IpStats.corr(method='pearson')
        InputCorrelationFileName='CorrelationInput'+str(OutputFileName)
        CorrOutput.to_csv(InputCorrelationFileName,sep='\t')
    
    if(TestFileName==""):   
        sys.exit()

    if(TestFileName!=''):
        print "\n\t Opening test file name: "+str(TestFileName)
        TestStats=pd.read_csv(TestFileName,sep=',',header=0) #sep='\t' sep=','
        TestStats = TestStats.fillna(0)
        TestStats = MapColVals(TestStats,['Sex',"Ticket","Cabin","Embarked"])
        print "\t WARNING: NaN across test data is substituted with 0"

        if(CorrelationFlag==2):
           CorrOutput=IpStats.corr(method='pearson')
           InputCorrelationFileName='CorrelationInput'+str(OutputFileName)
           CorrOutput.to_csv(InputCorrelationFileName,sep='\t')        

        TestX=TestStats[XParams]
        TempTestSet = TestStats 
        #FilteredTestX=OverlapData(TestX,TrainX,XParams,TestValueTolerance=0.1);        TempTestSet=FilteredTestX

        print "\t Test-shape "+str(TempTestSet.shape)+"\t Test columns "+str(TempTestSet.columns);
        if(IpCheckAllModelsFlag):
            XParams=AllParams    
            
        TestX=(TempTestSet[XParams]).as_matrix()
        if TestResProvided:
            TestY=(TempTestSet[YParams]).as_matrix()
        
        TrainX=(IpStats[XParams]).as_matrix()
        TrainY=(IpStats[YParams]).as_matrix()

        print "\t Test shape- x "+str(TestX.shape)#+" y "+str(TestY.shape)
        print "\t Train shape-x "+str(TrainX.shape)+" y "+str(TrainY.shape)
        print "\t XParams: "+str(XParams)
        print "\t YParams: "+str(YParams)
        #sys.exit()
        VarianceScoreCollection={}
        ManualRelativeErrorCollection={}
        NumModelsToUse=-1
        ModelToChose=-1
        if(IpRegressionFlag):
            NumModelsToUse=GlobalNumModels
            ModelToChose=ChoseRegModel
        elif(IpClassificationFlag):
            NumModelsToUse=GlobalNumClassModels
            ModelToChose=ChoseClassModel
        elif(IpClusteringFlag):
            NumModelsToUse=GlobalNumClusteringModels
            ModelToChose=ChoseClusteringModel    
        
        for ModelIdx in range(NumModelsToUse):
            if(not(IpCheckAllModelsFlag)):
                ModelIdx=ModelToChose 
            print "\n"
            (LearnModel,Method)=ChoseLearnModel(UseModel=ModelIdx,RegressionFlag=[IpRegressionFlag,IpClassificationFlag,IpClusteringFlag])#True)
            LearnModel.fit(TrainX,TrainY.ravel())
            OutputY=LearnModel.predict(TestX)
            AbsError=[]
            ManualMeanRelativeError=0.0
            if(TestResProvided):
                if(IpRegressionFlag):
                    for i in range(len(OutputY)):
                        Temp=float(abs(OutputY[i]-TestY[i][0])/TestY[i][0])
                        ManualMeanRelativeError+=Temp
                        AbsError.append(Temp)
                elif(IpClassificationFlag):
                    for i in range(len(OutputY)):
                        Temp = False
                        if (OutputY[i] == TestY[i][0]):
                            Temp = True
                        else:
                            ManualMeanRelativeError+= 1
                        AbsError.append(Temp)

                ManualMeanRelativeError/=len(OutputY)
                MeanRelativeError=( np.mean(( abs(OutputY- TestY[0])/TestY[0]) ** 1) )
                VarianceScore=LearnModel.score(TestX,TestY)
                print "\t MeanRelativeError for the test set is: "+str(MeanRelativeError)+" Variance-score: "+str(VarianceScore)+" ManualMeanRelativeError "+str(ManualMeanRelativeError)

                TestOutputComparison=open('TestOutputComparison.dat','w')
                TestVecLen=len(TestX[0])
                TempTestSet['AbsError']=AbsError
                for i in range(len(OutputY)):
                    TestOutputComparison.write("\n\t"+str(i)+"\t"+str(round(OutputY[i],6))+"\t"+str(round(TestY[i][0],6))+"\t"+str(round(AbsError[i],6)))
                    #NecessaryRow=FindInDF(XParams[TestVecLen-1],XParams[TestVecLen-2],TestX[i][TestVecLen-1],TestX[i][TestVecLen-2],TempTestSet)
                    #for CurrParam in NecessaryRow:
                        #TestOutputComparison.write("\t"+str(CurrParam))
                TestOutputComparison.close()    
        
                VarianceScoreCollection[Method]=VarianceScore
                ManualRelativeErrorCollection[Method]=ManualMeanRelativeError
                if hasattr(LearnModel,feature_importances_):
                    print "\t Feature importance: "+str(LearnModel.feature_importances_);
                    #print "\t Feature importance: "+str(LearnModel.coef_); sys.exit()

                if(ModelDiagnostics):
                    ErrorBinning(TempTestSet,AllParams)
                print "\t MeanRelativeError for the test set is: "+str(MeanRelativeError)+" Variance-score: "+str(VarianceScore)+" ManualMeanRelativeError "+str(ManualMeanRelativeError); 
            else:
                TestOutputComparison=open('TestOutputComparison.dat','w')
                TestVecLen=len(TestX[0])
                if IpRegressionFlag:
                    for i in range(len(OutputY)):
                        TestOutputComparison.write("\n\t"+str(i)+"\t"+str(round(OutputY[i],6)))
                elif IpClassificationFlag:
                    for i in range(len(OutputY)):
                        #TestOutputComparison.write("\n\t"+str(i)+"\t"+str(OutputY[i]))
                        TestOutputComparison.write("\n"+str(TempTestSet["PassengerId"][i])+","+str(OutputY[i]))
                TestOutputComparison.close()                    
                sys.exit()
        print "\t Format: <Method> <Variance-score> <ManualRelativeError> "    
        for CurrMethod in  VarianceScoreCollection:
            if(CurrMethod in ManualRelativeErrorCollection):
                print "\t "+str(CurrMethod)+"\t "+str(round(VarianceScoreCollection[CurrMethod],6))+"\t "+str(round(ManualRelativeErrorCollection[CurrMethod],6))

    print "\n\n"     
          
if __name__ == "__main__":
   BuildModels(sys.argv[1:])

