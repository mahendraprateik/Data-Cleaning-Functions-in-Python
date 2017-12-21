
# coding: utf-8

# In[ ]:

"""

READ THIS

This assignment is the most challenging assignment so far. Please make sure you start early to finish it on time.
I will give you helper functions and explain the concepts on 10/30/2017
This class will prepare your datasetr for model building.
General pipeline before you build a predictive model is
Our general variable treatment follows the pipeline below
pipeline = [drop_nan_col, drop_zero_var_col, drop_zero_car_col,drop_high_levels, 
            replace_missing, encode_target, transform, create_dummies]
            
Your task is to implement these functions.
You can use pandas functions or any functions you want, such as get_dummies, sum, np.exp, etc. use built-in functions
in this assignment.

"""


# In[ ]:

# Name:

import pandas as pd
# In[220]:

class variableTreatment():
    
    def drop_nan_col(self, df, threshold): 
        """
        Objective: Drops columns most of whose rows missing
        
        Inputs:
        1. Dataframe df: Pandas dataframe
        2. threshold: Determines which columns will be dropped.
                      if threshold is .9, the columns with 90% missing value will be dropped
        
        Outputs:
        1. Dataframe df with dropped columns (if no columns are dropped, you will return the same dataframe)
        """
        for c in df.columns:
            if (float(df[c].isnull().sum())/df.shape[0]) > threshold:
                df.drop(c, axis = 1, inplace = True)
            else:
                pass
        return df


    
    
    def drop_zero_var_col(self, df):
        """
        Objective: Drops numerical columns with zero variance
        
        Inputs:
        1. Dataframe df: Pandas dataframe
        
        
        Outputs:
        1. Dataframe df with dropped columns (if no columns are dropped, you will return the same dataframe)
        """
        
        for c in df.select_dtypes(include = ['float64', 'float32', 'int64']).columns:
            if np.array(df[c]).std() == 0:
                df.drop(c, axis = 1, inplace = True)
            else:
                pass
        return df
        
    
        
        
    def drop_zero_car_col(self, df):
        """
        Objective: Drops categorical columns with same levels, such as a column with all 'yes' values
        
        Inputs:
        1. Dataframe df: Pandas dataframe
        
        
        Outputs:
        1. Dataframe df with dropped columns (if no columns are dropped, you will return the same dataframe)
        """
        for c in df.select_dtypes(include = ['object']).columns:
            if len(df[c].unique().tolist()) == 1:
                df.drop(c, axis = 1, inplace = True)
            else:
                pass
        return df
    
        
        
    def drop_high_levels(self, df, threshold):
        """
        this task will eliminate categorical columns if this column has a lot of levels. 
        inputs:
        1. Dataframe df: Pandas dataframe
        2. Threshold: How many levels you want at most
        
        outputs:
        1. Dataframe df: updated dataframe without dropped columns
        
        """
        for c in df.select_dtypes(include = ['object']).columns:
            if len(df[c].unique().tolist()) > threshold:
                df.drop(c, axis = 1, inplace = True)
            else:
                pass
        return df
                
        

    def replace_missing(self, df, num_val):
        """
        Objective: Replaces missing values with given values
        Note: replace missing categorical variables with 'unknown' string
        
        Inputs:
        1. Dataframe df: Pandas dataframe
        2. num_val: User decides with what values they want to replace the missing numerical values. 
                    This value can be mean median mode or zero
    
        
        
        Outputs:
        1. Dataframe df with imputed missing values
        """
        #df.select_dtypes(include = ['object']).fillna(value = "unknown", inplace = True) #fillna(value = 'unknown')
        #df.select_dtypes(include = ['float64', 'float32', 'int']).fillna(value = 'num_val', inplace = True)#fillna(value = median)
        for c in df.select_dtypes(include = ['object']):
            df[c].fillna(value = 'unknown', inplace = True)
        if num_val == 'median':
            for v in df.select_dtypes(include = ['float64', 'float32', 'int']).columns:
                df[v].fillna(df[v].median(), inplace=True)
        elif num_val == 'mean':
            for v in df.select_dtypes(include = ['float64', 'float32', 'int']).columns:
                df[v].fillna(df[v].mean(), inplace=True)
        elif num_val =='mode':
            for v in df.select_dtypes(include = ['float64', 'float32', 'int']).columns:
                df[v].fillna(df[v].mode(), inplace=True)
        else:
            df[v].fillna(0, inplace=True)
    
        return df
    
    
    def encode_target(self, df, target_name):
        """
        Objective: Encodes the class label if class column is categorical.
                   If class column is numerical just return the same dataframe without doing anything
                   Do not forget that clas label might have more than 2 levels (yes and no is two levels)
                   Target levels can be agree, stringly agree, disagree strongly disagree, neutral (5 levels)
                   Do not hard code.
                   
        Inputs: 
        1. Dataframe df: Pandas dataframe
        
        Outputs:
        1. Dataframe df with encoded binary class labels. 
        """
        if(df[target_name].dtype != 'object'):
            return df
        else:
            e = df[target_name].unique().tolist()
            for i in range(0, len(df[target_name].value_counts())):
                df.loc[(df[target_name] == e[i]), target_name ] = i
            
            df[target_name] = df[target_name].astype(int)
            return df
        
    def transform(self, df, label_name):
        """
        Objective: Transforms numerical values in a way that it will increase model accuracy.
        
        This is the dictionary to use
        {'asis':0, 'log':0, 'exp':0, 'sqrt':0, 'pow2':0}
        
        inputs:
        1. Dataframe df: Pandas dataframe 
        
         outputs:
        1. Dataframe df with transformed values
        """
        for c in df.select_dtypes(include = ['float64', 'float32', 'int64', 'int']).columns:
            corr = {'asis':0, 'log':0, 'exp':0, 'sqrt':0, 'pow2':0}
            corr['asis'] = abs(np.corrcoef(df[c].astype(float), df[label_name].astype(float))[1][0])
            if all(df[c]>0):
                corr['log'] =  abs(np.corrcoef(np.log(df[c].astype(float)), df[label_name].astype(float))[1][0])
            else:
                corr['log'] = 0
            corr['exp'] = abs(np.corrcoef(np.exp(df[c].astype(float)), df[label_name].astype(float))[1][0])
            corr['sqrt'] = abs(np.corrcoef(np.sqrt(df[c].astype(float)), df[label_name].astype(float))[1][0])
            corr['pow2'] = abs(np.corrcoef(np.power(df[c].astype(float),2), df[label_name].astype(float))[1][0])
            maximum = max(corr, key=corr.get)
            if maximum == 'asis':
                pass
            else:
                
                if maximum == 'log':
                    df[c + str("-") + maximum] = df[c].apply(np.log)
                elif maximum == 'exp':
                    df[c + str("-") + maximum] = df[c].apply(np.exp)
                elif maximum == 'sqrt':
                    df[c + str("-") + maximum] = df[c].apply(np.sqrt)
                elif maximum == 'pow2':
                    df[c + str("-") + maximum] = df[c].apply(lambda x: x**2)
                df.drop(c, axis = 1, inplace = True)
        c = df["class"]
        df.drop('class', axis = 1, inplace = True)
        df['class'] = c     
        return df
    
    def create_dummies(self, df, label_name):
        """
        Objective: Creates dummy variables for categorical variables 
        (0 1 binary columns for each level for a categorical column - ignore one of the levels)
        
        Inputs:
        1. Dataframe df: Pandas dataframe
        
        Outputs:
        1. Dataframe df with dummy variables
        """
        import pandas as pd
        f = df[label_name]
        df.drop(label_name, axis = 1, inplace = True)
        df = pd.get_dummies(df, drop_first = True)
        df[label_name] = f
        return df
        
        
        


# In[221]:

# In[222]:

# create and instance from the class variableTreatment
VT = variableTreatment()


# In[223]:

VT


# In[224]:

df.head()


# In[225]:

VT.drop_nan_col(df, 0.9).head()


# In[226]:

VT.drop_zero_var_col(df).head()


# In[227]:

VT.drop_zero_car_col(df).head()


# In[228]:

VT.drop_high_levels(df, 100).head()


# In[229]:

VT.replace_missing(df, 'median').head()


# In[230]:

VT.encode_target(df, 'class').head()


# In[231]:

VT.transform(df, 'class').head()



