from operator import le
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import pickle

data=pd.read_excel("CCPP/Folds5x2_pp.xlsx")
df=pd.DataFrame(data)

y=df['PE']
X=df.drop(['PE'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
liner_reg=LinearRegression()
liner_reg.fit(X_train,y_train)
print(X_test)
y_pred=liner_reg.predict(X_test)

pickle.dump(liner_reg, open('model.pkl','wb'))

