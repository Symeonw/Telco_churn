import acquire
import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
df = get_telco()
df["tenure_years"] = [n/12 for n in df.tenure]
df_2 = df.drop(columns = ["customer_id", "payment_type_id", "internet_service_type_id", "contract_type_id"], inplace=True)


df_2 = get_telco()
df_2.drop(columns = ["customer_id", "payment_type_id", "internet_service_type_id", "contract_type_id"], inplace=True)
train_2, test_2 = train_test_split(df_2, train_size = .75, random_state = 123)
X_train_2 = train.drop(columns = "churn")
X_test_2 = test.drop(columns = "churn")
y_train_2 = train.churn
y_test_2 = test.churn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer




numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
numeric_features = train._2select_dtypes(include=['int64', 'float64']).columns
categorical_features = train_2.select_dtypes(include=['object']).drop(['churn'], axis=1).columns
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifiers = [
    KNeighborsClassifier(12),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True, random_state = 123),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]
for classifier in classifiers:
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])
    pipe.fit(X_train_2, y_train_2)   
    print(classifier)
    print("model score: %.3f" % pipe.score(X_test_2, y_test_2))
