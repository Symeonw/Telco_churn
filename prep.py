import acquire
import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
df = get_telco()
df["tenure_years"] = [n/12 for n in df.tenure]
df.drop(columns = ["customer_id", "payment_type_id", "internet_service_type_id", "contract_type_id"], inplace=True)
df.columns
df.info()
for col in ['gender', 'senior_citizen', 'partner', 'dependents',\
    'phone_service', 'multiple_lines', 'online_security', 'online_backup',\
    'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',\
    'paperless_billing', 'churn','contract_type', 'internet_service_type', 'payment_type']:
    df[col] = df[col].astype('category')
df["total_charges"] = pd.to_numeric(df["total_charges"], errors= "coerce").dropna()
for col in ["tenure_years", "monthly_charges", "tenure"]:
    df[col] = df[col].astype("float")

df_dict = df.to_dict(orient='records')
from sklearn.feature_extraction import DictVectorizer
dv_df = DictVectorizer(sparse=False)
df_encoded = dv_df.fit_transform(df_dict)
df = pd.get_dummies(df, prefix_sep='_', drop_first=True)

df["pd"] = df.partner_Yes + df.dependents_Yes
df["phone_multiple"] = df.phone_service_Yes + df.multiple_lines_Yes
df.phone_multiple
df

train, test = train_test_split(df, train_size = .75, random_state = 123)
X_train = train.drop(columns = "churn_Yes")
X_test = test.drop(columns = "churn_Yes")
y_train = train.churn_Yes
y_test = test.churn_Yes
scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
train_scaled_data = pd.DataFrame(scaler.transform(train))
test_scaled_data = pd.DataFrame(scaler.transform(test))
new_col = ['tenure', 'monthly_charges', 'total_charges', 'tenure_years',\
       'gender_Male', 'senior_citizen_1', 'partner_Yes', 'dependents_Yes',\
       'phone_service_Yes', 'multiple_lines_No phone service',\
       'multiple_lines_Yes', 'online_security_No internet service',\
       'online_security_Yes', 'online_backup_No internet service',\
       'online_backup_Yes', 'device_protection_No internet service',\
       'device_protection_Yes', 'tech_support_No internet service',\
       'tech_support_Yes', 'streaming_tv_No internet service',\
       'streaming_tv_Yes', 'streaming_movies_No internet service',\
       'streaming_movies_Yes', 'paperless_billing_Yes', 'churn_Yes',\
       'contract_type_One year', 'contract_type_Two year',\
       'internet_service_type_Fiber optic', 'internet_service_type_None',\
       'payment_type_Credit card (automatic)', 'payment_type_Electronic check',\
       'payment_type_Mailed check', 'pd', 'phone_multiple']
train_scaled_data.columns = new_col
test_scaled_data.columns = new_col
train_scaled_data.columns
test_scaled_data.columns
test_scaled_data
X_train_scaled = train_scaled_data.drop(columns = "churn_Yes")
X_test_scaled = test_scaled_data.drop(columns = "churn_Yes")
y_train_scaled = train_scaled_data.churn_Yes
y_test_scaled = test_scaled_data.churn_Yes