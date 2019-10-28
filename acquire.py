
import stats
from env import host, user, password
url = f'mysql+pymysql://{user}:{password}@{host}/telco_churn'
def get_telco():
    df = pd.read_sql('''select * from customers join contract_types using 
    (contract_type_id) join internet_service_types using (internet_service_type_id) 
    join payment_types using (payment_type_id);
    ''' ,url)
    return df
