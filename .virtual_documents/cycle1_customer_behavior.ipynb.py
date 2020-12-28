import glob
import warnings
import numpy  as np
import pandas as pd

import seaborn           as sns
import matplotlib.pyplot as plt

from IPython.display       import Image
from IPython.core.display  import HTML


def jupyter_settings():
    get_ipython().run_line_magic("matplotlib", " inline")
    get_ipython().run_line_magic("pylab", " inline")
    
    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24
    
    display( HTML('<style>.container { width:100% get_ipython().getoutput("important; }</style>'))")
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option('display.expand_frame_repr', False)
    
    warnings.filterwarnings("ignore")
    
    sns.set()


seed = 42
np.random.seed()

jupyter_settings()


for file in glob.glob("01-Data\*.csv"):
    print(file)


df_raw = pd.read_csv('01-Data/olist_orders_dataset.csv', \
                     parse_dates=['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',\
                                  'order_delivered_customer_date', 'order_estimated_delivery_date'])


df_raw.sample(2)


df01 = df_raw.copy()


df01.columns


print(f'Number of Rows: {df01.shape[0]}')
print(f'Number of Columns: {df01.shape[1]}')


df01.dtypes


df01.isnull().sum()


df02 = df01.copy()


df02['order_purchase_year_week'] = df02['order_purchase_timestamp'].dt.strftime('get_ipython().run_line_magic("Y-%W')", "")


df02.sample().T


df03 = df02.copy()


df03 = df03[df03['order_status'] == 'canceled'][['order_id', 'order_purchase_year_week']]
df03.head()


df04 = df03.copy()


df04 = df04.groupby('order_purchase_year_week').count().reset_index()
df04 = df04.sort_values('order_purchase_year_week')

count_data = np.array(df04['order_id'])
n_count_data = len(count_data)

plt.figure(figsize=(18,8))
plt.bar(np.arange(n_count_data),count_data)
plt.xlim([0, n_count_data ]);


import pymc3 as pm
with pm.Model() as model:
    # Prior
    alpha = 1.0 / count_data.mean()
    lambda_01 = pm.Exponential('lambda_01', alpha)
    lambda_02 = pm.Exponential('lambda_02', alpha) 

    tau = pm.DiscreteUniform('tau', lower=0, upper=n_count_data-1)


    # Posterior
    idx = np.arange(n_count_data)
    lambda_ = pm.math.switch(tau > idx, lambda_01, lambda_02)
    obs = pm.Poisson('obs', lambda_, observed=count_data)

    # Likelihood
    trace = pm.sample(draws=10000, tune=5000, step=pm.Metropolis())


# posterior
plt.figure(figsize=(18, 10))

plt.subplot(3, 1, 1)
plt.hist(trace['lambda_01'], histtype='stepfilled', bins=30, density=True );
plt.xlim([0, 15 ]);

plt.subplot(3, 1, 2)
plt.hist(trace['lambda_02'], histtype='stepfilled', bins=30, density=True);
plt.xlim([0, 15 ]);

plt.subplot(3, 1, 3)
w = 1.0 / trace['tau'].shape[0] * np.ones_like(trace['tau'])
plt.hist(trace['tau'], bins=n_count_data, weights=w, rwidth=2.);
plt.xlim([40, n_count_data-30 ]);


trace['lambda_01'].mean()


trace['lambda_02'].mean()


plt.figure(figsize=(18,10))
# Behaviour plot
canceled_by_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
    ix = day < trace['tau']
    canceled_by_day[day] = (trace['lambda_01'][ix].sum() + trace['lambda_02'][~ix].sum()) / trace['tau'].shape[0]
    
# plots 
plt.plot(range(n_count_data), canceled_by_day, color='red')
plt.ylim([0, 30])

plt.bar(np.arange( n_count_data ), count_data)
plt.xlim([0, n_count_data ]);



