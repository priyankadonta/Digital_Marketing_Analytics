#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import keras

get_ipython().run_line_magic('matplotlib', 'widget')
import ipywidgets as widgets
import ipympl


plt.style.use('ggplot')


# In[2]:


def add_derived_columns(df):                               # step 1: add JID and standartize timestamps 
    df_ext = df.copy()
    df_ext['jid'] = df_ext['uid'].map(str) + '_' + df_ext['conversion_id'].map(str)
    
    min_max_scaler = MinMaxScaler() # to standardize the timestamp
    for cname in ('timestamp'):
        x = df_ext[cname].values.reshape(-1, 1) 
        df_ext[cname + '_norm'] = min_max_scaler.fit_transform(x)
    
    return df_ext


# In[3]:


def sample_campaigns(df, n_campaigns):                     # step 2.1: reduce the dataset by sampling campaigns
    campaigns = np.random.choice( df['campaign'].unique(), n_campaigns, replace = False ) # randomly samples the data
    return df[ df['campaign'].isin(campaigns) ]


# In[4]:


def filter_journeys_by_length(df, min_touchpoints):        # step 2.2: remove short (trivial) journeys
    grouped = df.groupby(['jid'])['uid'].count().reset_index(name="count")
    return df[df['jid'].isin( grouped[grouped['count'] >= min_touchpoints]['jid'].values )]


# In[5]:


df3 = pd.read_csv("C:/Spring_2020/ADM/Assignment 3 Part 1/Assignment 3 Part 2/df3.csv")
#df2.head()


# In[6]:


#df3 = filter_journeys_by_length(df2, 1) # as count of jid is min when we put 2 it doesn't show any results after grouping
#df3.head(20)


# In[7]:


def balance_conversions(df):                               # step 3: balance the dataset: 
    df_minority = df[df.conversion == 1]                   # The number of converted and non-converted events should be equal.
    df_majority = df[df.conversion == 0]                   # We take all converted journeys and iteratively add non-converted 
                                                           # samples until the datset is balanced. We do it this way becasue  
    df_majority_jids = np.array_split(                     # we are trying to balance the number of events, but can add only
          df_majority['jid'].unique(),                     # the whole journeys.
          100 * df_majority.shape[0]/df_minority.shape[0] )
    
    df_majority_sampled = pd.DataFrame(data=None, columns=df.columns)
    for jid_chunk in df_majority_jids:
        df_majority_sampled = pd.concat([
            df_majority_sampled, 
            df_majority[df_majority.jid.isin(jid_chunk)]
        ])
        if df_majority_sampled.shape[0] > df_minority.shape[0]:
            break
    
    return pd.concat([df_majority_sampled, df_minority]).sample(frac=1).reset_index(drop=True)


# In[8]:


df4 = balance_conversions(df3) # the generated output has 13 rows with 6 converted and 7  non converted
#df4.head(20) 


# In[9]:


def map_one_hot(df, column_names, result_column_name):      # step 4: one-hot encoding for categorical variables
    mapper = {}                                             # We use custom mapping becasue IDs in the orginal dataset
    for i, col_name in enumerate(column_names):             # are not sequential, and standard one-not encoding 
        for val in df[col_name].unique():                   # provided by Keras does not handle this properly.
            mapper[val*10 + i] = len(mapper)
    
    def one_hot(values):
        v = np.zeros( len(mapper) )
        for i, val in enumerate(values): 
            mapped_val_id = mapper[val*10 + i]
            v[mapped_val_id] = 1
        return v    
    
    df_ext = df.copy()
    df_ext[result_column_name] = df_ext[column_names].values.tolist()
    df_ext[result_column_name] = df_ext[result_column_name].map(one_hot)
    
    return df_ext


# In[10]:


# all categories are mapped to one vector  
df5 = map_one_hot(df4, ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat8'], 'cats') #added vector cats 
#df5.head(15)
#df5.to_csv("C:/Spring_2020/ADM/Assignment 3 Part 1/Assignment 3 Part 2/20 campaigns-20200302T130818Z-001/20 campaigns/df5samples.csv")


# In[11]:


# the final dataframe used for modeling  
df6 = map_one_hot(df5, ['campaign'], 'campaigns').sort_values(by=['timestamp_norm'])   
#df6.head(15) 
#df6.to_csv("C:/Spring_2020/ADM/Assignment 3 Part 1/Assignment 3 Part 2/20 campaigns-20200302T130818Z-001/20 campaigns/df6samples.csv")


# In[12]:


get_ipython().run_line_magic('matplotlib', 'widget')


# In[13]:


# Data exploration

def journey_length_histogram(df):
    counts = df.groupby(['jid'])['uid'].count().reset_index(name="count").groupby(['count']).count()
    return counts.index, counts.values / df.shape[0]

hist_x, hist_y = journey_length_histogram(df4)

plt.plot(range(len(hist_x)), hist_y, label='all journeys')
plt.yscale('log')
plt.xlim(0, 80)
plt.xlabel('Journey length (number of touchpoints)')
plt.ylabel('Fraction of journeys')
plt.show()


#This returns the number of times one journey has occured

#The dataset contains journeys with up to 40 events(number of touchpoints) or more, but the number of journeys falls exponentially with the length:
#The number of journeys with several events is considerable; thus, it makes sense to try methods for sequential data.
# this means there is one user id who has impressions of 40 and the 


# In[14]:


n_campaigns = 20
def last_touch_attribution(df):
    
    def count_by_campaign(df):
        counters = np.zeros(n_campaigns)
        for campaign_one_hot in df['campaigns'].values:
            campaign_id = np.argmax(campaign_one_hot)
            counters[campaign_id] = counters[campaign_id] + 1
        return counters
        
    campaign_impressions = count_by_campaign(df)
    
    df_converted = df[df['conversion'] == 1]
    idx = df_converted.groupby(['jid'])['timestamp_norm'].transform(max) == df_converted['timestamp_norm']
    campaign_conversions = count_by_campaign(df_converted[idx])
        
    return campaign_conversions / campaign_impressions
    
lta = last_touch_attribution(df6)


# In[15]:


# Visualization of the attribution scores

campaign_idx = range(0, 3)

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
plt.bar( range(len(lta[campaign_idx])), lta[campaign_idx], label='LTA' )
plt.xlabel('Campaign ID')
plt.ylabel('Return per impression')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


fig.canvas.toolbar_visible = True
fig.canvas.header_visible = True
fig.canvas.footer_visible = True


# In[16]:


n_campaigns = 20
def first_touch_attribution(df):
    
    def count_by_campaign(df):
        counters = np.zeros(n_campaigns)
        for campaign_one_hot in df['campaigns'].values:
            campaign_id = np.argmax(campaign_one_hot)
            counters[campaign_id] = counters[campaign_id] + 1
        return counters
        
    campaign_impressions = count_by_campaign(df)
    
    df_converted = df[df['conversion'] == 1]
    idx = df_converted.groupby(['jid'])['timestamp_norm'].transform(min) == df_converted['timestamp_norm']
    campaign_conversions = count_by_campaign(df_converted[idx])
        
    return campaign_conversions / campaign_impressions
    
fta = first_touch_attribution(df6)


# In[17]:


# Visualization of the attribution scores

campaign_idx = range(0, 4)

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
plt.bar( range(len(fta[campaign_idx])), fta[campaign_idx], label='FTA' )
plt.xlabel('Campaign ID')
plt.ylabel('Return per impression')
plt.legend(loc='upper left')
plt.show()


# In[18]:


##Linear

def linear(df):
    
    def count_by_campaign(df):
        counters = np.zeros(n_campaigns)
        for campaign_one_hot in df['campaigns'].values:
            campaign_id = np.argmax(campaign_one_hot)
            counters[campaign_id] = counters[campaign_id] + 1
        return counters
        
    campaign_impressions = count_by_campaign(df)
    
    df_converted = df[df['conversion'] == 1]
    campaign_count = df_converted.groupby('jid')['campaign'].transform('count') #gives count of campaigns in one journey id
    
    return df['campaign'],1/campaign_count
df6=df6.head(500)
campaign_id, campaign_count = linear(df6)


# In[19]:


lin = pd.DataFrame()
lin['campaign_id'] = campaign_id
lin['campaign_count'] = campaign_count
lin.plot.bar(x='campaign_id', y='campaign_count', rot=0)


# In[37]:


# # Visualization of the attribution scores

# campaign_idx = range(0, 4)

# fig = plt.figure(figsize=(15,4))
# ax = fig.add_subplot(111)
# # lin.plot.bar()
# plt.bar(lin['campaign_id'], lin['campaign_count'], align='center', alpha=0.5)
# plt.xlabel('Campaign ID')
# plt.ylabel('Return per impression')
# plt.legend(loc='upper left')
# plt.show()
lin.plot.bar(x='campaign_id', y='campaign_count', rot=0)


# In[ ]:


min=lin['campaign_id'].min()
min


# In[ ]:


from ipywidgets import AppLayout, FloatSlider

plt.ioff()

slider = FloatSlider(
    orientation='horizontal',
    description='Factor:',
    value=100000,
    min=1000,
    max=10000000
)

slider.layout.margin = '0px 30% 0px 30%'
slider.layout.width = '40%'

fig = plt.figure()
fig.canvas.header_visible = False
fig.canvas.layout.min_height = '400px'
plt.title('Plotting of linear graph'.format(slider.value))

x = lin['campaign_id']
y = lin['campaign_count']

lines = plt.plot(x,y)

def update_lines(change):
    plt.title('Plotting of linear graph'.format(change.new))
    lines[0].set_data(x, y)
    fig.canvas.draw()
    fig.canvas.flush_events()

slider.observe(update_lines, names='value')

AppLayout(
    center=fig.canvas,
    footer=slider,
    pane_heights=[0, 6, 1]
)


# In[ ]:





# In[31]:


#df6.head(5)
#df6 = df6.groupby[df['jid'],df['campaign']]
#df6 = map_one_hot(df5, ['campaign'], 'campaigns').sort_values(by=['timestamp_norm']) 

def u_shaped(df):
    df6['conversion'] == 1
    df7 = df6.groupby(['campaign', 'jid']) 
    df7.head()


    df9 = pd.DataFrame()
    for key, item in df7:
        #print('new_group')
        df8 = df7.get_group(key).reset_index()
        #print(df8)
        df8['weight'] = 1

        n=len(df8)
    
#     df8['weight'].iloc[[0,-1]]=1
        i = 0
        for i in range(1,n):
            if i == 0 or i == n-1:
                df8.loc[i,'weight']=1
            else:
                df8.loc[i,'weight'] = 1/(n-1)
        
    #print(len(df7.get_group(key)), "\n\n")
    #print('assign weights')
        df9 = df9.append(df8)
    return df9
    
df10 = u_shaped(df6)
#df10


# In[32]:


df10[['jid','weight']].plot(kind='bar')
plt.show()


# In[33]:


#Logistic Regression

def features_for_logistic_regression(df):

    def pairwise_max(series):
        return np.max(series.tolist(), axis = 0).tolist()
    
    aggregation = {
        'campaigns': pairwise_max,
        'cats': pairwise_max,
        'click': 'sum',
        'cost': 'sum',
        'conversion': 'max'
    }
    
    df_agg = df.groupby(['jid']).agg(aggregation)
    
    df_agg['features'] = df_agg[['campaigns', 'cats', 'click', 'cost']].values.tolist()
    
    return (
        np.stack(df_agg['features'].map(lambda x: np.hstack(x)).values),
        df_agg['conversion'].values
    )


# In[34]:


x, y = features_for_logistic_regression(df6)
#print(np.shape(x))


# In[35]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.20, random_state = 1)


# In[36]:


# Quick sanity check
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
score = logisticRegr.score(x_test, y_test)
print(score)


# In[37]:


from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.constraints import NonNeg

m = np.shape(x)[1]
    
model = Sequential()  
model.add(Dense(1, input_dim=m, activation='sigmoid', name = 'contributions')) 

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) 
history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_val, y_val)) 
score = model.evaluate(x_test, y_test, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])


# In[39]:


# Visualization of the attribution scores
from sklearn.utils.extmath import softmax

keras_logreg = model.get_layer('contributions').get_weights()[0].flatten()[0:n_campaigns]
keras_logreg = softmax([keras_logreg]).flatten()

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
plt.bar(range(len(keras_logreg[campaign_idx])), keras_logreg[campaign_idx] )
plt.xlabel('Campaign ID')
plt.ylabel('Return per impression')
plt.show()


# In[40]:


#Basic LSTM
def features_for_lstm(df, max_touchpoints):
    
    df_proj = df[['jid', 'campaigns', 'cats', 'click', 'cost', 'time_since_last_click_norm', 'timestamp_norm', 'conversion']]
    
    x2d = df_proj.values
    
    x3d_list = np.split(x2d[:, 1:], np.cumsum(np.unique(x2d[:, 0], return_counts=True)[1])[:-1])
    
    x3d = []
    y = []
    for xi in x3d_list:
        journey_matrix = np.apply_along_axis(np.hstack, 1, xi)
        journey_matrix = journey_matrix[ journey_matrix[:, 5].argsort() ] # sort impressions by timestamp
        n_touchpoints = len(journey_matrix)
        padded_journey = []
        if(n_touchpoints >= max_touchpoints):
            padded_journey = journey_matrix[0:max_touchpoints]
        else:
            padded_journey = np.pad(journey_matrix, ((0, max_touchpoints - n_touchpoints), (0, 0)), 'constant', constant_values=(0))
            
        x3d.append(padded_journey[:, 0:-1])
        y.append(np.max(padded_journey[:, -1]))
        
    return np.stack(x3d), y

x, y = features_for_lstm(df6, max_touchpoints = 15)
print(np.shape(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.20, random_state = 1)


# In[41]:


from keras.models import Sequential 
from keras.layers import Dense, LSTM

n_steps, n_features = np.shape(x)[1:3]
    
model = Sequential() 
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, input_shape=(n_steps, n_features)))
model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) 
history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1, validation_data=(x_val, y_val)) 
score = model.evaluate(x_test, y_test, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])


# In[42]:


#LSTM with attention
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Input, Lambda, RepeatVector, Permute, Flatten, Activation, Multiply
from keras.constraints import NonNeg
from keras import backend as K
from keras.models import Model

n_steps, n_features = np.shape(x)[1:3]

hidden_units = 64

main_input = Input(shape=(n_steps, n_features))
    
embeddings = Dense(128, activation='linear', input_shape=(n_steps, n_features))(main_input)

activations = LSTM(hidden_units, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embeddings)

attention = Dense(1, activation='tanh')(activations)
attention = Flatten()(attention)
attention = Activation('softmax', name = 'attention_weigths')(attention)
attention = RepeatVector(hidden_units * 1)(attention)
attention = Permute([2, 1])(attention)

weighted_activations = Multiply()([activations, attention])
weighted_activations = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(hidden_units,))(weighted_activations)

main_output = Dense(1, activation='sigmoid')(weighted_activations)

model = Model(inputs=main_input, outputs=main_output)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) 
history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1, validation_data=(x_val, y_val)) 
score = model.evaluate(x_test, y_test, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])


# In[43]:


#Analysis of LSTM-A Model
def get_campaign_id(x_journey_step):
    return np.argmax(x_journey_step[0:n_campaigns])

attention_model = Model(inputs=model.input, outputs=model.get_layer('attention_weigths').output)

a = attention_model.predict(x_train)

attributions = np.zeros(n_campaigns)
campaign_freq = np.ones(n_campaigns)
for i, journey in enumerate(a):
    for step, step_contribution in enumerate(journey):
        if(np.sum(x_train[i][step]) > 0):
            campaign_id = get_campaign_id(x_train[i][step])
            attributions[campaign_id] = attributions[campaign_id] + step_contribution
            campaign_freq[campaign_id] = campaign_freq[campaign_id] + 1


# In[44]:


lstm_a = (attributions/campaign_freq)

fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(111)
plt.bar( range(len(lstm_a[campaign_idx])), lstm_a[campaign_idx], label='LSTM-A' )
plt.xlabel('Campaign ID')
plt.ylabel('Contribution')
plt.legend(loc='upper left')
plt.show()


# In[46]:


fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)

ratio = max(lta[campaign_idx]) / max(keras_logreg[campaign_idx])
plt.bar(np.linspace(0, len(campaign_idx), len(campaign_idx)), lta[campaign_idx], width=0.4, alpha=0.7, label='LTA' )
plt.bar(np.linspace(0, len(campaign_idx), len(campaign_idx)) - 0.3, keras_logreg[campaign_idx], width=0.4, alpha=0.7, label='Keras Log Reg' )
plt.xlabel('Campaign ID')
plt.ylabel('Contribution')
plt.legend(loc='upper left')
plt.show()


# In[49]:


fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)

ratio = max(fta[campaign_idx]) / max(keras_logreg[campaign_idx])
plt.bar(np.linspace(0, len(campaign_idx), len(campaign_idx)), fta[campaign_idx], width=0.4, alpha=0.7, label='FTA' )
plt.bar(np.linspace(0, len(campaign_idx), len(campaign_idx)) - 0.3, keras_logreg[campaign_idx], width=0.4, alpha=0.7, label='Keras Log Reg' )
plt.xlabel('Campaign ID')
plt.ylabel('Contribution')
plt.legend(loc='upper left')
plt.show()


# In[47]:


#Simulation

# Key assumption: If one of the campaigns in a journey runs out of budget, 
# then the conversion reward is fully lost for the entire journey
# including both past and future campaigns

def simulate_budget_roi(df, budget_total, attribution, verbose=False):
    budgets = np.ceil(attribution * (budget_total / np.sum(attribution)))
    
    if(verbose):
        print(budgets)
    
    blacklist = set()
    conversions = set()
    for i in range(df.shape[0]):
        campaign_id = get_campaign_id(df.loc[i]['campaigns']) 
        print('campaign_id')
        print(campaign_id)
        jid = df.loc[i]['jid']
        #print(jid)
        if jid not in blacklist:
            if budgets[campaign_id] >= 1:
                budgets[campaign_id] = budgets[campaign_id] - 1
                if(df.loc[i]['conversion'] == 1):
                    conversions.add(jid)
                #print('jid blacklist')
                #print(jid)
            else:
                blacklist.add(jid)
        
        if(verbose):
            if(i % 10000 == 0):
                print('{:.2%} : {:.2%} budget spent'.format(i/df.shape[0], 1.0 - np.sum(budgets)/budget_total ))
        
        if(np.sum(budgets) < budget_total * 0.02):
            break
            
    return len(conversions.difference(blacklist))


# In[48]:


pitches = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
attributions = [lta, keras_logreg]

for i, pitch in enumerate(pitches):
    for j, attribution in enumerate(attributions):
        reward = simulate_budget_roi(df6, 10000, attribution**pitch)
        print('{} {} : {}'.format(i, j, reward))
        plt.plot(pitches,reward)
        


# In[ ]:




