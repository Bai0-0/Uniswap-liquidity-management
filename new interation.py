
# %% data processing
import numpy as np
import pandas as pd
df = pd.read_csv(r'C:\Users\wenyu\Desktop\USDC-WETH.csv')
df=df['tick']
#df=np.log(df)
train_df = df.iloc[0:6000]
test_df = df.iloc[6001:len(df)]
info = df.describe()
train_var = np.sqrt(np.var(df))
train_mean = np.mean(df)

# %%
#calculate how much token be swap,return token_delta[alpha_delta,beta_delta]
def cal_swap(old_price, new_price, range_low, range_high, a_r, b_r):

    if(new_price<old_price):
        if(old_price<=range_low):
            token_delta = [0,0]
            return token_delta
        if(old_price>=range_high):

            if(new_price>=range_high):
                token_delta = [0,0]
                return token_delta

            else:
                L=a_r/(np.sqrt(range_high)-np.sqrt(range_low)) #Liquidity
                new_price=max(new_price,range_low)
                delta_alpha = (np.sqrt(new_price) - np.sqrt(range_high))*L
                delta_beta = (1/np.sqrt(new_price) - 1/np.sqrt(range_high))*L
                token_delta = [delta_alpha,delta_beta]
                return token_delta
        else:
            new_price=max(new_price,range_low)
            delta_alpha=(np.sqrt(new_price) - np.sqrt(old_price))*a_r/(np.sqrt(old_price)-np.sqrt(range_low))
            delta_beta = (1/np.sqrt(new_price) - 1/np.sqrt(old_price))*b_r/(1/np.sqrt(old_price)-1/np.sqrt(range_high))
            token_delta = [delta_alpha,delta_beta]
            return token_delta
    
    else:

        if(old_price>=range_high):
            token_delta = [0,0]
            return token_delta    
        if(old_price<=range_low):
        
            if(new_price<=range_low):
                token_delta = [0,0]
                return token_delta 
        
            else:
        
                L=b_r/(1/np.sqrt(range_low)-1/np.sqrt(range_high))
                new_price=min(new_price,range_high)
                delta_alpha = (np.sqrt(new_price) - np.sqrt(range_low))*L
                delta_beta = (1/np.sqrt(new_price) - 1/np.sqrt(range_low))*L
                token_delta = [delta_alpha,delta_beta]
                return token_delta
        
        else:
        
            new_price=min(new_price,range_high)
            delta_alpha=(np.sqrt(new_price) - np.sqrt(old_price))*a_r/(np.sqrt(old_price)-np.sqrt(range_low))
            delta_beta = (1/np.sqrt(new_price) - 1/np.sqrt(old_price))*b_r/(1/np.sqrt(old_price)-1/np.sqrt(range_high))
            token_delta = [delta_alpha,delta_beta]
            return token_delta
        
   
#Update the amount of reserved tokens
#swap is token_delta
#cp is current price
def meet_wealth(swap,cp,transaction_fee,fee_rate,a_r,b_r):
    a_r += swap[0]
    b_r +=swap[1]

    if(swap[1]<=0) :
    
        transaction_fee += abs(swap[0]) * fee_rate
    
    else:
    
        transaction_fee += abs(swap[1])
    return a_r,b_r,transaction_fee
    
# %%
a = np.arange(0.05,4,0.05)#price range,i.e 0.5 times of price volatility
b = 2*a #break_out_interval is 2 times of a
gas = 100
fee_rate = 0.003

data = {'range':a,
        'break range':b,
        'wealth':np.zeros(len(a)),
        'gas cost':np.zeros(len(a)),
        'reset_times':np.zeros(len(a)),
        'transaction fee':np.zeros(len(a)),
        'pnl':np.zeros(len(a)),
        'APR':np.zeros(len(a))
        }
result = pd.DataFrame(data)

for i in range(0,len(a)):  
    cp = train_df[0]
    price_range = [cp-a[i]*train_var,cp+a[i]*train_var]
    break_range = [cp-b[i]*train_var,cp+b[i]*train_var]
    
    a_r = 35000
    b_r=a_r*(np.sqrt(1/cp)-np.sqrt(1/price_range[1]))/(np.sqrt(cp)-np.sqrt(price_range[0]))
    wealth = a_r + b_r *cp
    ini_wealth = wealth

    gas_cost = 0
    wealth = 0
    pnl = 0
    transaction_fee = 0
    apr = 0
    reset_times = 0  #when reset breakout_interval
    #price_out_times = 0
   
    for j in range(1,len(train_df)):
        op = cp
        cp = train_df[j] #update old_price and new_price
        swap = cal_swap(op,cp,price_range[0],price_range[1],a_r, b_r)
        a_r, b_r, transction_fee = meet_wealth(swap, cp,transaction_fee,fee_rate,a_r,b_r)
        wealth = a_r + b_r*cp

        if cp > break_range[1] or cp < break_range[0]:
            #reset price range based on current price
            price_range = [cp-a[i]*train_var,cp+a[i]*train_var]
            break_range = [cp-b[i]*train_var,cp+b[i]*train_var]
            reset_times+=1
            gas_cost += gas

            #add liquidity based on current price and my wealth 
            temp = cp + (np.sqrt(cp)-np.sqrt(1/price_range[1]))/(np.sqrt(1/cp)-np.sqrt(1/price_range[0]))
            b_r = wealth/temp
            a_r = b_r * (np.sqrt(cp)-np.sqrt(1/price_range[1]))/(np.sqrt(1/cp)-np.sqrt(1/price_range[0]))

    pnl=wealth+transaction_fee-gas_cost-ini_wealth
    result.iloc[i, 2] = wealth
    result.iloc[i, 3] = gas_cost
    result.iloc[i, 4] = reset_times
    result.iloc[i, 5] = transaction_fee
    result.iloc[i, 6] = pnl
    result.iloc[i, 7] = pnl/ini_wealth

    result.to_csv('result.csv')
    

            


      





# %%
