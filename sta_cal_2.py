from binance_API_USDT import BINANCE
import csv
import time
import requests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import statsmodels.api as sm
import seaborn as sns
from tenacity import *
import urllib3
urllib3.disable_warnings()
s = requests.session()
s.keep_alive = False

headers_list = [
    {
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; SM-G955U Build/R16NW) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (iPad; CPU OS 13_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/87.0.4280.77 Mobile/15E148 Safari/604.1',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.109 Safari/537.36 CrKey/1.54.248666',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.188 Safari/537.36 CrKey/1.54.250320',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (BB10; Touch) AppleWebKit/537.10+ (KHTML, like Gecko) Version/10.0.9.2372 Mobile Safari/537.10+',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (PlayBook; U; RIM Tablet OS 2.1.0; en-US) AppleWebKit/536.2+ (KHTML like Gecko) Version/7.2.1.0 Safari/536.2+',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.3; en-us; SM-N900T Build/JSS15J) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.1; en-us; GT-N7100 Build/JRO03C) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.0; en-us; GT-I9300 Build/IMM76D) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 7.0; SM-G950U Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.84 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; SM-G965U Build/R16NW) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.111 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.1.0; SM-T837A) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.80 Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; en-us; KFAPWI Build/JDQ39) AppleWebKit/535.19 (KHTML, like Gecko) Silk/3.13 Safari/535.19 Silk-Accelerated=true',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.4.2; en-us; LGMS323 Build/KOT49I.MS32310c) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Windows Phone 10.0; Android 4.2.1; Microsoft; Lumia 550) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2486.0 Mobile Safari/537.36 Edge/14.14263',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Moto G (4)) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 10 Build/MOB31T) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 4.4.2; Nexus 4 Build/KOT49H) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; Nexus 5X Build/OPR4.170623.006) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 7.1.1; Nexus 6 Build/N6F26U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; Nexus 6P Build/OPP3.170518.006) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 7 Build/MOB30X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (compatible; MSIE 10.0; Windows Phone 8.0; Trident/6.0; IEMobile/10.0; ARM; Touch; NOKIA; Lumia 520)',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (MeeGo; NokiaN9) AppleWebKit/534.13 (KHTML, like Gecko) NokiaBrowser/8.5.0 Mobile Safari/534.13',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 9; Pixel 3 Build/PQ1A.181105.017.A1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.158 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 10; Pixel 4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 11; Pixel 3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.181 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 5.0; SM-G900P Build/LRX21T) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; Pixel 2 XL Build/OPD1.170816.004) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (iPad; CPU OS 11_0 like Mac OS X) AppleWebKit/604.1.34 (KHTML, like Gecko) Version/11.0 Mobile/15A5341f Safari/604.1',
        'content-type': 'application/json'
    }
]

@retry(stop=stop_after_delay(15))
def get_info(symbol = 'BTCUSDT',limit = 1500):
    global binance
    binance = BINANCE()
    interval = "15m"
    #limit = "73"
    output = []
    #klines = []
    title_name = ['date', 'open', 'high', 'low', 'close']
    endTime = int(time.time()*1000)
    while limit-1500 > 0:
        klines = []
        number = 0
        limit = limit-1500
        number = number+1500
        body = {
            "symbol": symbol,
            "interval": interval,
            "limit": number,
            "endTime": endTime
        }
        respond = binance.IO('GET', '/fapi/v1/klines', body)
        for i in range(len(respond)):
            kline = {}
            kline['date'] = respond[i][0]
            kline['open'] = respond[i][1]
            kline['high'] = respond[i][2]
            kline['low'] = respond[i][3]
            kline['close'] = respond[i][4]
            klines.append(kline)
        endTime = 2*klines[0]['date'] - klines[1]['date']
        klines = list(reversed(klines))
        output.extend(klines)
    if limit > 0:
        klines = []
        body = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "endTime": endTime
        }
        respond = binance.IO('GET', '/fapi/v1/klines', body)
        for i in range(len(respond)):
            kline = {}
            kline['date'] = respond[i][0]
            kline['open'] = respond[i][1]
            kline['high'] = respond[i][2]
            kline['low'] = respond[i][3]
            kline['close'] = respond[i][4]
            klines.append(kline)
        klines = list(reversed(klines))
        output.extend(klines)
        output = list(reversed(output))
    with open('kline_data.csv', 'w', encoding='utf-8', newline='') as file_obj:
        # 1.创建DicetWriter对象
        dictWriter = csv.DictWriter(file_obj, title_name)
        # 2.写表头
        dictWriter.writeheader()
        # 3.写入数据(一次性写入多行)
        dictWriter.writerows(output)

def minus(symbols,i):
    for j in range(i+1,len(symbols)):
        data_symbol_minus_list = []
        for n in range(len(data[symbols[i]])):
            data_symbol_minus_list.append(data[symbols[i]][n] - data[symbols[j]][n])
        avg = np.ones(len(data_symbol_minus_list))*np.average(data_symbol_minus_list)
        var = np.ones(len(data_symbol_minus_list))*np.var(data_symbol_minus_list, ddof = 1)
        data_symbol_minus[f'{symbols[i]} - {symbols[j]}'] = data_symbol_minus_list
        plt.clf()
        plt.plot(data_symbol_minus_list)
        plt.plot(avg,"-")
        plt.plot(var,"*")
        plt.savefig(f'D:\\学校文件\\Python\\fig\\{symbols[i]} - {symbols[j]}.png')
        #plt.pause(0.2)

binance = BINANCE()
exchanges_info = binance.IO('GET','/fapi/v1/exchangeInfo',{})

symbols = []
'''print("Check symbols")
for i in range(len(exchanges_info['symbols'])):
    if (exchanges_info['symbols'][i]['status'] == 'TRADING') and (exchanges_info['symbols'][i]['symbol'][-4:] != 'BUSD') :
        symbols.append(exchanges_info['symbols'][i]['symbol'])
print("Symbols check finished")
symbols = symbols.copy()
print(symbols)
print(f"total calculate times = {len(symbols)*(len(symbols)-1)/2}")'''

data = {}
data_matric = []
data_org = {}
kline_num = 2880
'''for i in range(len(symbols)):
    print(f'{round(i/len(symbols)*100,3)}%')
    get_info(symbols[i], kline_num)
    data_symbol = pd.read_csv('kline_data.csv', index_col=0, encoding='gb2312') # gb2312
    data_symbol_close = data_symbol['close'].copy()
    if len(data_symbol_close) < kline_num:
        break
    data_symbol_close_rate = []
    data_symbol_close_rate2one = []
    for j in range(len(data_symbol_close)):
        if j == 0:
            data_symbol_close_rate.append(0.0)
        else:
            data_symbol_close_rate.append((data_symbol_close[data_symbol_close.index[j]]-data_symbol_close[data_symbol_close.index[j-1]])
                                          / data_symbol_close[data_symbol_close.index[j-1]] + data_symbol_close_rate[-1])
    #归一化
    j = 0
    for j in range(len(data_symbol_close_rate)):
        Max = max(data_symbol_close_rate)
        Min = min(data_symbol_close_rate)
        if Max-Min == 0 :
            print(data_symbol_close_rate)
            print(data_symbol_close)
            print(symbols[i])
        mean = sum(data_symbol_close_rate) / len(data_symbol_close_rate)
        Max1 = np.max(np.abs(data_symbol_close_rate))
        data_symbol_close_rate2one.append((data_symbol_close_rate[j] - mean) / Max1)
    if len(data_symbol_close_rate2one) == kline_num:
        data[symbols[i]] = data_symbol_close_rate2one
        data_org[symbols[i]] = list(data_symbol_close.values)
        data_matric.append(data_symbol_close_rate2one)
    time.sleep(3)'''

#plt.ion()

data_symbol_minus = {}
print('=================================================')

#print(data_symbol_minus)
#plt.ioff()
'''title_name = symbols
with open('kline_data_symbol_close_rate2one.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # 1.创建DicetWriter对象
    dictWriter = csv.DictWriter(file_obj, title_name)
    # 2.写表头
    dictWriter.writeheader()
    # 3.写入数据(一次性写入多行)
    dictWriter.writerows(data)
with open('kline_data_org.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # 1.创建DicetWriter对象
    dictWriter = csv.DictWriter(file_obj, title_name)
    # 2.写表头
    dictWriter.writeheader()
    # 3.写入数据(一次性写入多行)
    dictWriter.writerows(data_org)'''

#df_data = pd.DataFrame(data).copy()
#df_data_org = pd.DataFrame(data_org).copy()
df_data = pd.read_csv('kline_data_symbol_close_rate2one.csv', index_col=0, encoding='gb2312') # gb2312
df_data_org = pd.read_csv('kline_data_org.csv', index_col=0, encoding='gb2312') # gb2312
symbols = list(df_data.columns)
data = df_data.to_dict('list')
data_matric = df_data.values
data_org = df_data.to_dict('list')
df_data_eul = []
df_data_corrlation = []
df_data_r2 = []
df_data_a = []
df_data_p = []
columns_value = df_data.columns.values
'''#计算欧氏距离
for i in range(len(columns_value)):
    column = []
    for j in range(len(columns_value)):
        sy_i_minus_sy_j = np.sqrt(np.sum((df_data[columns_value[i]] - df_data[columns_value[j]]) ** 2))
        column.append(sy_i_minus_sy_j)
    df_data_eul.append((column - np.mean(column)) / max(column))
df_data_eul = pd.DataFrame(df_data_eul,index=columns_value,columns=columns_value).copy()
#计算相关系数
for i in range(len(columns_value)):
    column = []
    for j in range(len(columns_value)):
        a_diff = df_data[columns_value[i]] - np.mean(df_data[columns_value[i]])
        p_diff = df_data[columns_value[j]] - np.mean(df_data[columns_value[j]])
        numerator = np.sum(a_diff * p_diff)
        denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
        sy_i_minus_sy_j = numerator / denominator
        column.append(sy_i_minus_sy_j)
    df_data_corrlation.append(column)
df_data_corrlation = pd.DataFrame(df_data_corrlation,index=columns_value,columns=columns_value).copy()
#线性拟合并计算R2和斜率a
index = df_data.index
for i in range(len(columns_value)):
    column_r2 = []
    column_a = []
    for j in range(len(columns_value)):
        slope, intercept, r_value, p_value, std_err = stats.linregress(index, df_data[columns_value[i]] - df_data[columns_value[j]])
        column_r2.append(r_value**2)
        if slope != 0:
            column_a.append(-slope)
        else:
            column_a.append(-slope)
    df_data_r2.append(column_r2)
    df_data_a.append((column_a - np.mean(column_a)) / max(column_a))
#df_data_r2 = pd.DataFrame(df_data_r2, index=columns_value, columns=columns_value).copy()
df_data_a = pd.DataFrame(df_data_a, index=columns_value, columns=columns_value).copy()
# 相减后方差计算
var = 1/np.var(df_data[columns_value[0]] - df_data[columns_value[1]])'''
#计算协整 p 值并记录
for i in range(len(columns_value)):
    column_p = []
    print(f'{round((i+1) / len(columns_value) * 100, 3)}%')
    for j in range(len(columns_value)):
        result = sm.tsa.stattools.coint(np.reshape(df_data_org[columns_value[i]],-1),np.reshape(df_data_org[columns_value[j]],-1))
        column_p.append(-result[1])
    df_data_p.append(column_p)
df_data_p = pd.DataFrame(df_data_p,index=columns_value,columns=columns_value)
print(df_data_p)
df_data_p.to_csv("df_data_p.csv",sep=',')
#corr_matric = np.array(df_data_eul)
#corr_matric = np.array(df_data_r2)
#corr_matric = np.array(df_data_a)
corr_matric = np.array(df_data_p)
'''corr_matric = 18.3556 * np.array(df_data_eul) + 13.6656 * np.array(df_data_corrlation) + 2.8558 * np.array(df_data_r2) - \
              5.5576 * var + 957.7492 * np.array(df_data_a) + 13.974'''
correlate = {}
for i in range(len(corr_matric)):
    for j in range(i+1,len(corr_matric)):
        if correlate == {} :
            correlate[f"{symbols[i]}|{symbols[j]}"] = corr_matric[i][j]
        else:
            if len(correlate.keys()) < 10:
                for n in range(len(correlate.keys())):
                    if corr_matric[i][j] > correlate[list(correlate.keys())[n]]:
                    #if corr_matric[i][j] > -0.05 :
                        correlate[f"{symbols[i]}|{symbols[j]}"] = corr_matric[i][j]
                        break
            elif len(correlate) == 10:
                isBreak = False
                for n in range(5):
                    if corr_matric[i][j] > correlate[list(correlate.keys())[n]]:
                    #if corr_matric[i][j] > -0.05:
                        value = min(list(correlate.values()))
                        for m in range(len(correlate.keys())):
                            if correlate[list(correlate.keys())[m]] == value:
                                del correlate[list(correlate.keys())[m]]
                                break
                        correlate[f"{symbols[i]}|{symbols[j]}"] = corr_matric[i][j]
                        break
print('\n')
print(correlate)

'''for i in range(len(corr_matric)):
    for j in range(i+1,len(corr_matric)):
        if corr_matric[i][j][0] < corr_matric[i][j][2][0]:
            correlate[f"{symbols[i]}|{symbols[j]}"] = corr_matric[i][j]
            break
print('\n')
print(correlate)'''

print('\n=================================================\n')
correlate_keys = list(correlate.keys())
symbol_pairs = []
for i in range(len(correlate_keys)):
    symbol1 = correlate_keys[i].split('|')[0]
    symbol2 = correlate_keys[i].split('|')[1]
    symbol_pairs.append([symbol1,symbol2])
print(symbol_pairs)
print('\n')

for i in range(len(symbol_pairs)):
    print(f'{round(i / len(symbol_pairs) * 100, 3)}%')
    data_symbol_minus_list = []
    for n in range(len(data[symbol_pairs[i][0]])):
        data_symbol_minus_list.append(data[symbol_pairs[i][0]][n] - data[symbol_pairs[i][1]][n])
    avg = np.ones(len(data_symbol_minus_list)) * np.average(data_symbol_minus_list)
    var = np.ones(len(data_symbol_minus_list)) * np.var(data_symbol_minus_list, ddof=1)
    data_symbol_minus[f'{symbol_pairs[i][0]} - {symbol_pairs[i][1]}'] = data_symbol_minus_list
    plt.clf()
    plt.plot(data_symbol_minus_list)
    plt.plot(data[symbol_pairs[i][0]])
    plt.plot(data[symbol_pairs[i][1]])
    sns.heatmap(df_data_p,camp='Blues',annot=True)
    #plt.plot(avg, "-")
    #plt.plot(var, "*")
    plt.savefig(f'D:\\学校文件\\Python\\fig\\{symbol_pairs[i][0]} - {symbol_pairs[i][1]}.png')

    print(f"最大 {symbol_pairs[i][0]} - {symbol_pairs[i][1]} = {max(abs(max(data_symbol_minus_list)),abs(min(data_symbol_minus_list)))}")
    print(f"AVG {symbol_pairs[i][0]} - {symbol_pairs[i][1]} = {avg[0]}")
    print(f"VAR {symbol_pairs[i][0]} - {symbol_pairs[i][1]} = {var[0]}")
    result = sm.tsa.stattools.coint(df_data_org[symbol_pairs[i][0]], df_data_org[symbol_pairs[i][1]])
    print(f"result = {result}\n")
    # plt.pause(0.2)
plt.show()

