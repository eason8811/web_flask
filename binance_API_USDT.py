import json
import re
import xlwt
import requests
import time
import hashlib
import hmac
import sys
import os
import shutil
import xlrd
import csv

s = requests.session()
s.keep_alive = False


def create_sheet(f, table_name, try_count):
    if try_count <= 1:
        return
    try:
        sheet1 = f.add_sheet(table_name + '-' + str(11 - try_count), cell_overwrite_ok=True)
        return sheet1
    except Exception as e:
        return create_sheet(f, table_name, try_count - 1)


def create_a_new_excel(excel_path, table_name, title_name, use_lists):
    table1_invalid_start_x = 1;
    table1_invalid_start_y = 2;
    max_buf_len = []
    if os.path.exists(excel_path):
        os.remove(excel_path)
    f = xlwt.Workbook(encoding='utf-8')  # 新建excel

    font = xlwt.Font()
    font.bold = True
    borders = xlwt.Borders()
    borders.left = xlwt.Borders.THIN
    borders.right = xlwt.Borders.THIN
    borders.top = xlwt.Borders.THIN
    borders.bottom = xlwt.Borders.THIN
    alignment = xlwt.Alignment()
    alignment.horz = xlwt.Alignment.HORZ_CENTER  # 水平方向
    alignment.vert = xlwt.Alignment.VERT_TOP
    style1 = xlwt.XFStyle()
    style1.font = font
    # style1.borders = borders
    style1.alignment = alignment

    style2 = xlwt.XFStyle()
    style2.alignment.wrap = 1  # 自动换行

    try:
        sheet1 = f.add_sheet(table_name, cell_overwrite_ok=True)
    except Exception as e:
        sheet1 = create_sheet(f, table_name, 10)

    for item in range(0, len(title_name)):
        sheet1.write(1, item + table1_invalid_start_x, title_name[item], style=style1)
        max_buf_len.append(len(title_name[item]))
        sheet1.col(1).width = 256 * (max_buf_len[item])

    item = 0
    column_len = len(title_name)
    for use_list in use_lists:
        for column_index in range(0, column_len):
            if len(use_list[title_name[column_index]]) > max_buf_len[column_index]:
                max_buf_len[column_index] = len(use_list[title_name[column_index]])
            if max_buf_len[column_index] > 150:
                max_buf_len[column_index] = 150
            sheet1.col(column_index + table1_invalid_start_x).width = 256 * (max_buf_len[column_index] + 3)
            sheet1.write(table1_invalid_start_y + item, column_index + table1_invalid_start_x,
                         use_list[title_name[column_index]], style2)
        item += 1
    f.save(excel_path)

def dict2csv(dic, filename):
    """
    将字典写入csv文件，要求字典的值长度一致。
    :param dic: the dict to csv
    :param filename: the name of the csv file
    :return: None
    """
    file = open(filename, 'a', encoding='utf-8', newline='')
    csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
    csv_writer.writeheader()
    '''for i in range(len(dic[list(dic.keys())[0]])):   # 将字典逐行写入csv
        dic1 = {key: dic[key][i] for key in dic.keys()}
        csv_writer.writerow(dic1)'''
    csv_writer.writerow(dic)
    file.close()

apikey = 'i7F6rx3Tcz6eJlSVzBc4dpV6qyszCiCOIpSz7gv9mdyq9UjVizrlu2kkmlvUIJSw'
Secret_KEY = 'mwU7KCworFZ17WIOqRuGaRmtwT3nnUDBhtg8HQf9CHFB7KVSxev0Rwym5mgfWjDx'

# 币安api接口
class BINANCE:
    def param2string(self,param):
        s = ''
        for k in param.keys():
            s += k
            s += '='
            s += str(param[k])
            s += '&'
        return s[:-1]

    def IO(self,method,request_path,body):
        header = {
        'Connection': 'close',
        'X-MBX-APIKEY': apikey,
        }
        if body != '':
            body['signature'] = hmac.new(Secret_KEY.encode('utf-8'), binance.param2string(body).encode('utf-8'), hashlib.sha256).hexdigest()
            if method == 'GET':
                body = binance.param2string(body)
                #tell = 'https://fapi.binance.com{0}?{1}'.format(request_path,body)
                response = requests.get(url=f'https://fapi.binance.com{str(request_path)}',
                                        headers=header,params=body,verify=False).json()                      #GET方法
                return response
            elif method == 'POST':
                response = requests.post(url=f'https://fapi.binance.com{str(request_path)}',
                                         headers=header, data=body).json()                      #POST方法
                return response
            elif method == 'DELETE':
                response = requests.delete(url=f'https://fapi.binance.com{str(request_path)}',
                                           headers=header,params=body).json()                   #DELETE方法
                return response
        else:
            response = requests.get(url=f'https://fapi.binance.com{str(request_path)}',
                                    headers=header).json()
            return response

global binance
binance = BINANCE()

#symbol = ['BTCUSDT','ETHUSDT','BNBUSDT','LTCUSDT','EOSUSDT','ATOMUSDT','IOTXUSDT','XRPUSDT'] # len = 8
symbol = ['BTCUSDT']
interval = "5m"
limit = "1000"

result = []
t = time.time()
body = {

}

res = binance.IO('GET','/fapi/v1/ticker/price',body)
#print(res)

'''body = {
    'symbol' : 'BTCUSDT',
    'interval' : '5m',
    'startTime' : str(1579503838*1000),
    'limit' : str(10),
}

res = binance.IO('GET','/fapi/v1/klines',body)
print(res)
'''
'''#获取历史K线
for i in range(len(symbol)):
    body = {
        "symbol":symbol[i],
        "interval":interval,
        'endTime':str(1682608436000),
        "limit":limit
    }
    info = binance.IO('GET', '/fapi/v1/klines', body)
    onesymbolresult = []
    for j in range(len(info)):
        klinemodule = {
            "date": 0,
            "close": 0,
            "high": 0,
            "low": 0,
            "open": 0,
            "valume": 0
        }
        klinemodule["date"] = str(info[j][0])
        klinemodule["open"] = info[j][1]
        klinemodule["high"] = info[j][2]
        klinemodule["low"] = info[j][3]
        klinemodule["close"] = info[j][4]
        klinemodule["valume"] = info[j][5]
        onesymbolresult.append(klinemodule)
    result.append(onesymbolresult)

title_name = ["date","close","high","low","open","valume"]

file = open('crypto_data.csv', 'a', encoding='utf-8', newline='')
csv_writer = csv.DictWriter(file, fieldnames=list(result[0][i].keys()))
csv_writer.writeheader()
for i in range(len(result[0])):
    csv_writer.writerow(result[0][i])
    #print(result[0][i])
file.close()'''

'''
#GET请求示例:查看余额
body = {"timestamp": f"{int(time.time() * 1000)}"}
info = binance.IO('GET','/fapi/v2/balance',body)
print(info)

#POST请求示例:调整杠杆倍数
body = {
    "symbol":"BTCUSDT",
    "leverage":int(10),
    "timestamp": f"{int(time.time() * 1000)}"
}
info  = binance.IO('POST','/fapi/v1/leverage',body)
print(info)'''


