import pymysql
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import Counter
from utils.common import *

class Preprocessing:
    def __init__(self, dapp_name, dapp_addrs):
        self.dapp_name = dapp_name
        self.dapp_addresses = dapp_addrs
        
    def run(self):
        tx = self.get_TX_by_DappName()
        userAddrs, contractAddrs = self.getEOAsOfGame(tx, self.dapp_addresses)

        for user in userAddrs:
            related_TXs = [i for i in tx if i[3] == user or i[5] == user]
            txsFromUser = getTXsByAddress(user)
            periodTimes = self.getPeriodFromTXs(related_TXs)
            TXsInPeriodTimes = self.getTXsInPeriodTimes(txsFromUser, periodTimes)
        self.saveExtendTXsToDB(TXsInPeriodTimes)
    
    def get_TX_by_DappName(self):
        query = "SELECT td.*, tt.trace_graph FROM tx_origin_new AS td " + \
                "LEFT JOIN tx_trace AS tt ON td.tx_hash = tt.tx_hash " + \
                "WHERE td.dapp_name = %s"
        data = query_db(query, (self.dapp_name,))
        return data
    
    def getPeriodFromTXs(self, related_TXs):
        date_format = "%d/%m/%Y %H:%M:%S"
        periodTime = []
        
        for tx in related_TXs:
            date_str = tx[1]
            dt_object = datetime.strptime(date_str, date_format)
            
            min_date = dt_object - timedelta(days = 7)
            min_date = min_date.replace(tzinfo = timezone.utc)
            
            max_date = dt_object + timedelta(days = 7)
            max_date = max_date.replace(tzinfo = timezone.utc)
            
            periodTime.append((tx[0], int(min_date.timestamp()), int(max_date.timestamp())))
        
        return  periodTime
    
    def getEOAsOfGame(list_tx, dapp_addrs):
        userAddrs = set()
        contractAddrs = set()
        
        for tx_detail in list_tx:
            # print(tx_detail)
            from_addr = tx_detail[3]
            to_addr = tx_detail[5]
            
            if from_addr not in dapp_addrs:
                if tx_detail[4] == "user":
                    userAddrs.add(from_addr)
                elif tx_detail[4] in ['contract', 'self-destruct contract']:
                    contractAddrs.add(from_addr)
            
            if to_addr not in dapp_addrs:
                if tx_detail[6] == "user":
                    userAddrs.add(to_addr) 
                elif tx_detail[6] in ['contract', 'self-destruct contract']:
                    contractAddrs.add(to_addr)
                
        return userAddrs, contractAddrs
    
    def getTXsInPeriodTimes(self, TXs, periodTimes):
        result = []
        
        TXs_timestamp_arr = np.array([int(tx['timeStamp']) for tx in TXs], dtype = int)
        for period in periodTimes:
            start = np.searchsorted(TXs_timestamp_arr, period[1])
            end = np.searchsorted(TXs_timestamp_arr, period[2], side = 'right')
            for tx in TXs[start:end]:
                tx['seed'] = period[0] # tx_hash
                result.append(tx)
                
        duplicated_TXs = [item for item, count in Counter([tx['hash'] for tx in result]).items() if count > 1]
        for tx_hash in duplicated_TXs:
            dupSeedTxs = [tx for tx in result if tx['hash'] == tx_hash]
            tx = dupSeedTxs[0]
            
            tx['seed'] = ", ".join(set([tx['seed'] for tx in dupSeedTxs]))
            result = [tx for tx in result if tx['hash'] != tx_hash]
            result.append(tx)
                
        print(f"- Timestamp from {period[1]} to {period[2]} have {len(result)} transactions")        
        return result
    
    def saveExtendTXsToDB(self, TXs_extend):
        db = pymysql.connect(host = host_db, user = user_db, db = name_db)
        cursor = db.cursor()
        query = "INSERT INTO tx_extend(tx_hash,timestamp,from_addr,to_addr,from_addr_type,to_addr_type,tx_status,tx_method,input,value,contractAddress,graph_distance,date_distance,game_addr,similarity) " + \
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                
        for tx in TXs_extend:
            param = (
                tx["hash"], tx["timeStamp"],
                tx["from"], tx["to"], get_address_type(tx["from"]), get_address_type(tx["to"]),
                tx["isError"], tx["methodId"],
                tx["input"], tx["value"],
                tx["contractAddress"],
                None,None,None,None
            )
            # print(param)
            cursor.execute(query, param)
            db.commit()
        
        db.close()

# Test functions   
if __name__ == "__main__": 
    dapp_name = "lastwinner"
    
    query = "SELECT address FROM dapp_info WHERE name = %s"
    dapp_addrs = [i[0] for i in query_db(query, (dapp_name,))]
    
    prepr = Preprocessing(dapp_name, dapp_addrs)
    prepr.run()