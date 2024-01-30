import cloudscraper
import time
import pymysql
from web3 import Web3
from bs4 import BeautifulSoup

host_db = "172.23.96.1" # "localhost"
user_db = "discord-user"
name_db = 'dapp_analysis_rearrange'
password = "vlvlvlvl123"

ETHSCAN_API_KEY = "66PEZRV8MGWR8VRS6BCNX7VM7BNW3GYXUJ"
ETHSCAN_URL = f"http://api.etherscan.io/api?&apikey={ETHSCAN_API_KEY}&"

def create_db_connection():
    db = pymysql.connect(host = host_db, user = user_db, db = name_db, password = password)
    cursor = db.cursor()
    return db, cursor

# Just a function to get data in database
def query_db(query, args = None):
    try:
        db = pymysql.connect(host = host_db, user = user_db, db = name_db, password = password)
        cursor = db.cursor()
        cursor.execute(query, args)
        result = cursor.fetchall()
        
    except:
        db.rollback()
        result = None
    
    db.close()
    return result

# Determine the type of an Ethereum address.
def get_address_type(w3, address):
    # Check if the address length is correct
    if len(address) != 42:
        return 'Unknown'
    
    # Get the bytecode of the address
    code = w3.eth.get_code(Web3.to_checksum_address(address))
    lenght_code = len(code)
    
    if lenght_code > 1:
        return "contract"
    
    url = "https://etherscan.io/address/" + address + "#code"
    scraper = cloudscraper.create_scraper()
    
    resp = None
    # Retry until a successful response is received
    while True:
        resp = scraper.get(url)
        if resp.status_code != 200:
            time.sleep(15) # maybe meet rate limit
        else:
            break
    
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    
    # Check for elements indicating a self-destruct contract
    contract_element = soup.find_all("li", id = "ContentPlaceHolder1_li_contracts")
    if len(contract_element) > 0:
        return 'self-destruct contract'
    
    # Check for elements indicating an exchange
    exchange_element = soup.find("div", id = "ContentPlaceHolder1_divLabels").text
    if "Exchange" in exchange_element:
        return "exchange"
    
    # Return an user type if address do not meet any types above
    return "user"

def getTXsByAddress(addr, start_block = 0, end_block = 99999999):
    scraper = cloudscraper.create_scraper()
    url_req = ETHSCAN_URL + f"module=account&action=txlist&address={addr}&startblock={start_block}&endblock={end_block}&page=1&offset=10&sort=asc&page="
    
    result = []
    for page in range(1, 5):
        req = scraper.get(url_req + str(page))
        resp = req.json()
        if resp['message'] != "OK":
            break
        
        for tx in resp['result']:
            result.append(tx)
        return result