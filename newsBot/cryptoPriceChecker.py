import requests
import pandas as pd
import os
import configparser
import smtplib
from email.mime.text import MIMEText




config = configparser.ConfigParser()
config.read(os.environ['PYTHONPATH'] + "/config/prod.env")
api_key = config["COINMARKETCAP"]["api_key"]
url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest'


# Specify the ids of the symbols in your watchlist 
# -------- {Symbol:Id} -----------------
watchlist_symbols_dict = {'GMT': '18069','SILLY': '28789', 'AURY': '11367', 'DIO': '3908', 'GENE': '13632',
                     'OTK': '24381', 'DFL': '10294', 'MBS': '13011', 'MCRT': '15721', 'CWAR': '12722', 
                     'WLKN': '18775', 'FCON': '14904', 'ELU': '16569', 'TAKI': '19362', 'HXD': '23707', 'SNS': '17231'} 

# must be a string of the ids separated by commas
parameters = {'id' : ','.join(watchlist_symbols_dict.values())}

headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': api_key}

# Make the API request
response = requests.get(url, headers=headers, params=parameters)
data = response.json()

if response.status_code == 200:
    # Extract relevant data from the 'data' field in the response
    all_cryptos = data['data']

    df = pd.DataFrame(all_cryptos)
    df = df.transpose()

    columns_to_include = ['id', 'name', 'symbol', 'quote']

    df = df[columns_to_include]
    df = pd.concat([df, df["quote"].apply(pd.Series)['USD'].apply(pd.Series)], axis=1)
    df = df[['id','name','symbol','price','percent_change_1h','percent_change_24h','last_updated']]
    df['last_updated'] = pd.to_datetime(df['last_updated']).dt.strftime("%a, %b %d | %I:%M %p")

    # Use or print the data from the watchlist
    df.to_excel('coinmarketcap_watchlist_data.xlsx', index=False)
else:
    print(f'Error: {data["status"]["error_message"]}')



subject = "Crypto Price Checker"
body = df.to_html()
sender = "nikolaislavev96@gmail.com"
recipients = ["nikolayslavev96@gmail.com"]
password = "zmcr zjmg xvkj sptv"


msg = MIMEText(body, "html")
msg['Subject'] = subject
msg['From'] = sender
msg['To'] = ', '.join(recipients)
with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
    smtp_server.login(sender, password)
    smtp_server.sendmail(sender, recipients, msg.as_string())
    
print("Message sent!")