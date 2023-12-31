import requests
import json
import os
import time
import configparser
from chatGpt import chatGpt
import configparser
from datetime import datetime
from newsBot.discordDbUtils import *



msg = """<:DFL_heart:982311606266511390> GM DeFi Landers, @everyone

Hope you are starting a week very well. As promised, we are beginning to announce several important things as 2023 is ending.

**Today we bring to you a DFL trading competition with $10,000 worth of prizes!**

We have partnered up with one of the biggest Asia based centralized exchanges, Gate.io, to incentivize people trading DFL! If you trade our currency on their exchange, starting from today to 26th of December (1 week), you will receive rewards based on your volume. Additionally, if you refer a friend to Gate.io you can win additional prizes, so let's help each other out. One thing, you will need to KYC on their exchange. Gate.io is one of the oldest and most trusted exchanges out there, but make sure to do your own research. Learn more details about competition here: https://go.gate.io/w/rGS483vb

**IMPORTANT** - Keep in mind that, our currency on Gate is listed under ticker of **DEFILAND** and not DFL. Please make sure that you are trading DEFILAND/USDT or DEFILAND/ETH pairs as those are the only eligible pairs on Gate.io.

We are taking DFL and it's future very serious and we will deliver more and more exciting events around it! Let's share this news with everyone and slowly but surely make DFL great again! https://fxtwitter.com/gate_io/status/1736997365746171923
"""


class DiscordNews():
    chatGptPromptMsg = """Give me yes or no answers and why you think so in 1 sentence to the questions below. The questions are regarding the post labeled "POST/". 
                    1/ Does it mention anything positive?
                    2/ Is the post not about the past?
                    3/ Does it mention a listing or partnership with a centralized exchange?
                    4/ Does it mention a Venture Capital funding or investment?
                    5/ Does it mention a token staking announcement?
                    6/ Does it mention a game release announcement?
                    7/ Does it mention an epic store or steam announcement?
                    8/ Does it mention a listing on Binance, Coinbase or Crypto.com?
                    9/ Does it mention the use of Immutable X?
                    10/ Does it mention a partnership with a well-known company?
                    POST/ """
    
    def __init__(self):        
        self.config = configparser.ConfigParser()
        self.config.read(os.environ['PYTHONPATH'] + "/config/prod.env")
        self.authorization = self.config["DISCORD"]["authorization"]
        self.discordNewsInfo = getDiscordNewsInfo(self.config)
        
    def execute(self):
        id = 0
        success = False
        while True:
            channelRow = self.discordNewsInfo.iloc[id]
            print(channelRow["channel_name"])
            reqText = DiscordNews.retrieveMsg(channelRow["channel_id"], self.authorization)
            reqText["timestamp_new"] = datetime.strptime(reqText["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").replace(microsecond=0).replace(tzinfo=None)
            discordNewsDate = reqText["timestamp_new"].strftime('%Y-%m-%d %H:%M:%S')
            discordReadDate = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if (reqText['timestamp_new'] != channelRow["last_news_date"].to_pydatetime()):
                updateDiscordNewsInfo(channelRow["channel_id"], reqText["timestamp_new"], self.config)
                self.discordNewsInfo.loc[id, "last_news_date"] = reqText['timestamp_new']
            
                print(f"CONTENT: {reqText['content']}")
                print(f"CONTENT CREATED: {reqText['timestamp']}")

                for i in range(3):
                    response = chatGpt(self.chatGptPromptMsg + reqText["content"])
                    print(response)

                    if (response.count("Yes") == 3):
                        with open(os.environ['PYTHONPATH'] + "/newsBot/newsBot.txt", "a") as file:
                            file.write(f"\n{self.discordNewsInfo.loc[id, 'token_name']},{self.discordNewsInfo.loc[id, 'token_id']},{discordReadDate},{discordNewsDate}")
                        
                        success = True
                        break
                
                if success:
                    break

            id = id + 1 if id + 1 != len(self.discordNewsInfo) else 0
            time.sleep(10)

    def retrieveMsg(channelId, authorization):
        req = requests.get(f"https://discord.com/api/v9/channels/{channelId}/messages?limit=1", headers = {"authorization": authorization})
        reqAllTexts = json.loads(req.text)
        if len(reqAllTexts) != 1:
            raise Exception(f"reqAllTexts has length of {len(reqAllTexts)}")
        
        return reqAllTexts[0]
    
    def test(self, msg):
        response = chatGpt(self.chatGptPromptMsg + msg)
        print(response)
        
        
if __name__ == "__main__":
    discordNews = DiscordNews()
    discordNews.execute()
    #discordNews.test(msg)
