import requests
import json
import time
import configparser
from chatGpt import chatGpt
import configparser
from datetime import datetime
from newsBot.discordDbUtils import *


#channelIdDict = {"Wild Forest": 943101879095791657, "Battle Crush": 1145901840542736404, "The Heist": 1067136426782760970}

config = configparser.ConfigParser()
config.read("config/prod.env")


class DiscordNews(config):
    chatGptPromptMsg = """Give me yes or no answers and why you think so in 1 sentence to the questions below. The questions are regarding the post labeled "POST/". 
                    1/ Is the information positive and worth getting hyped about?
                    2/ Does it inform that something new has happened or about to happen?
                    3/ Does it involve token listing on any major Centralised Exchange (CEX)?
                    4/ Does it involve any Venture Capital (VC) funding or investment?
                    5/ Does it involve any token staking announcement?
                    6/ Does it involve game release announcement?
                    7/ Does it involve epic store or steam announcement?
                    8/ Does it involve listing on Binance or Coinbase?
                    9/ Does it involve the use of Immutable X?
                    10/ Does it involve any partnership with a well-known company?
                    POST/ """
    
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read("config/prod.env")
        self.authorization = self.config["DISCORD"]["authorization"]
        self.discordNewsInfo = getDiscordNewsInfo(self.config)
        
    def execute(self):
        id = 0
        while True:
            channelRow = self.discordNewsInfo.iloc[id]
            print(channelRow["channel_name"])
            reqText = DiscordNews.retrieveMsg(channelRow["channel_id"], self.authorization)
            reqText["timestamp_new"] = datetime.strptime(reqText["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").replace(microsecond=0).replace(tzinfo=None)
            discordNewsDate = reqText["timestamp_new"].strftime('%Y-%m-%d %H:%M:%S')
            discordReadDate = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if True:
            #if (reqText['timestamp_new'] != channelRow["last_news_date"].to_pydatetime()):
                updateDiscordNewsInfo(channelRow["channel_id"], reqText["timestamp_new"], self.config)
                self.discordNewsInfo.loc[id, "last_news_date"] = reqText['timestamp_new']
            
                response = chatGpt(self.chatGptPromptMsg + reqText["content"])
                print(reqText["content"])
                print(response)

                if True:
                #if (response.count("Yes") == 3):
                    with open("D:/Sidework/AlgoTrading/newsBot/newsBot.txt", "a") as file:
                        file.write(f"\n{self.discordNewsInfo.loc[id, 'token_name']},{self.discordNewsInfo.loc[id, 'token_id']},{discordReadDate},{discordNewsDate}")
                        
                    break

            id = id + 1 if id + 1 != len(self.discordNewsInfo) else 0
            time.sleep(10)

    def retrieveMsg(channelId, authorization):
        req = requests.get(f"https://discord.com/api/v9/channels/{channelId}/messages?limit=1", headers = {"authorization": authorization})
        reqAllTexts = json.loads(req.text)
        if len(reqAllTexts) != 1:
            raise Exception(f"reqAllTexts has length of {len(reqAllTexts)}")
        
        print(f"CONTENT LENGTH: {len(reqAllTexts[0]['content'])}")
        print(f"CONTENT CREATED: {reqAllTexts[0]['timestamp']}")
        return reqAllTexts[0]
        
        
if __name__ == "__main__":
    discordNews = DiscordNews()
    discordNews.execute()

# have a sql table with all the projects and the last timestamp recorded
# load all of that in memory for quick access and when new reported, update the table
# add to logger when news appears and add to logger on error

# check all channels in order with 2sec delay?
# test with 10sec delay first to check