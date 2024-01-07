import requests
import json
import os
import time
import configparser
from openai import RateLimitError
from chatGpt import chatGpt
import configparser
from datetime import datetime
from newsBot.discordDbUtils import *



msg = """We are on the third week of the Outland Odyssey Gamefest! Join various challenges, earn points, and aim for the top spot on the Avoria Champions Leaderboard to win fantastic $SHILL rewards!

🌄 Into the Outlands: give your real-time feedback to help improve the game's quality.
⏱ Instant Fun: race against the clock to complete a specific level.
🧩 Riddle Hunters: solve riddles that will be posted at random times and uncover hidden words.

Become the true hero of Avoria now!
"""


class DiscordNews():
    chatGptPromptMsg = """Regarding the text in between the symbols ##########, answer each of the following questions with yes or no and two sentences of why you think so:
1/ Is the post negative?
2/ Is the post about the past?
3/ Does it mention a listing or partnership with a centralized exchange?
4/ Does it mention a Venture Capital funding or investment?
5/ Does it mention a token staking announcement?
6/ Does it explicitly mention the release of a new game?
7/ Does it mention an epic store or steam announcement?
8/ Does it mention a listing on Binance, Coinbase or Crypto.com?
9/ Does it mention the use of Immutable X?
10/ Does it mention a partnership with a well-known company?

########## 
"""
    
    def __init__(self):        
        self.config = configparser.ConfigParser()
        self.config.read(os.environ['PYTHONPATH'] + "/config/prod.env")
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
            if (reqText['timestamp_new'] == channelRow["last_news_date"].to_pydatetime()):
                id = id + 1 if id + 1 != len(self.discordNewsInfo) else 0
                time.sleep(10)
                continue
            
            updateDiscordNewsInfo(channelRow["channel_id"], reqText["timestamp_new"], self.config)
            self.discordNewsInfo.loc[id, "last_news_date"] = reqText['timestamp_new']
        
            print(f"CONTENT: {reqText['content']}")
            print(f"CONTENT CREATED: {reqText['timestamp']}")
            
            if not self.tradeOnChatGpt(reqText["content"]):
                continue
            
            with open(os.environ['PYTHONPATH'] + "/newsBot/newsBot.txt", "a") as file:
                file.write(f"\n{self.discordNewsInfo.loc[id, 'token_name']},{self.discordNewsInfo.loc[id, 'token_id']},{discordReadDate},{discordNewsDate}")
            
            break
            

    def tradeOnChatGpt(self, postMsg):
        response = chatGpt(self.chatGptPromptMsg + postMsg + """\n##########""")
        print(response)                
        if ("1/ No" in response and "2/ No" in response and response.count("Yes") > 0):
            return True
        
        return False

    def retrieveMsg(channelId, authorization):
        req = requests.get(f"https://discord.com/api/v9/channels/{channelId}/messages?limit=1", headers = {"authorization": authorization})
        reqAllTexts = json.loads(req.text)
        if len(reqAllTexts) != 1:
            raise Exception(f"reqAllTexts has length of {len(reqAllTexts)}")
        
        return reqAllTexts[0]
    
    def test(self, msg):
        self.chatGptPromptMsg = """Regarding the text in between the symbols ##########, answer each of the following questions with yes or no and two sentences of why you think so:
1/ Is the post negative?
2/ Is the post about the past?
3/ Does it mention a listing or partnership with a centralized exchange?
4/ Does it mention a Venture Capital funding or investment?
5/ Does it mention a token staking announcement?
6/ Does it mention that a new game will be released?
7/ Does it mention an epic store or steam announcement?
8/ Does it mention a listing on Binance, Coinbase or Crypto.com?
9/ Does it mention the use of Immutable X?
10/ Does it mention a partnership with a well-known company?

########## 
"""
        
        msg1 = """##########"""
        
        msg2 = """Yes or no regarding the post labeled POST/ from the first request: 
        3/ Does it mention a listing or partnership with a centralized exchange?"""
        msg3 = """Yes or no regarding the post labeled POST/ from the first request: 
        6/ Does it mention the release of a new game?"""
        msg4 = """Yes or no regarding the post labeled POST/ from the first request:
        9/ Does it mention the use of Immutable X?
        """
        response = chatGpt([self.chatGptPromptMsg + msg + msg1])
        print(response)
        return response
        
        
if __name__ == "__main__":
    discordNews = DiscordNews()
    discordNews.execute()
    #discordNews.test(msg)
