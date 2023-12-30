from openai import OpenAI
import configparser
import os

msg = """
Is the following post positive and worth getting hyped about? Give me a yes or no answer.

🌟 Major Announcement, Rangers! 🌟

Mark your calendars for November 28th! The moment you've all been waiting for is here! Illuvium Beta 3: PVP is coming, and that's not all—we’re live on the 
@EpicGames
 Store! While you can’t download it just yet, add it to your wishlist and brace for the ultimate Arena PVP experience. 

🎉 Prepare yourselves:
🔹 Early access opens to all our fans
🔹 Smooth, auto-updating gameplay
🔹 Exposure to Epic’s 180M+ users

Get hyped, spread the word, and prep for an epic battle on the Epic Games Store!

**Epic Games Release:** November 28th 2023 

#Illuvium #Beta3 #PVP #EpicGamesStore
"""


msg = "How can I swap tokens in PancakeSwap using Python?"

msg = "Is true that 2 + 2 = 4? Give me yes or no answer."

def chatGpt(msg):
    #model = "gpt-4-1106-preview"
    model = "gpt-3.5-turbo"
    
    # defaults to os.environ.get("OPENAI_API_KEY")
    config = configparser.ConfigParser()
    config.read(os.environ['PYTHONPATH'] + "/config/prod.env")    
    chatGptAccessKey = config["CHATGPT"]["chatGptAccessKey"]
    client = OpenAI(api_key = chatGptAccessKey)
    chat_completion = client.chat.completions.create(messages = [{"role": "user", "content": msg}], model = model)    
    response = chat_completion.choices[0].message.content
    return response

if __name__ == "__main__":
    chatGpt(msg)