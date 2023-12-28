
url = "https://discord.com/api/oauth2/authorize?client_id=1175770970573770842&permissions=8&scope=bot"

import discord

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        print(f'Channel: {message.channel.name} Message from {message.author}: {message.content}')

        
        
        


intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)
client.run(token)

# client = discord.Client()
# guild = discord.Guild

# @client.event
# async def on_ready():
#     print('Hello {0.user} !'.format(client))
#     await client.change_presence(activity=discord.Game('_scan help'))

# if __name__ == "__main__":
#     client.run(token)