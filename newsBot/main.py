# 1/
# Listen real-time to a twitter feed (all the time) and upload it on pythonanywhere
# Since we don't have any other real-time consoles from pythonanywhere, need to set it up so that it restarts every hour or 10 hours?
# Use teepy and create a developers accout on twitter

# 2/
# Parse the post and decide whether to act based on predefined keywords - was it called tokenization? or web scrapping?
# Is it possible to use chatgpt to ask him based on that post - Is the person saying something positive for some asset (or is he suggesting to buy something or is he saying that he has bought sth)

# 3/
# If we have decided to act on the tweet (i.e. buy something), connect to the exchange and buy the asset
# Decide on an algorithm to sell. The whole point is to sell quickly when everybody else has bought before the hype has gone (10min to 1hr max)
# Might want to sell periodically and not on one go
# We buy on one go and the whole point is to buy as quickly as possible
# We already lose ~5sec on retrieving the news (because of the twitter api) and the parsing and buying needs to be done as quickly as possible
# We might lose some time on connecting to the exchange so we can async connect as soon as we receive a tweet


from web3 import Web3

from solana.account import Account
from solana.keypair import Keypair
from solana.rpc.api import Client
from pyserum.market import Market
import base58


def solana():
    # Replace with the appropriate RPC endpoint
    rpc_url = "https://api.mainnet-beta.solana.com"
    rpc_url = "https://api.devnet.solana.com"
    rpc_url = "https://api.testnet.solana.com"
    rpc_url = "https://crimson-methodical-asphalt.solana-mainnet.quiknode.pro/dd2889007cd522839fcda84834f9baf586fe4f89/"
    #rpc_url = "https://rpc.ankr.com/solana"
    genesis_rpc_url = "https://ssc-dao.genesysgo.net"
    serum_rpc_url = "https://solana-api.projectserum.com"
    rpc_client = Client(rpc_url)

    # import base58
    # import base64
    # a = base58.b58decode(solana_prv_key_wallet)
    # a = a[12:76]
    # b = a.hex()
    # c = bytes.fromhex(b)

    #account = Account(private_key_bytes=bytes.fromhex(solana_prv_key_wallet))
    #account = Account(secret_key=solana_prv_key_wallet)
    
    market_address = sol_usdt
    market = Market.load(rpc_client, market_address)
    
    #open_orders_address = market.find_open_orders_account(account.public_key())
    open_orders_address = market.find_open_orders_accounts_for_owner(solana_wallet)
    open_orders = market.load_orders_for_owner(solana_wallet)
    
    bytes=base58.b58decode(solana_prv_key_wallet)
    secret_key=bytes[:32]
    keypair=Keypair.from_secret_key(secret_key)
    print(keypair.public_key)
    
    quantity = 0.01
    result = market.place_order(payer = solana_wallet, owner = keypair, side= 1, limit_price = 1 , max_quantity = 0.01, order_type = 1)
    

        # order_type: OrderType,
        # side: Side,
        # limit_price: float,
        # max_quantity: float,
        # client_id: int = 0,
        # opts: TxOpts = TxOpts()
    
    print(result)
  
solana()


def metamask():
    w3 = Web3(Web3.HTTPProvider(sepolia))
    public_address = pubKey
    friends_address = friendKey
    private_key = prvKey

    address1 = Web3.to_checksum_address(public_address)
    address2 = Web3.to_checksum_address(friends_address)
    
    balance = w3.eth.get_balance(public_address)
    balanceEther = w3.from_wei(balance,'ether')

    nonce = w3.eth.get_transaction_count(address1)
    tx = {"nonce": nonce, "to": address2, "value": w3.to_wei(0.001, "ether"), "gas": 2100, "gasPrice": w3.to_wei(40, "gwei")}
    signed_tx = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

#chatGpt()
#metamask()

a = "SepoliaETH"