//const solanaWeb3 = require("@solana/web3.js");
import { Connection, Keypair, VersionedTransaction } from '@solana/web3.js';
import fetch from 'cross-fetch';
import { Wallet } from '@project-serum/anchor';
import bs58 from 'bs58';
//import { parseErrorForTransaction } from '@mercurial-finance/optimist';
import * as fs from 'fs';
import { createInterface } from 'readline';
import { setTimeout } from "timers/promises";
import nodemailer from "nodemailer";
import ConfigParser from 'configparser';
import {SecretsManagerClient, GetSecretValueCommand} from "@aws-sdk/client-secrets-manager";

//const ConfigParser = require('configparser');

async function get_secret(secretKey) {
  const client = new SecretsManagerClient({region: "eu-north-1",});
  let response;
  response = await client.send(new GetSecretValueCommand({SecretId: "NSsecrets", VersionStage: "AWSCURRENT"}));
  return JSON.parse(response.SecretString)[secretKey];
}

const solana_prv_key_wallet = await get_secret("solana_prv_key_wallet")

console.log(1);

const config = new ConfigParser();
config.read(process.env.PROJECTDIR + '/config/prod.env');
const rpc_url = config.get('JS', 'rpc_url');
//const solana_prv_key_wallet = config.get('JS', 'solana_prv_key_wallet');

console.log(2);

var lineToSplit = process.argv[2];
var lineSplit = lineToSplit.split(',');
const tokenName = lineSplit[0];
const tokenId = lineSplit[1];
const discordReadDate = lineSplit[2];
const discordNewsDate = lineSplit[3];

console.log(3);

// check if written line is within the last 1 hour!!!!!! to avoid trading the same thing twice
var ONE_HOUR = 60 * 60 * 1000; /* ms */
const s = '1970-01-01 00:03:44';
const myDate = new Date(discordReadDate);
var isNew = ((new Date) - myDate) < ONE_HOUR;
if (isNew == false) {
  throw new Error('Discord Date is not within 1 hour of execution');
}

console.log(4);

const connection = new Connection(rpc_url);
const wallet = new Wallet(Keypair.fromSecretKey(bs58.decode(solana_prv_key_wallet || '')));

console.log(5);

// // Retrieve the `indexed-route-map`
// const indexedRouteMap = await (await fetch('https://quote-api.jup.ag/v6/indexed-route-map')).json();
// const getMint = (index) => indexedRouteMap["mintKeys"][index];
// const getIndex = (mint) => indexedRouteMap["mintKeys"].indexOf(mint);

// console.log(6);

// // Generate the route map by replacing indexes with mint addresses
// var generatedRouteMap = {};
// Object.keys(indexedRouteMap['indexedRouteMap']).forEach((key, index) => {
//   generatedRouteMap[getMint(key)] = indexedRouteMap["indexedRouteMap"][key].map((index) => getMint(index))
// });

// console.log(7);

// // List all possible input tokens by mint address
// const allInputMints = Object.keys(generatedRouteMap);

// // List all possition output tokens that can be swapped from the mint address for SOL.
// // SOL -> X
// const swappableOutputForSOL = generatedRouteMap['So11111111111111111111111111111111111111112'];
// console.log({ allInputMints, swappableOutputForSOL })

// Swapping SOL to USDC with input 0.1 SOL and 0.5% slippage
const inputMint = 'So11111111111111111111111111111111111111112';
//const outputMint = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'; // usdc
var outputMint = tokenId
var solAmount = 0.001

const fetchRequest = 'https://quote-api.jup.ag/v6/quote?inputMint=' + inputMint + '&outputMint=' + outputMint + '&amount=' + solAmount*1000000000 + '&slippageBps=5000';
console.log(fetchRequest)

const quoteResponse = await (await fetch(fetchRequest)).json();
console.log({quoteResponse})


// get serialized transactions for the swap
const { swapTransaction } = await (
    await fetch('https://quote-api.jup.ag/v6/swap', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        // quoteResponse from /quote api
        quoteResponse,
        // user public key to be used for the swap
        userPublicKey: wallet.publicKey.toString(),
        // auto wrap and unwrap SOL. default is true
        wrapAndUnwrapSol: true,
        // feeAccount is optional. Use if you want to charge a fee.  feeBps must have been passed in /quote API.
        // feeAccount: "fee_account_public_key"
      })
    })
  ).json();


// deserialize the transaction
const swapTransactionBuf = Buffer.from(swapTransaction, 'base64');
var transaction = VersionedTransaction.deserialize(swapTransactionBuf);
console.log(transaction);

// sign the transaction
transaction.sign([wallet.payer]);

console.log("7")

// Execute the transaction
const rawTransaction = transaction.serialize()

var txid;
var retry = 0
while (true) {
  try {
    txid = await connection.sendRawTransaction(rawTransaction, {
      skipPreflight: false,
      maxRetries: 5
    });
    break;
  }
  catch(err) {
    retry++;
    if (retry == 100)
      throw err;

    await setTimeout(500);
    console.log("Retrying..." + retry)
  }
}

console.log(txid);
//await connection.confirmTransaction(txid);

const latestBlockHash = await connection.getLatestBlockhash('finalized');
await connection.confirmTransaction({blockhash: latestBlockHash.blockhash, lastValidBlockHeight: latestBlockHash.lastValidBlockHeight, signature: txid});

const tradeDate = new Date().toLocaleString();

console.log(`https://solscan.io/tx/${txid}`);
console.log("88")


var transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: 'nikolaislavev96@gmail.com',
    pass: 'zmcr zjmg xvkj sptv'
  }
});

var emailSubject = 'Bought token: ' + tokenName;
var emailText = 'Fetch request: ' + fetchRequest + 
                '\n\nToken Id: ' + tokenId + 
                '\nToken Name: ' + tokenName +
                '\nDiscord News Date: ' + discordNewsDate +
                '\nDiscord Read Date: ' + discordReadDate +
                '\nTrade Date: ' + tradeDate + 
                '\n\nQuote response: ' + JSON.stringify(quoteResponse) + 
                '\n\ntxid: ' + `https://solscan.io/tx/${txid}`;
var mailOptions = {
  from: 'nikolaislavev96@gmail.com',
  to: 'nikolayslavev96@gmail.com',
  subject: emailSubject,
  text: emailText
};

transporter.sendMail(mailOptions, function(error, info){
  if (error) {
    console.log(error);
  } else {
    console.log('Email sent: ' + info.response);
  }
});
