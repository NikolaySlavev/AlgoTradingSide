#!/bin/bash

export PYTHONPATH="D:/Sidework/AlgoTrading"
export PROJECTDIR="D:/Sidework/AlgoTrading"

while true
do
    python "D:/Sidework/AlgoTrading/newsBot/discordScrapBot.py"
    if [ $? -ne 0 ]; then
        echo "Python code failed"
        exit 1
    fi

    tradeLine=$(tail -n 1 "D:/Sidework/AlgoTrading/newsBot/newsBot.txt")
    if [[ $tradeLine == *traded ]]; then
        echo "Last line already traded"
        exit 1
    fi
    echo ",traded" >> "D:/Sidework/AlgoTrading/newsBot/newsBot.txt"

    node "D:/Sidework/AlgoTrading/newsBot/solanaSwapTokens.js" "$tradeLine"
    if [ $? -ne 0 ]; then
        echo "JS code failed"
        exit 1
    fi
done