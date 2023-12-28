$env:PYTHONPATH = "D:/Sidework/AlgoTrading"
$env:PROJECTDIR = "D:/Sidework/AlgoTrading"

for (;;) {
    python "D:/Sidework/AlgoTrading/newsBot/discordScrapBot.py"
    if ($LASTEXITCODE -ne 0) { 
        throw "Python code failed"
    }

    $tradeLine = Get-content -tail 1 "D:/Sidework/AlgoTrading/newsBot/newsBot.txt"
    if ($tradeLine.EndsWith("traded")) {
        throw "Last line already traded" 
    }
    Add-Content -path "D:/Sidework/AlgoTrading/newsBot/newsBot.txt" ",traded"

    node "D:/Sidework/AlgoTrading/newsBot/solanaSwapTokens.js" $tradeLine
    if ($LASTEXITCODE -ne 0) { 
        throw "JS code failed" 
    }
}

