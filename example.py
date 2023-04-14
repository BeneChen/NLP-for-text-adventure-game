commands = """
go <direction> 
pick <item> 
talk to <character>
enter the battle with <monster> [vifi,ligi,act]
"""
import CommandExtractor
commandExtractor = CommandExtractor.CommandExtractor([line.rstrip() for line in commands.strip().split('\n')])
import time
time1 = time.time()
sent = "I want to go west,but I decide to pick up the knife the knife"
commandGenerator = commandExtractor.commandGenerator(sent, lambda x: x == 'pick knife' or x== 'go up')
print(next(commandGenerator))
time2 = time.time()
print(time2 - time1)