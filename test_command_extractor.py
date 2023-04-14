import unittest
from CommandExtractor import CommandExtractor, NormalWord, ReplaceableWord, Word
from CommandExtractor import Command


commands = """
go <direction> 
pick <item> 
talk to <character>
enter the battle with <monster> [vifi,ligi,act]
"""

class ClassExtractor(unittest.TestCase):
    # with open('command.txt', 'r') as w:
    print([line.rstrip() for line in commands.strip().split('\n')])
    commandExtractor = CommandExtractor([line.rstrip() for line in commands.strip().split('\n')])
    print(len([line.rstrip() for line in commands.strip().split('\n')]))
    print([command._pos for command in commandExtractor.commandLines[3] ])
    def test_initialise(self):
        commands = self.commandExtractor.commandLines
        for command in commands:
            self.assertEqual(isinstance(command, Command), True)
        self.assertEqual(commands[0][0].getContent(), 'go')
        self.assertEqual(commands[0][1].getContent(), 'direction')
        self.assertEqual(isinstance(commands[0][1], ReplaceableWord),True)
        self.assertEqual(isinstance(commands[0][1], Word),True)
        self.assertEqual(isinstance(commands[0][1], NormalWord),False)
    
    def test_reset(self):
        commands = """
            go <direction> 
            pick <item> 
            enter the battle with <monster> [vifi,ligi,act]
            """
        self.commandExtractor.resetCommandSet([line.rstrip() for line in commands.strip().split('\n')])
        self.assertEqual(len(self.commandExtractor.commandLines), 3)
