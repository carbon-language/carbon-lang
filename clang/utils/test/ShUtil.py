import Util

class ShLexer:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.end = len(data)

    def eat(self):
        c = self.data[self.pos]
        self.pos += 1
        return c

    def look(self):
        return self.data[self.pos]

    def maybe_eat(self, c):
        """
        maybe_eat(c) - Consume the character c if it is the next character,
        returning True if a character was consumed. """
        if self.data[self.pos] == c:
            self.pos += 1
            return True
        return False

    def lex_arg(self, c):
        if c in "'\"":
            str = self.lex_arg_quoted(c)
        else:
            str = c
        while self.pos != self.end:
            c = self.look()
            if c.isspace() or c in "|><&":
                break
            elif c == '"':
                self.eat()
                str += self.lex_arg_quoted('"')
            else:
                str += self.eat()
        return str

    def lex_arg_quoted(self, delim):
        str = ''
        while self.pos != self.end:
            c = self.eat()
            if c == delim:
                return str
            elif c == '\\' and delim == '"':
                # Shell escaping is just '\"' to avoid termination, no actual
                # escaping.
                if self.pos == self.end:
                    Util.warning("escape at end of quoted argument in: %r" % 
                                 self.data)
                    return str
                c = self.eat()
                if c != delim:
                    str += '\\'
                str += c
            else:
                str += c
        Util.warning("missing quote character in %r" % self.data)
        return str

    def lex_one_token(self):
        """
        lex_one_token - Lex a single 'sh' token. """

        c = self.eat()
        if c == ';':
            return (c)
        if c == '|':
            if self.maybe_eat('|'):
                return ('||',)
            return (c,)
        if c == '&':
            if self.maybe_eat('&'):
                return ('&&',)
            if self.maybe_eat('>'): 
                return ('&>',)
            return (c,)
        if c == '>':
            if self.maybe_eat('&'):
                return ('>&',)
            if self.maybe_eat('>'):
                return ('>>',)
            return (c,)
        if c == '<':
            if self.maybe_eat('&'):
                return ('<&',)
            if self.maybe_eat('>'):
                return ('<<',)
        return self.lex_arg(c)

    def lex(self):
        while self.pos != self.end:
            if self.look().isspace():
                self.eat()
            else:
                yield self.lex_one_token()

###

import unittest

class TestShLexer(unittest.TestCase):
    def lex(self, str):
        return list(ShLexer(str).lex())

    def testops(self):
        self.assertEqual(self.lex('a2>c'),
                         ['a2', ('>',), 'c'])
        self.assertEqual(self.lex('a 2>c'),
                         ['a', '2', ('>',), 'c'])
        
    def testquoting(self):
        self.assertEqual(self.lex(""" 'a' """),
                         ['a'])
        self.assertEqual(self.lex(""" "hello\\"world" """),
                         ['hello"world'])
        self.assertEqual(self.lex(""" "hello\\'world" """),
                         ["hello\\'world"])
        self.assertEqual(self.lex(""" he"llo wo"rld """),
                         ["hello world"])

if __name__ == '__main__':
    unittest.main()
