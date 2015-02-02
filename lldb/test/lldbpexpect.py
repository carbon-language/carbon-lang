import lldb
from lldbtest import *
import lldbutil
import os
import unittest2
import sys
import pexpect

class PExpectTest(TestBase):
    
    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def doTest(self):
        # put your commands here
        pass

    def launchArgs(self):
        return ""

    def launch(self):
        self.timeout = 5
        self.child = pexpect.spawn('%s %s' % self.lldbHere, self.launchArgs())

    def expect(self, patterns=None, timeout=None):
        if patterns is None: patterns = '.*'
        return self.child.expect(patterns, timeout=timeout)

    def sendimpl(self, sender, command, patterns=None, timeout=None):
        if timeout is None: timeout = self.timeout
        sender(command)
        if patterns is not None: return self.expect(patterns=patterns, timeout=timeout)
        return None

    def send(self, command, patterns=None, timeout=None):
        self.sendimpl(self.child.send, command, patterns, timeout)

    def sendline(self, command, patterns=None, timeout=None):
        self.sendimpl(self.child.sendline, command, patterns, timeout)

    def quit(self, gracefully=None):
        if gracefully is None: gracefully = True
        self.child.sendeof()
        self.child.close(force=not gracefully)
        self.child = None
