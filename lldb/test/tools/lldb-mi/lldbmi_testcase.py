"""
Base class for lldb-mi test cases.
"""

from lldbtest import *
import pexpect
import unittest2

class MiTestCaseBase(Base):

    mydir = Base.compute_mydir(__file__)
    myexe = "a.out"
    mylog = "child.log"

    @classmethod
    def classCleanup(cls):
        TestBase.RemoveTempFile(cls.myexe)
        TestBase.RemoveTempFile(cls.mylog)

    def setUp(self):
        Base.setUp(self)
        self.buildDefault()
        self.child_prompt = "(gdb)"

    def tearDown(self):
        if self.TraceOn():
            print "\n\nContents of %s:" % self.mylog
            print open(self.mylog, "r").read()
        Base.tearDown(self)

    def spawnLldbMi(self, args=None):
        self.child = pexpect.spawn("%s --interpreter %s" % (
            self.lldbMiExec, args if args else ""))
        self.child.setecho(True)
        self.child.logfile_read = open(self.mylog, "w")

    def runCmd(self, cmd):
        self.child.sendline(cmd)

    def expect(self, pattern, *args, **kwargs):
        self.child.expect(pattern, *args, **kwargs)
