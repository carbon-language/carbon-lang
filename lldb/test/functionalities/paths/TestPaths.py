"""
Test some lldb command abbreviations.
"""
import commands
import lldb
import os
import time
import unittest2
from lldbtest import *
import lldbutil

class TestPaths(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_paths (self):
        '''Test to make sure no file names are set in the lldb.SBFileSpec objects returned by lldb.SBHostOS.GetLLDBPath() for paths that are directories'''
        dir_path_types = [lldb.ePathTypeLLDBShlibDir, 
                         lldb.ePathTypeSupportExecutableDir,
                         lldb.ePathTypeHeaderDir,
                         lldb.ePathTypePythonDir,
                         lldb.ePathTypeLLDBSystemPlugins,
                         lldb.ePathTypeLLDBUserPlugins,
                         lldb.ePathTypeLLDBTempSystemDir]
                        
        for path_type in dir_path_types:
            f = lldb.SBHostOS.GetLLDBPath(path_type);
            # No directory path types should have the filename set
            self.assertTrue (f.GetFilename() == None);
        

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

