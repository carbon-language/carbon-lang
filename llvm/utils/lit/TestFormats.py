import os

import Test
import TestRunner
import Util

class GoogleTest(object):
    def __init__(self, test_sub_dir, test_suffix):
        self.test_sub_dir = str(test_sub_dir)
        self.test_suffix = str(test_suffix)

    def getGTestTests(self, path):
        """getGTestTests(path) - [name]
        
        Return the tests available in gtest executable."""

        lines = Util.capture([path, '--gtest_list_tests']).split('\n')
        nested_tests = []
        for ln in lines:
            if not ln.strip():
                continue

            prefix = ''
            index = 0
            while ln[index*2:index*2+2] == '  ':
                index += 1
            while len(nested_tests) > index:
                nested_tests.pop()
            
            ln = ln[index*2:]
            if ln.endswith('.'):
                nested_tests.append(ln)
            else:
                yield ''.join(nested_tests) + ln

    def getTestsInDirectory(self, testSuite, path_in_suite,
                            litConfig, localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for filename in os.listdir(source_path):
            # Check for the one subdirectory (build directory) tests will be in.
            if filename != self.test_sub_dir:
                continue

            filepath = os.path.join(source_path, filename)
            for subfilename in os.listdir(filepath):
                if subfilename.endswith(self.test_suffix):
                    execpath = os.path.join(filepath, subfilename)

                    # Discover the tests in this executable.
                    for name in self.getGTestTests(execpath):
                        testPath = path_in_suite + (filename, subfilename, name)
                        yield Test.Test(testSuite, testPath, localConfig)

    def execute(self, test, litConfig):
        testPath,testName = os.path.split(test.getSourcePath())

        cmd = [testPath, '--gtest_filter=' + testName]
        out, err, exitCode = TestRunner.executeCommand(cmd)
            
        if not exitCode:
            return Test.PASS,''

        return Test.FAIL, out + err

###

class FileBasedTest(object):
    def getTestsInDirectory(self, testSuite, path_in_suite,
                            litConfig, localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for filename in os.listdir(source_path):
            filepath = os.path.join(source_path, filename)
            if not os.path.isdir(filepath):
                base,ext = os.path.splitext(filename)
                if ext in localConfig.suffixes:
                    yield Test.Test(testSuite, path_in_suite + (filename,),
                                    localConfig)

class ShTest(FileBasedTest):
    def __init__(self, execute_external = False, require_and_and = False):
        self.execute_external = execute_external
        self.require_and_and = require_and_and

    def execute(self, test, litConfig):
        return TestRunner.executeShTest(test, litConfig,
                                        self.execute_external,
                                        self.require_and_and)

class TclTest(FileBasedTest):
    def execute(self, test, litConfig):
        return TestRunner.executeTclTest(test, litConfig)
