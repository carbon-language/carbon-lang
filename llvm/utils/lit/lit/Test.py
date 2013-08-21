import os

# Test results.

class ResultCode(object):
    """Test result codes."""

    def __init__(self, name, isFailure):
        self.name = name
        self.isFailure = isFailure

    def __repr__(self):
        return '%s%r' % (self.__class__.__name__,
                         (self.name, self.isFailure))

PASS        = ResultCode('PASS', False)
XFAIL       = ResultCode('XFAIL', False)
FAIL        = ResultCode('FAIL', True)
XPASS       = ResultCode('XPASS', True)
UNRESOLVED  = ResultCode('UNRESOLVED', True)
UNSUPPORTED = ResultCode('UNSUPPORTED', False)

class Result(object):
    """Wrapper for the results of executing an individual test."""

    def __init__(self, code, output='', elapsed=None):
        # The result code.
        self.code = code
        # The test output.
        self.output = output
        # The wall timing to execute the test, if timing.
        self.elapsed = elapsed

# Test classes.

class TestSuite:
    """TestSuite - Information on a group of tests.

    A test suite groups together a set of logically related tests.
    """

    def __init__(self, name, source_root, exec_root, config):
        self.name = name
        self.source_root = source_root
        self.exec_root = exec_root
        # The test suite configuration.
        self.config = config

    def getSourcePath(self, components):
        return os.path.join(self.source_root, *components)

    def getExecPath(self, components):
        return os.path.join(self.exec_root, *components)

class Test:
    """Test - Information on a single test instance."""

    def __init__(self, suite, path_in_suite, config):
        self.suite = suite
        self.path_in_suite = path_in_suite
        self.config = config
        # The test result, once complete.
        self.result = None

    def setResult(self, result):
        if self.result is not None:
            raise ArgumentError("test result already set")
        if not isinstance(result, Result):
            raise ArgumentError("unexpected result type")

        self.result = result

    def getFullName(self):
        return self.suite.config.name + ' :: ' + '/'.join(self.path_in_suite)

    def getSourcePath(self):
        return self.suite.getSourcePath(self.path_in_suite)

    def getExecPath(self):
        return self.suite.getExecPath(self.path_in_suite)
