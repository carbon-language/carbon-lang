import os

# Test results.

class ResultCode(object):
    """Test result codes."""

    # We override __new__ and __getnewargs__ to ensure that pickling still
    # provides unique ResultCode objects in any particular instance.
    _instances = {}
    def __new__(cls, name, isFailure):
        res = cls._instances.get(name)
        if res is None:
            cls._instances[name] = res = super(ResultCode, cls).__new__(cls)
        return res
    def __getnewargs__(self):
        return (self.name, self.isFailure)

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
        # A list of conditions under which this test is expected to fail. These
        # can optionally be provided by test format handlers, and will be
        # honored when the test result is supplied.
        self.xfails = []
        # The test result, once complete.
        self.result = None

    def setResult(self, result):
        if self.result is not None:
            raise ArgumentError("test result already set")
        if not isinstance(result, Result):
            raise ArgumentError("unexpected result type")

        self.result = result

        # Apply the XFAIL handling to resolve the result exit code.
        if self.isExpectedToFail():
            if self.result.code == PASS:
                self.result.code = XPASS
            elif self.result.code == FAIL:
                self.result.code = XFAIL
        
    def getFullName(self):
        return self.suite.config.name + ' :: ' + '/'.join(self.path_in_suite)

    def getSourcePath(self):
        return self.suite.getSourcePath(self.path_in_suite)

    def getExecPath(self):
        return self.suite.getExecPath(self.path_in_suite)

    def isExpectedToFail(self):
        """
        isExpectedToFail() -> bool

        Check whether this test is expected to fail in the current
        configuration. This check relies on the test xfails property which by
        some test formats may not be computed until the test has first been
        executed.
        """

        # Check if any of the xfails match an available feature or the target.
        for item in self.xfails:
            # If this is the wildcard, it always fails.
            if item == '*':
                return True

            # If this is an exact match for one of the features, it fails.
            if item in self.config.available_features:
                return True

            # If this is a part of the target triple, it fails.
            if item in self.suite.config.target_triple:
                return True

        return False
