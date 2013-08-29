from __future__ import absolute_import
import unittest

import lit.Test

"""
TestCase adaptor for providing a 'unittest' compatible interface to 'lit' tests.
"""

class UnresolvedError(RuntimeError):
    pass
        
class LitTestCase(unittest.TestCase):
    def __init__(self, test, lit_config):
        unittest.TestCase.__init__(self)
        self._test = test
        self._lit_config = lit_config

    def id(self):
        return self._test.getFullName()

    def shortDescription(self):
        return self._test.getFullName()

    def runTest(self):
        result = self._test.config.test_format.execute(
            self._test, self._lit_config)

        # Support deprecated result from execute() which returned the result
        # code and additional output as a tuple.
        if isinstance(result, tuple):
            code, output = result
            result = lit.Test.Result(code, output)
        elif not isinstance(result, lit.Test.Result):
            raise ValueError("unexpected result from test execution")

        if result.code is lit.Test.UNRESOLVED:
            raise UnresolvedError(result.output)
        elif result.code.isFailure:
            self.fail(result.output)
