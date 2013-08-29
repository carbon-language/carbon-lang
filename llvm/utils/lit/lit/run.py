import time
import traceback

import lit.Test

class Run(object):
    """
    This class represents a concrete, configured testing run.
    """

    def __init__(self, lit_config, tests):
        self.lit_config = lit_config
        self.tests = tests

    def execute_test(self, test):
        result = None
        startTime = time.time()
        try:
            result = test.config.test_format.execute(test, self.lit_config)

            # Support deprecated result from execute() which returned the result
            # code and additional output as a tuple.
            if isinstance(result, tuple):
                code, output = result
                result = lit.Test.Result(code, output)
            elif not isinstance(result, lit.Test.Result):
                raise ValueError("unexpected result from test execution")
        except KeyboardInterrupt:
            raise
        except:
            if self.lit_config.debug:
                raise
            output = 'Exception during script execution:\n'
            output += traceback.format_exc()
            output += '\n'
            result = lit.Test.Result(lit.Test.UNRESOLVED, output)
        result.elapsed = time.time() - startTime

        test.setResult(result)
