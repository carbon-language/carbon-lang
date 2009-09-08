import TestRunner

class ShTest:
    def __init__(self, execute_external = False, require_and_and = False):
        self.execute_external = execute_external
        self.require_and_and = require_and_and

    def execute(self, test, litConfig):
        return TestRunner.executeShTest(test, litConfig,
                                        self.execute_external,
                                        self.require_and_and)

