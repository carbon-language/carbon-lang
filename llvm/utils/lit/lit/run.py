class Run(object):
    """
    This class represents a concrete, configured testing run.
    """

    def __init__(self, lit_config, tests):
        self.lit_config = lit_config
        self.tests = tests
