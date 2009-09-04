class TestingConfig:
    """"
    TestingConfig - Information on a how to run a group of tests.
    """
    
    @staticmethod
    def frompath(path):
        data = {}
        f = open(path)
        exec f in {},data

        return TestingConfig(suffixes = data.get('suffixes', []),
                             environment = data.get('environment', {}))

    def __init__(self, suffixes, environment):
        self.suffixes = set(suffixes)
        self.environment = dict(environment)

        # Variables set internally.
        self.root = None
        self.useValgrind = None
        self.useExternalShell = None
        self.valgrindArgs = []

        # FIXME: These need to move into a substitutions mechanism.
        self.clang = None
        self.clangcc = None
