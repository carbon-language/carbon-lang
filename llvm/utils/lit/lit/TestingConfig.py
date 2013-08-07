import os
import sys

PY2 = sys.version_info[0] < 3

class TestingConfig:
    """"
    TestingConfig - Information on the tests inside a suite.
    """

    @staticmethod
    def frompath(path, parent, litConfig, mustExist, config = None):
        if config is None:
            # Set the environment based on the command line arguments.
            environment = {
                'LIBRARY_PATH' : os.environ.get('LIBRARY_PATH',''),
                'LD_LIBRARY_PATH' : os.environ.get('LD_LIBRARY_PATH',''),
                'PATH' : os.pathsep.join(litConfig.path +
                                         [os.environ.get('PATH','')]),
                'SYSTEMROOT' : os.environ.get('SYSTEMROOT',''),
                'TERM' : os.environ.get('TERM',''),
                'LLVM_DISABLE_CRASH_REPORT' : '1',
                }

            if sys.platform == 'win32':
                environment.update({
                        'INCLUDE' : os.environ.get('INCLUDE',''),
                        'PATHEXT' : os.environ.get('PATHEXT',''),
                        'PYTHONUNBUFFERED' : '1',
                        'TEMP' : os.environ.get('TEMP',''),
                        'TMP' : os.environ.get('TMP',''),
                        })

            # Set the default available features based on the LitConfig.
            available_features = []
            if litConfig.useValgrind:
                available_features.append('valgrind')
                if litConfig.valgrindLeakCheck:
                    available_features.append('vg_leak')

            config = TestingConfig(parent,
                                   name = '<unnamed>',
                                   suffixes = set(),
                                   test_format = None,
                                   environment = environment,
                                   substitutions = [],
                                   unsupported = False,
                                   on_clone = None,
                                   test_exec_root = None,
                                   test_source_root = None,
                                   excludes = [],
                                   available_features = available_features,
                                   pipefail = True)

        if os.path.exists(path):
            # FIXME: Improve detection and error reporting of errors in the
            # config file.
            f = open(path)
            cfg_globals = dict(globals())
            cfg_globals['config'] = config
            cfg_globals['lit'] = litConfig
            cfg_globals['__file__'] = path
            try:
                data = f.read()
                if PY2:
                    exec("exec data in cfg_globals")
                else:
                    exec(data, cfg_globals)
                if litConfig.debug:
                    litConfig.note('... loaded config %r' % path)
            except SystemExit:
                e = sys.exc_info()[1]
                # We allow normal system exit inside a config file to just
                # return control without error.
                if e.args:
                    raise
            f.close()
        else:
            if mustExist:
                litConfig.fatal('unable to load config from %r ' % path)
            elif litConfig.debug:
                litConfig.note('... config not found  - %r' %path)

        config.finish(litConfig)
        return config

    def __init__(self, parent, name, suffixes, test_format,
                 environment, substitutions, unsupported, on_clone,
                 test_exec_root, test_source_root, excludes,
                 available_features, pipefail):
        self.parent = parent
        self.name = str(name)
        self.suffixes = set(suffixes)
        self.test_format = test_format
        self.environment = dict(environment)
        self.substitutions = list(substitutions)
        self.unsupported = unsupported
        self.on_clone = on_clone
        self.test_exec_root = test_exec_root
        self.test_source_root = test_source_root
        self.excludes = set(excludes)
        self.available_features = set(available_features)
        self.pipefail = pipefail

    def clone(self, path):
        # FIXME: Chain implementations?
        #
        # FIXME: Allow extra parameters?
        cfg = TestingConfig(self, self.name, self.suffixes, self.test_format,
                            self.environment, self.substitutions,
                            self.unsupported, self.on_clone,
                            self.test_exec_root, self.test_source_root,
                            self.excludes, self.available_features,
                            self.pipefail)
        if cfg.on_clone:
            cfg.on_clone(self, cfg, path)
        return cfg

    def finish(self, litConfig):
        """finish() - Finish this config object, after loading is complete."""

        self.name = str(self.name)
        self.suffixes = set(self.suffixes)
        self.environment = dict(self.environment)
        self.substitutions = list(self.substitutions)
        if self.test_exec_root is not None:
            # FIXME: This should really only be suite in test suite config
            # files. Should we distinguish them?
            self.test_exec_root = str(self.test_exec_root)
        if self.test_source_root is not None:
            # FIXME: This should really only be suite in test suite config
            # files. Should we distinguish them?
            self.test_source_root = str(self.test_source_root)
        self.excludes = set(self.excludes)

    @property
    def root(self):
        """root attribute - The root configuration for the test suite."""
        if self.parent is None:
            return self
        else:
            return self.parent.root

