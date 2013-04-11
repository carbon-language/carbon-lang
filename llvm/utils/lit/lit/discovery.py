"""
Test discovery functions.
"""

import os
import sys

from lit.TestingConfig import TestingConfig
from lit import LitConfig, Test

def dirContainsTestSuite(path, lit_config):
    cfgpath = os.path.join(path, lit_config.site_config_name)
    if os.path.exists(cfgpath):
        return cfgpath
    cfgpath = os.path.join(path, lit_config.config_name)
    if os.path.exists(cfgpath):
        return cfgpath

def getTestSuite(item, litConfig, cache):
    """getTestSuite(item, litConfig, cache) -> (suite, relative_path)

    Find the test suite containing @arg item.

    @retval (None, ...) - Indicates no test suite contains @arg item.
    @retval (suite, relative_path) - The suite that @arg item is in, and its
    relative path inside that suite.
    """
    def search1(path):
        # Check for a site config or a lit config.
        cfgpath = dirContainsTestSuite(path, litConfig)

        # If we didn't find a config file, keep looking.
        if not cfgpath:
            parent,base = os.path.split(path)
            if parent == path:
                return (None, ())

            ts, relative = search(parent)
            return (ts, relative + (base,))

        # We found a config file, load it.
        if litConfig.debug:
            litConfig.note('loading suite config %r' % cfgpath)

        cfg = TestingConfig.frompath(cfgpath, None, litConfig, mustExist = True)
        source_root = os.path.realpath(cfg.test_source_root or path)
        exec_root = os.path.realpath(cfg.test_exec_root or path)
        return Test.TestSuite(cfg.name, source_root, exec_root, cfg), ()

    def search(path):
        # Check for an already instantiated test suite.
        res = cache.get(path)
        if res is None:
            cache[path] = res = search1(path)
        return res

    # Canonicalize the path.
    item = os.path.realpath(item)

    # Skip files and virtual components.
    components = []
    while not os.path.isdir(item):
        parent,base = os.path.split(item)
        if parent == item:
            return (None, ())
        components.append(base)
        item = parent
    components.reverse()

    ts, relative = search(item)
    return ts, tuple(relative + tuple(components))

def getLocalConfig(ts, path_in_suite, litConfig, cache):
    def search1(path_in_suite):
        # Get the parent config.
        if not path_in_suite:
            parent = ts.config
        else:
            parent = search(path_in_suite[:-1])

        # Load the local configuration.
        source_path = ts.getSourcePath(path_in_suite)
        cfgpath = os.path.join(source_path, litConfig.local_config_name)
        if litConfig.debug:
            litConfig.note('loading local config %r' % cfgpath)
        return TestingConfig.frompath(cfgpath, parent, litConfig,
                                    mustExist = False,
                                    config = parent.clone(cfgpath))

    def search(path_in_suite):
        key = (ts, path_in_suite)
        res = cache.get(key)
        if res is None:
            cache[key] = res = search1(path_in_suite)
        return res

    return search(path_in_suite)

def getTests(path, litConfig, testSuiteCache, localConfigCache):
    # Find the test suite for this input and its relative path.
    ts,path_in_suite = getTestSuite(path, litConfig, testSuiteCache)
    if ts is None:
        litConfig.warning('unable to find test suite for %r' % path)
        return (),()

    if litConfig.debug:
        litConfig.note('resolved input %r to %r::%r' % (path, ts.name,
                                                        path_in_suite))

    return ts, getTestsInSuite(ts, path_in_suite, litConfig,
                               testSuiteCache, localConfigCache)

def getTestsInSuite(ts, path_in_suite, litConfig,
                    testSuiteCache, localConfigCache):
    # Check that the source path exists (errors here are reported by the
    # caller).
    source_path = ts.getSourcePath(path_in_suite)
    if not os.path.exists(source_path):
        return

    # Check if the user named a test directly.
    if not os.path.isdir(source_path):
        lc = getLocalConfig(ts, path_in_suite[:-1], litConfig, localConfigCache)
        yield Test.Test(ts, path_in_suite, lc)
        return

    # Otherwise we have a directory to search for tests, start by getting the
    # local configuration.
    lc = getLocalConfig(ts, path_in_suite, litConfig, localConfigCache)

    # Search for tests.
    if lc.test_format is not None:
        for res in lc.test_format.getTestsInDirectory(ts, path_in_suite,
                                                      litConfig, lc):
            yield res

    # Search subdirectories.
    for filename in os.listdir(source_path):
        # FIXME: This doesn't belong here?
        if filename in ('Output', '.svn', '.git') or filename in lc.excludes:
            continue

        # Ignore non-directories.
        file_sourcepath = os.path.join(source_path, filename)
        if not os.path.isdir(file_sourcepath):
            continue

        # Check for nested test suites, first in the execpath in case there is a
        # site configuration and then in the source path.
        file_execpath = ts.getExecPath(path_in_suite + (filename,))
        if dirContainsTestSuite(file_execpath, litConfig):
            sub_ts, subiter = getTests(file_execpath, litConfig,
                                       testSuiteCache, localConfigCache)
        elif dirContainsTestSuite(file_sourcepath, litConfig):
            sub_ts, subiter = getTests(file_sourcepath, litConfig,
                                       testSuiteCache, localConfigCache)
        else:
            # Otherwise, continue loading from inside this test suite.
            subiter = getTestsInSuite(ts, path_in_suite + (filename,),
                                      litConfig, testSuiteCache,
                                      localConfigCache)
            sub_ts = None

        N = 0
        for res in subiter:
            N += 1
            yield res
        if sub_ts and not N:
            litConfig.warning('test suite %r contained no tests' % sub_ts.name)

def find_tests_for_inputs(lit_config, inputs):
    """
    find_tests_for_inputs(lit_config, inputs) -> [Test]

    Given a configuration object and a list of input specifiers, find all the
    tests to execute.
    """

    # Expand '@...' form in inputs.
    actual_inputs = []
    for input in inputs:
        if os.path.exists(input) or not input.startswith('@'):
            actual_inputs.append(input)
        else:
            f = open(input[1:])
            try:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        actual_inputs.append(ln)
            finally:
                f.close()
                    
    # Load the tests from the inputs.
    tests = []
    test_suite_cache = {}
    local_config_cache = {}
    for input in actual_inputs:
        prev = len(tests)
        tests.extend(getTests(input, lit_config,
                              test_suite_cache, local_config_cache)[1])
        if prev == len(tests):
            lit_config.warning('input %r contained no tests' % input)

    # If there were any errors during test discovery, exit now.
    if lit_config.numErrors:
        print >>sys.stderr, '%d errors, exiting.' % lit_config.numErrors
        sys.exit(2)

    return tests

def load_test_suite(inputs):
    import platform
    import unittest
    from lit.LitTestCase import LitTestCase

    # Create the global config object.
    litConfig = LitConfig.LitConfig(progname = 'lit',
                                    path = [],
                                    quiet = False,
                                    useValgrind = False,
                                    valgrindLeakCheck = False,
                                    valgrindArgs = [],
                                    noExecute = False,
                                    ignoreStdErr = False,
                                    debug = False,
                                    isWindows = (platform.system()=='Windows'),
                                    params = {})

    tests = find_tests_for_inputs(litConfig, inputs)

    # Return a unittest test suite which just runs the tests in order.
    return unittest.TestSuite([LitTestCase(test, litConfig) for test in tests])

