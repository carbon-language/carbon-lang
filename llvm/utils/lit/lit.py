#!/usr/bin/env python

"""
lit - LLVM Integrated Tester.

See lit.pod for more information.
"""

import math, os, platform, random, re, sys, time, threading, traceback

import ProgressBar
import TestRunner
import Util

from TestingConfig import TestingConfig
import LitConfig
import Test

# Configuration files to look for when discovering test suites. These can be
# overridden with --config-prefix.
#
# FIXME: Rename to 'config.lit', 'site.lit', and 'local.lit' ?
gConfigName = 'lit.cfg'
gSiteConfigName = 'lit.site.cfg'

kLocalConfigName = 'lit.local.cfg'

class TestingProgressDisplay:
    def __init__(self, opts, numTests, progressBar=None):
        self.opts = opts
        self.numTests = numTests
        self.current = None
        self.lock = threading.Lock()
        self.progressBar = progressBar
        self.completed = 0

    def update(self, test):
        # Avoid locking overhead in quiet mode
        if self.opts.quiet and not test.result.isFailure:
            self.completed += 1
            return

        # Output lock.
        self.lock.acquire()
        try:
            self.handleUpdate(test)
        finally:
            self.lock.release()

    def finish(self):
        if self.progressBar:
            self.progressBar.clear()
        elif self.opts.quiet:
            pass
        elif self.opts.succinct:
            sys.stdout.write('\n')

    def handleUpdate(self, test):
        self.completed += 1
        if self.progressBar:
            self.progressBar.update(float(self.completed)/self.numTests,
                                    test.getFullName())

        if self.opts.succinct and not test.result.isFailure:
            return

        if self.progressBar:
            self.progressBar.clear()

        print '%s: %s (%d of %d)' % (test.result.name, test.getFullName(),
                                     self.completed, self.numTests)

        if test.result.isFailure and self.opts.showOutput:
            print "%s TEST '%s' FAILED %s" % ('*'*20, test.getFullName(),
                                              '*'*20)
            print test.output
            print "*" * 20

        sys.stdout.flush()

class TestProvider:
    def __init__(self, tests, maxTime):
        self.maxTime = maxTime
        self.iter = iter(tests)
        self.lock = threading.Lock()
        self.startTime = time.time()

    def get(self):
        # Check if we have run out of time.
        if self.maxTime is not None:
            if time.time() - self.startTime > self.maxTime:
                return None

        # Otherwise take the next test.
        self.lock.acquire()
        try:
            item = self.iter.next()
        except StopIteration:
            item = None
        self.lock.release()
        return item

class Tester(threading.Thread):
    def __init__(self, litConfig, provider, display):
        threading.Thread.__init__(self)
        self.litConfig = litConfig
        self.provider = provider
        self.display = display

    def run(self):
        while 1:
            item = self.provider.get()
            if item is None:
                break
            self.runTest(item)

    def runTest(self, test):
        result = None
        startTime = time.time()
        try:
            result, output = test.config.test_format.execute(test,
                                                             self.litConfig)
        except KeyboardInterrupt:
            # This is a sad hack. Unfortunately subprocess goes
            # bonkers with ctrl-c and we start forking merrily.
            print '\nCtrl-C detected, goodbye.'
            os.kill(0,9)
        except:
            if self.litConfig.debug:
                raise
            result = Test.UNRESOLVED
            output = 'Exception during script execution:\n'
            output += traceback.format_exc()
            output += '\n'
        elapsed = time.time() - startTime

        test.setResult(result, output, elapsed)
        self.display.update(test)

def dirContainsTestSuite(path):
    cfgpath = os.path.join(path, gSiteConfigName)
    if os.path.exists(cfgpath):
        return cfgpath
    cfgpath = os.path.join(path, gConfigName)
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
        cfgpath = dirContainsTestSuite(path)

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
        cfgpath = os.path.join(source_path, kLocalConfigName)
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
        return ()

    if litConfig.debug:
        litConfig.note('resolved input %r to %r::%r' % (path, ts.name,
                                                        path_in_suite))

    return getTestsInSuite(ts, path_in_suite, litConfig,
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
    for res in lc.test_format.getTestsInDirectory(ts, path_in_suite,
                                                  litConfig, lc):
        yield res

    # Search subdirectories.
    for filename in os.listdir(source_path):
        # FIXME: This doesn't belong here?
        if filename in ('Output', '.svn') or filename in lc.excludes:
            continue

        # Ignore non-directories.
        file_sourcepath = os.path.join(source_path, filename)
        if not os.path.isdir(file_sourcepath):
            continue

        # Check for nested test suites, first in the execpath in case there is a
        # site configuration and then in the source path.
        file_execpath = ts.getExecPath(path_in_suite + (filename,))
        if dirContainsTestSuite(file_execpath):
            subiter = getTests(file_execpath, litConfig,
                               testSuiteCache, localConfigCache)
        elif dirContainsTestSuite(file_sourcepath):
            subiter = getTests(file_sourcepath, litConfig,
                               testSuiteCache, localConfigCache)
        else:
            # Otherwise, continue loading from inside this test suite.
            subiter = getTestsInSuite(ts, path_in_suite + (filename,),
                                      litConfig, testSuiteCache,
                                      localConfigCache)

        for res in subiter:
            yield res

def runTests(numThreads, litConfig, provider, display):
    # If only using one testing thread, don't use threads at all; this lets us
    # profile, among other things.
    if numThreads == 1:
        t = Tester(litConfig, provider, display)
        t.run()
        return

    # Otherwise spin up the testing threads and wait for them to finish.
    testers = [Tester(litConfig, provider, display)
               for i in range(numThreads)]
    for t in testers:
        t.start()
    try:
        for t in testers:
            t.join()
    except KeyboardInterrupt:
        sys.exit(2)

def main():
    global options
    from optparse import OptionParser, OptionGroup
    parser = OptionParser("usage: %prog [options] {file-or-path}")

    parser.add_option("-j", "--threads", dest="numThreads", metavar="N",
                      help="Number of testing threads",
                      type=int, action="store", default=None)
    parser.add_option("", "--config-prefix", dest="configPrefix",
                      metavar="NAME", help="Prefix for 'lit' config files",
                      action="store", default=None)

    group = OptionGroup(parser, "Output Format")
    # FIXME: I find these names very confusing, although I like the
    # functionality.
    group.add_option("-q", "--quiet", dest="quiet",
                     help="Suppress no error output",
                     action="store_true", default=False)
    group.add_option("-s", "--succinct", dest="succinct",
                     help="Reduce amount of output",
                     action="store_true", default=False)
    group.add_option("-v", "--verbose", dest="showOutput",
                     help="Show all test output",
                     action="store_true", default=False)
    group.add_option("", "--no-progress-bar", dest="useProgressBar",
                     help="Do not use curses based progress bar",
                     action="store_false", default=True)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Test Execution")
    group.add_option("", "--path", dest="path",
                     help="Additional paths to add to testing environment",
                     action="append", type=str, default=[])
    group.add_option("", "--vg", dest="useValgrind",
                     help="Run tests under valgrind",
                     action="store_true", default=False)
    group.add_option("", "--vg-arg", dest="valgrindArgs", metavar="ARG",
                     help="Specify an extra argument for valgrind",
                     type=str, action="append", default=[])
    group.add_option("", "--time-tests", dest="timeTests",
                     help="Track elapsed wall time for each test",
                     action="store_true", default=False)
    group.add_option("", "--no-execute", dest="noExecute",
                     help="Don't execute any tests (assume PASS)",
                     action="store_true", default=False)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Test Selection")
    group.add_option("", "--max-tests", dest="maxTests", metavar="N",
                     help="Maximum number of tests to run",
                     action="store", type=int, default=None)
    group.add_option("", "--max-time", dest="maxTime", metavar="N",
                     help="Maximum time to spend testing (in seconds)",
                     action="store", type=float, default=None)
    group.add_option("", "--shuffle", dest="shuffle",
                     help="Run tests in random order",
                     action="store_true", default=False)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Debug and Experimental Options")
    group.add_option("", "--debug", dest="debug",
                      help="Enable debugging (for 'lit' development)",
                      action="store_true", default=False)
    group.add_option("", "--show-suites", dest="showSuites",
                      help="Show discovered test suites",
                      action="store_true", default=False)
    group.add_option("", "--no-tcl-as-sh", dest="useTclAsSh",
                      help="Don't run Tcl scripts using 'sh'",
                      action="store_false", default=True)
    parser.add_option_group(group)

    (opts, args) = parser.parse_args()

    if not args:
        parser.error('No inputs specified')

    if opts.configPrefix is not None:
        global gConfigName, gSiteConfigName
        gConfigName = '%s.cfg' % opts.configPrefix
        gSiteConfigName = '%s.site.cfg' % opts.configPrefix

    if opts.numThreads is None:
        opts.numThreads = Util.detectCPUs()

    inputs = args

    # Create the global config object.
    litConfig = LitConfig.LitConfig(progname = os.path.basename(sys.argv[0]),
                                    path = opts.path,
                                    quiet = opts.quiet,
                                    useValgrind = opts.useValgrind,
                                    valgrindArgs = opts.valgrindArgs,
                                    useTclAsSh = opts.useTclAsSh,
                                    noExecute = opts.noExecute,
                                    debug = opts.debug,
                                    isWindows = (platform.system()=='Windows'))

    # Load the tests from the inputs.
    tests = []
    testSuiteCache = {}
    localConfigCache = {}
    for input in inputs:
        prev = len(tests)
        tests.extend(getTests(input, litConfig,
                              testSuiteCache, localConfigCache))
        if prev == len(tests):
            litConfig.warning('input %r contained no tests' % input)

    # If there were any errors during test discovery, exit now.
    if litConfig.numErrors:
        print >>sys.stderr, '%d errors, exiting.' % litConfig.numErrors
        sys.exit(2)

    if opts.showSuites:
        suitesAndTests = dict([(ts,[])
                               for ts,_ in testSuiteCache.values()
                               if ts])
        for t in tests:
            suitesAndTests[t.suite].append(t)

        print '-- Test Suites --'
        suitesAndTests = suitesAndTests.items()
        suitesAndTests.sort(key = lambda (ts,_): ts.name)
        for ts,tests in suitesAndTests:
            print '  %s - %d tests' %(ts.name, len(tests))
            print '    Source Root: %s' % ts.source_root
            print '    Exec Root  : %s' % ts.exec_root

    # Select and order the tests.
    numTotalTests = len(tests)
    if opts.shuffle:
        random.shuffle(tests)
    else:
        tests.sort(key = lambda t: t.getFullName())
    if opts.maxTests is not None:
        tests = tests[:opts.maxTests]

    extra = ''
    if len(tests) != numTotalTests:
        extra = ' of %d' % numTotalTests
    header = '-- Testing: %d%s tests, %d threads --'%(len(tests),extra,
                                                      opts.numThreads)

    progressBar = None
    if not opts.quiet:
        if opts.succinct and opts.useProgressBar:
            try:
                tc = ProgressBar.TerminalController()
                progressBar = ProgressBar.ProgressBar(tc, header)
            except ValueError:
                print header
                progressBar = ProgressBar.SimpleProgressBar('Testing: ')
        else:
            print header

    # Don't create more threads than tests.
    opts.numThreads = min(len(tests), opts.numThreads)

    startTime = time.time()
    display = TestingProgressDisplay(opts, len(tests), progressBar)
    provider = TestProvider(tests, opts.maxTime)
    runTests(opts.numThreads, litConfig, provider, display)
    display.finish()

    if not opts.quiet:
        print 'Testing Time: %.2fs'%(time.time() - startTime)

    # Update results for any tests which weren't run.
    for t in tests:
        if t.result is None:
            t.setResult(Test.UNRESOLVED, '', 0.0)

    # List test results organized by kind.
    hasFailures = False
    byCode = {}
    for t in tests:
        if t.result not in byCode:
            byCode[t.result] = []
        byCode[t.result].append(t)
        if t.result.isFailure:
            hasFailures = True

    # FIXME: Show unresolved and (optionally) unsupported tests.
    for title,code in (('Unexpected Passing Tests', Test.XPASS),
                       ('Failing Tests', Test.FAIL)):
        elts = byCode.get(code)
        if not elts:
            continue
        print '*'*20
        print '%s (%d):' % (title, len(elts))
        for t in elts:
            print '    %s' % t.getFullName()
        print

    if opts.timeTests:
        byTime = list(tests)
        byTime.sort(key = lambda t: t.elapsed)
        if byTime:
            Util.printHistogram([(t.getFullName(), t.elapsed) for t in byTime],
                                title='Tests')

    for name,code in (('Expected Passes    ', Test.PASS),
                      ('Expected Failures  ', Test.XFAIL),
                      ('Unsupported Tests  ', Test.UNSUPPORTED),
                      ('Unresolved Tests   ', Test.UNRESOLVED),
                      ('Unexpected Passes  ', Test.XPASS),
                      ('Unexpected Failures', Test.FAIL),):
        if opts.quiet and not code.isFailure:
            continue
        N = len(byCode.get(code,[]))
        if N:
            print '  %s: %d' % (name,N)

    # If we encountered any additional errors, exit abnormally.
    if litConfig.numErrors:
        print >>sys.stderr, '\n%d error(s), exiting.' % litConfig.numErrors
        sys.exit(2)

    # Warn about warnings.
    if litConfig.numWarnings:
        print >>sys.stderr, '\n%d warning(s) in tests.' % litConfig.numWarnings

    if hasFailures:
        sys.exit(1)
    sys.exit(0)

if __name__=='__main__':
    # Bump the GIL check interval, its more important to get any one thread to a
    # blocking operation (hopefully exec) than to try and unblock other threads.
    import sys
    sys.setcheckinterval(1000)
    main()
