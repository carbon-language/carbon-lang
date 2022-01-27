from unittest2.test.support import EqualityMixin, LoggingResult

import sys
import unittest2


class Test(object):

    class Foo(unittest2.TestCase):

        def test_1(self): pass

        def test_2(self): pass

        def test_3(self): pass

        def runTest(self): pass


def _mk_TestSuite(*names):
    return unittest2.TestSuite(Test.Foo(n) for n in names)


class Test_TestSuite(unittest2.TestCase, EqualityMixin):

    # Set up attributes needed by inherited tests
    ################################################################

    # Used by EqualityMixin.test_eq
    eq_pairs = [(unittest2.TestSuite(), unittest2.TestSuite()),
                (unittest2.TestSuite(), unittest2.TestSuite([])),
                (_mk_TestSuite('test_1'), _mk_TestSuite('test_1'))]

    # Used by EqualityMixin.test_ne
    ne_pairs = [(unittest2.TestSuite(), _mk_TestSuite('test_1')),
                (unittest2.TestSuite([]), _mk_TestSuite('test_1')),
                (_mk_TestSuite('test_1', 'test_2'), _mk_TestSuite('test_1', 'test_3')),
                (_mk_TestSuite('test_1'), _mk_TestSuite('test_2'))]

    ################################################################
    # /Set up attributes needed by inherited tests

    # Tests for TestSuite.__init__
    ################################################################

    # "class TestSuite([tests])"
    #
    # The tests iterable should be optional
    def test_init__tests_optional(self):
        suite = unittest2.TestSuite()

        self.assertEqual(suite.countTestCases(), 0)

    # "class TestSuite([tests])"
    # ...
    # "If tests is given, it must be an iterable of individual test cases
    # or other test suites that will be used to build the suite initially"
    #
    # TestSuite should deal with empty tests iterables by allowing the
    # creation of an empty suite
    def test_init__empty_tests(self):
        suite = unittest2.TestSuite([])

        self.assertEqual(suite.countTestCases(), 0)

    # "class TestSuite([tests])"
    # ...
    # "If tests is given, it must be an iterable of individual test cases
    # or other test suites that will be used to build the suite initially"
    #
    # TestSuite should allow any iterable to provide tests
    def test_init__tests_from_any_iterable(self):
        def tests():
            yield unittest2.FunctionTestCase(lambda: None)
            yield unittest2.FunctionTestCase(lambda: None)

        suite_1 = unittest2.TestSuite(tests())
        self.assertEqual(suite_1.countTestCases(), 2)

        suite_2 = unittest2.TestSuite(suite_1)
        self.assertEqual(suite_2.countTestCases(), 2)

        suite_3 = unittest2.TestSuite(set(suite_1))
        self.assertEqual(suite_3.countTestCases(), 2)

    # "class TestSuite([tests])"
    # ...
    # "If tests is given, it must be an iterable of individual test cases
    # or other test suites that will be used to build the suite initially"
    #
    # Does TestSuite() also allow other TestSuite() instances to be present
    # in the tests iterable?
    def test_init__TestSuite_instances_in_tests(self):
        def tests():
            ftc = unittest2.FunctionTestCase(lambda: None)
            yield unittest2.TestSuite([ftc])
            yield unittest2.FunctionTestCase(lambda: None)

        suite = unittest2.TestSuite(tests())
        self.assertEqual(suite.countTestCases(), 2)

    ################################################################
    # /Tests for TestSuite.__init__

    # Container types should support the iter protocol
    def test_iter(self):
        test1 = unittest2.FunctionTestCase(lambda: None)
        test2 = unittest2.FunctionTestCase(lambda: None)
        suite = unittest2.TestSuite((test1, test2))

        self.assertEqual(list(suite), [test1, test2])

    # "Return the number of tests represented by the this test object.
    # ...this method is also implemented by the TestSuite class, which can
    # return larger [greater than 1] values"
    #
    # Presumably an empty TestSuite returns 0?
    def test_countTestCases_zero_simple(self):
        suite = unittest2.TestSuite()

        self.assertEqual(suite.countTestCases(), 0)

    # "Return the number of tests represented by the this test object.
    # ...this method is also implemented by the TestSuite class, which can
    # return larger [greater than 1] values"
    #
    # Presumably an empty TestSuite (even if it contains other empty
    # TestSuite instances) returns 0?
    def test_countTestCases_zero_nested(self):
        class Test1(unittest2.TestCase):

            def test(self):
                pass

        suite = unittest2.TestSuite([unittest2.TestSuite()])

        self.assertEqual(suite.countTestCases(), 0)

    # "Return the number of tests represented by the this test object.
    # ...this method is also implemented by the TestSuite class, which can
    # return larger [greater than 1] values"
    def test_countTestCases_simple(self):
        test1 = unittest2.FunctionTestCase(lambda: None)
        test2 = unittest2.FunctionTestCase(lambda: None)
        suite = unittest2.TestSuite((test1, test2))

        self.assertEqual(suite.countTestCases(), 2)

    # "Return the number of tests represented by the this test object.
    # ...this method is also implemented by the TestSuite class, which can
    # return larger [greater than 1] values"
    #
    # Make sure this holds for nested TestSuite instances, too
    def test_countTestCases_nested(self):
        class Test1(unittest2.TestCase):

            def test1(self): pass

            def test2(self): pass

        test2 = unittest2.FunctionTestCase(lambda: None)
        test3 = unittest2.FunctionTestCase(lambda: None)
        child = unittest2.TestSuite((Test1('test2'), test2))
        parent = unittest2.TestSuite((test3, child, Test1('test1')))

        self.assertEqual(parent.countTestCases(), 4)

    # "Run the tests associated with this suite, collecting the result into
    # the test result object passed as result."
    #
    # And if there are no tests? What then?
    def test_run__empty_suite(self):
        events = []
        result = LoggingResult(events)

        suite = unittest2.TestSuite()

        suite.run(result)

        self.assertEqual(events, [])

    # "Note that unlike TestCase.run(), TestSuite.run() requires the
    # "result object to be passed in."
    def test_run__requires_result(self):
        suite = unittest2.TestSuite()

        try:
            suite.run()
        except TypeError:
            pass
        else:
            self.fail("Failed to raise TypeError")

    # "Run the tests associated with this suite, collecting the result into
    # the test result object passed as result."
    def test_run(self):
        events = []
        result = LoggingResult(events)

        class LoggingCase(unittest2.TestCase):

            def run(self, result):
                events.append('run %s' % self._testMethodName)

            def test1(self): pass

            def test2(self): pass

        tests = [LoggingCase('test1'), LoggingCase('test2')]

        unittest2.TestSuite(tests).run(result)

        self.assertEqual(events, ['run test1', 'run test2'])

    # "Add a TestCase ... to the suite"
    def test_addTest__TestCase(self):
        class Foo(unittest2.TestCase):

            def test(self): pass

        test = Foo('test')
        suite = unittest2.TestSuite()

        suite.addTest(test)

        self.assertEqual(suite.countTestCases(), 1)
        self.assertEqual(list(suite), [test])

    # "Add a ... TestSuite to the suite"
    def test_addTest__TestSuite(self):
        class Foo(unittest2.TestCase):

            def test(self): pass

        suite_2 = unittest2.TestSuite([Foo('test')])

        suite = unittest2.TestSuite()
        suite.addTest(suite_2)

        self.assertEqual(suite.countTestCases(), 1)
        self.assertEqual(list(suite), [suite_2])

    # "Add all the tests from an iterable of TestCase and TestSuite
    # instances to this test suite."
    #
    # "This is equivalent to iterating over tests, calling addTest() for
    # each element"
    def test_addTests(self):
        class Foo(unittest2.TestCase):

            def test_1(self): pass

            def test_2(self): pass

        test_1 = Foo('test_1')
        test_2 = Foo('test_2')
        inner_suite = unittest2.TestSuite([test_2])

        def gen():
            yield test_1
            yield test_2
            yield inner_suite

        suite_1 = unittest2.TestSuite()
        suite_1.addTests(gen())

        self.assertEqual(list(suite_1), list(gen()))

        # "This is equivalent to iterating over tests, calling addTest() for
        # each element"
        suite_2 = unittest2.TestSuite()
        for t in gen():
            suite_2.addTest(t)

        self.assertEqual(suite_1, suite_2)

    # "Add all the tests from an iterable of TestCase and TestSuite
    # instances to this test suite."
    #
    # What happens if it doesn't get an iterable?
    def test_addTest__noniterable(self):
        suite = unittest2.TestSuite()

        try:
            suite.addTests(5)
        except TypeError:
            pass
        else:
            self.fail("Failed to raise TypeError")

    def test_addTest__noncallable(self):
        suite = unittest2.TestSuite()
        self.assertRaises(TypeError, suite.addTest, 5)

    def test_addTest__casesuiteclass(self):
        suite = unittest2.TestSuite()
        self.assertRaises(TypeError, suite.addTest, Test_TestSuite)
        self.assertRaises(TypeError, suite.addTest, unittest2.TestSuite)

    def test_addTests__string(self):
        suite = unittest2.TestSuite()
        self.assertRaises(TypeError, suite.addTests, "foo")

    def test_function_in_suite(self):
        def f(_):
            pass
        suite = unittest2.TestSuite()
        suite.addTest(f)

        # when the bug is fixed this line will not crash
        suite.run(unittest2.TestResult())

    def test_basetestsuite(self):
        class Test(unittest2.TestCase):
            wasSetUp = False
            wasTornDown = False

            @classmethod
            def setUpClass(cls):
                cls.wasSetUp = True

            @classmethod
            def tearDownClass(cls):
                cls.wasTornDown = True

            def testPass(self):
                pass

            def testFail(self):
                fail

        class Module(object):
            wasSetUp = False
            wasTornDown = False

            @staticmethod
            def setUpModule():
                Module.wasSetUp = True

            @staticmethod
            def tearDownModule():
                Module.wasTornDown = True

        Test.__module__ = 'Module'
        sys.modules['Module'] = Module
        self.addCleanup(sys.modules.pop, 'Module')

        suite = unittest2.BaseTestSuite()
        suite.addTests([Test('testPass'), Test('testFail')])
        self.assertEqual(suite.countTestCases(), 2)

        result = unittest2.TestResult()
        suite.run(result)
        self.assertFalse(Module.wasSetUp)
        self.assertFalse(Module.wasTornDown)
        self.assertFalse(Test.wasSetUp)
        self.assertFalse(Test.wasTornDown)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(result.testsRun, 2)

if __name__ == '__main__':
    unittest2.main()
