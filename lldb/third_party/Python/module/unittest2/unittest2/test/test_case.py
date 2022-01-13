import difflib
import pprint
import re
import six

from copy import deepcopy

import unittest2

from unittest2.test.support import (
    OldTestResult, EqualityMixin, HashingMixin, LoggingResult
)


class MyException(Exception):
    pass


class Test(object):
    "Keep these TestCase classes out of the main namespace"

    class Foo(unittest2.TestCase):

        def runTest(self): pass

        def test1(self): pass

    class Bar(Foo):

        def test2(self): pass

    class LoggingTestCase(unittest2.TestCase):
        """A test case which logs its calls."""

        def __init__(self, events):
            super(Test.LoggingTestCase, self).__init__('test')
            self.events = events

        def setUp(self):
            self.events.append('setUp')

        def test(self):
            self.events.append('test')

        def tearDown(self):
            self.events.append('tearDown')


class TestCleanUp(unittest2.TestCase):

    def testCleanUp(self):
        class TestableTest(unittest2.TestCase):

            def testNothing(self):
                pass

        test = TestableTest('testNothing')
        self.assertEqual(test._cleanups, [])

        cleanups = []

        def cleanup1(*args, **kwargs):
            cleanups.append((1, args, kwargs))

        def cleanup2(*args, **kwargs):
            cleanups.append((2, args, kwargs))

        test.addCleanup(cleanup1, 1, 2, 3, four='hello', five='goodbye')
        test.addCleanup(cleanup2)

        self.assertEqual(
            test._cleanups, [
                (cleanup1, (1, 2, 3), dict(
                    four='hello', five='goodbye')), (cleanup2, (), {})])

        result = test.doCleanups()
        self.assertTrue(result)

        self.assertEqual(
            cleanups, [
                (2, (), {}), (1, (1, 2, 3), dict(
                    four='hello', five='goodbye'))])

    def testCleanUpWithErrors(self):
        class TestableTest(unittest2.TestCase):

            def testNothing(self):
                pass

        class MockResult(object):
            errors = []

            def addError(self, test, exc_info):
                self.errors.append((test, exc_info))

        result = MockResult()
        test = TestableTest('testNothing')
        test._resultForDoCleanups = result

        exc1 = Exception('foo')
        exc2 = Exception('bar')

        def cleanup1():
            raise exc1

        def cleanup2():
            raise exc2

        test.addCleanup(cleanup1)
        test.addCleanup(cleanup2)

        self.assertFalse(test.doCleanups())

        (test1, (Type1, instance1, _)), (test2,
                                         (Type2, instance2, _)) = reversed(MockResult.errors)
        self.assertEqual((test1, Type1, instance1), (test, Exception, exc1))
        self.assertEqual((test2, Type2, instance2), (test, Exception, exc2))

    def testCleanupInRun(self):
        blowUp = False
        ordering = []

        class TestableTest(unittest2.TestCase):

            def setUp(self):
                ordering.append('setUp')
                if blowUp:
                    raise Exception('foo')

            def testNothing(self):
                ordering.append('test')

            def tearDown(self):
                ordering.append('tearDown')

        test = TestableTest('testNothing')

        def cleanup1():
            ordering.append('cleanup1')

        def cleanup2():
            ordering.append('cleanup2')
        test.addCleanup(cleanup1)
        test.addCleanup(cleanup2)

        def success(some_test):
            self.assertEqual(some_test, test)
            ordering.append('success')

        result = unittest2.TestResult()
        result.addSuccess = success

        test.run(result)
        self.assertEqual(ordering, ['setUp', 'test', 'tearDown',
                                    'cleanup2', 'cleanup1', 'success'])

        blowUp = True
        ordering = []
        test = TestableTest('testNothing')
        test.addCleanup(cleanup1)
        test.run(result)
        self.assertEqual(ordering, ['setUp', 'cleanup1'])

    def testTestCaseDebugExecutesCleanups(self):
        ordering = []

        class TestableTest(unittest2.TestCase):

            def setUp(self):
                ordering.append('setUp')
                self.addCleanup(cleanup1)

            def testNothing(self):
                ordering.append('test')

            def tearDown(self):
                ordering.append('tearDown')

        test = TestableTest('testNothing')

        def cleanup1():
            ordering.append('cleanup1')
            test.addCleanup(cleanup2)

        def cleanup2():
            ordering.append('cleanup2')

        test.debug()
        self.assertEqual(
            ordering, [
                'setUp', 'test', 'tearDown', 'cleanup1', 'cleanup2'])


class Test_TestCase(unittest2.TestCase, EqualityMixin, HashingMixin):

    # Set up attributes used by inherited tests
    ################################################################

    # Used by HashingMixin.test_hash and EqualityMixin.test_eq
    eq_pairs = [(Test.Foo('test1'), Test.Foo('test1'))]

    # Used by EqualityMixin.test_ne
    ne_pairs = [(Test.Foo('test1'), Test.Foo('runTest')),
                (Test.Foo('test1'), Test.Bar('test1')),
                (Test.Foo('test1'), Test.Bar('test2'))]

    ################################################################
    # /Set up attributes used by inherited tests

    # "class TestCase([methodName])"
    # ...
    # "Each instance of TestCase will run a single test method: the
    # method named methodName."
    # ...
    # "methodName defaults to "runTest"."
    #
    # Make sure it really is optional, and that it defaults to the proper
    # thing.
    def test_init__no_test_name(self):
        class Test(unittest2.TestCase):

            def runTest(self): raise MyException()

            def test(self): pass

        self.assertEqual(Test().id()[-13:], '.Test.runTest')

    # "class TestCase([methodName])"
    # ...
    # "Each instance of TestCase will run a single test method: the
    # method named methodName."
    def test_init__test_name__valid(self):
        class Test(unittest2.TestCase):

            def runTest(self): raise MyException()

            def test(self): pass

        self.assertEqual(Test('test').id()[-10:], '.Test.test')

    # "class unittest2.TestCase([methodName])"
    # ...
    # "Each instance of TestCase will run a single test method: the
    # method named methodName."
    def test_init__test_name__invalid(self):
        class Test(unittest2.TestCase):

            def runTest(self): raise MyException()

            def test(self): pass

        try:
            Test('testfoo')
        except ValueError:
            pass
        else:
            self.fail("Failed to raise ValueError")

    # "Return the number of tests represented by the this test object. For
    # TestCase instances, this will always be 1"
    def test_countTestCases(self):
        class Foo(unittest2.TestCase):

            def test(self): pass

        self.assertEqual(Foo('test').countTestCases(), 1)

    # "Return the default type of test result object to be used to run this
    # test. For TestCase instances, this will always be
    # unittest2.TestResult;  subclasses of TestCase should
    # override this as necessary."
    def test_defaultTestResult(self):
        class Foo(unittest2.TestCase):

            def runTest(self):
                pass

        result = Foo().defaultTestResult()
        self.assertEqual(type(result), unittest2.TestResult)

    # "When a setUp() method is defined, the test runner will run that method
    # prior to each test. Likewise, if a tearDown() method is defined, the
    # test runner will invoke that method after each test. In the example,
    # setUp() was used to create a fresh sequence for each test."
    #
    # Make sure the proper call order is maintained, even if setUp() raises
    # an exception.
    def test_run_call_order__error_in_setUp(self):
        events = []
        result = LoggingResult(events)

        class Foo(Test.LoggingTestCase):

            def setUp(self):
                super(Foo, self).setUp()
                raise RuntimeError('raised by Foo.setUp')

        Foo(events).run(result)
        expected = ['startTest', 'setUp', 'addError', 'stopTest']
        self.assertEqual(events, expected)

    # "With a temporary result stopTestRun is called when setUp errors.
    def test_run_call_order__error_in_setUp_default_result(self):
        events = []

        class Foo(Test.LoggingTestCase):

            def defaultTestResult(self):
                return LoggingResult(self.events)

            def setUp(self):
                super(Foo, self).setUp()
                raise RuntimeError('raised by Foo.setUp')

        Foo(events).run()
        expected = ['startTestRun', 'startTest', 'setUp', 'addError',
                    'stopTest', 'stopTestRun']
        self.assertEqual(events, expected)

    # "When a setUp() method is defined, the test runner will run that method
    # prior to each test. Likewise, if a tearDown() method is defined, the
    # test runner will invoke that method after each test. In the example,
    # setUp() was used to create a fresh sequence for each test."
    #
    # Make sure the proper call order is maintained, even if the test raises
    # an error (as opposed to a failure).
    def test_run_call_order__error_in_test(self):
        events = []
        result = LoggingResult(events)

        class Foo(Test.LoggingTestCase):

            def test(self):
                super(Foo, self).test()
                raise RuntimeError('raised by Foo.test')

        expected = ['startTest', 'setUp', 'test', 'addError', 'tearDown',
                    'stopTest']
        Foo(events).run(result)
        self.assertEqual(events, expected)

    # "With a default result, an error in the test still results in stopTestRun
    # being called."
    def test_run_call_order__error_in_test_default_result(self):
        events = []

        class Foo(Test.LoggingTestCase):

            def defaultTestResult(self):
                return LoggingResult(self.events)

            def test(self):
                super(Foo, self).test()
                raise RuntimeError('raised by Foo.test')

        expected = ['startTestRun', 'startTest', 'setUp', 'test', 'addError',
                    'tearDown', 'stopTest', 'stopTestRun']
        Foo(events).run()
        self.assertEqual(events, expected)

    # "When a setUp() method is defined, the test runner will run that method
    # prior to each test. Likewise, if a tearDown() method is defined, the
    # test runner will invoke that method after each test. In the example,
    # setUp() was used to create a fresh sequence for each test."
    #
    # Make sure the proper call order is maintained, even if the test signals
    # a failure (as opposed to an error).
    def test_run_call_order__failure_in_test(self):
        events = []
        result = LoggingResult(events)

        class Foo(Test.LoggingTestCase):

            def test(self):
                super(Foo, self).test()
                self.fail('raised by Foo.test')

        expected = ['startTest', 'setUp', 'test', 'addFailure', 'tearDown',
                    'stopTest']
        Foo(events).run(result)
        self.assertEqual(events, expected)

    # "When a test fails with a default result stopTestRun is still called."
    def test_run_call_order__failure_in_test_default_result(self):

        class Foo(Test.LoggingTestCase):

            def defaultTestResult(self):
                return LoggingResult(self.events)

            def test(self):
                super(Foo, self).test()
                self.fail('raised by Foo.test')

        expected = ['startTestRun', 'startTest', 'setUp', 'test', 'addFailure',
                    'tearDown', 'stopTest', 'stopTestRun']
        events = []
        Foo(events).run()
        self.assertEqual(events, expected)

    # "When a setUp() method is defined, the test runner will run that method
    # prior to each test. Likewise, if a tearDown() method is defined, the
    # test runner will invoke that method after each test. In the example,
    # setUp() was used to create a fresh sequence for each test."
    #
    # Make sure the proper call order is maintained, even if tearDown() raises
    # an exception.
    def test_run_call_order__error_in_tearDown(self):
        events = []
        result = LoggingResult(events)

        class Foo(Test.LoggingTestCase):

            def tearDown(self):
                super(Foo, self).tearDown()
                raise RuntimeError('raised by Foo.tearDown')

        Foo(events).run(result)
        expected = ['startTest', 'setUp', 'test', 'tearDown', 'addError',
                    'stopTest']
        self.assertEqual(events, expected)

    # "When tearDown errors with a default result stopTestRun is still called."
    def test_run_call_order__error_in_tearDown_default_result(self):

        class Foo(Test.LoggingTestCase):

            def defaultTestResult(self):
                return LoggingResult(self.events)

            def tearDown(self):
                super(Foo, self).tearDown()
                raise RuntimeError('raised by Foo.tearDown')

        events = []
        Foo(events).run()
        expected = ['startTestRun', 'startTest', 'setUp', 'test', 'tearDown',
                    'addError', 'stopTest', 'stopTestRun']
        self.assertEqual(events, expected)

    # "TestCase.run() still works when the defaultTestResult is a TestResult
    # that does not support startTestRun and stopTestRun.
    def test_run_call_order_default_result(self):

        class Foo(unittest2.TestCase):

            def defaultTestResult(self):
                return OldTestResult()

            def test(self):
                pass

        Foo('test').run()

    # "This class attribute gives the exception raised by the test() method.
    # If a test framework needs to use a specialized exception, possibly to
    # carry additional information, it must subclass this exception in
    # order to ``play fair'' with the framework.  The initial value of this
    # attribute is AssertionError"
    def test_failureException__default(self):
        class Foo(unittest2.TestCase):

            def test(self):
                pass

        self.assertTrue(Foo('test').failureException is AssertionError)

    # "This class attribute gives the exception raised by the test() method.
    # If a test framework needs to use a specialized exception, possibly to
    # carry additional information, it must subclass this exception in
    # order to ``play fair'' with the framework."
    #
    # Make sure TestCase.run() respects the designated failureException
    def test_failureException__subclassing__explicit_raise(self):
        events = []
        result = LoggingResult(events)

        class Foo(unittest2.TestCase):

            def test(self):
                raise RuntimeError()

            failureException = RuntimeError

        self.assertTrue(Foo('test').failureException is RuntimeError)

        Foo('test').run(result)
        expected = ['startTest', 'addFailure', 'stopTest']
        self.assertEqual(events, expected)

    # "This class attribute gives the exception raised by the test() method.
    # If a test framework needs to use a specialized exception, possibly to
    # carry additional information, it must subclass this exception in
    # order to ``play fair'' with the framework."
    #
    # Make sure TestCase.run() respects the designated failureException
    def test_failureException__subclassing__implicit_raise(self):
        events = []
        result = LoggingResult(events)

        class Foo(unittest2.TestCase):

            def test(self):
                self.fail("foo")

            failureException = RuntimeError

        self.assertTrue(Foo('test').failureException is RuntimeError)

        Foo('test').run(result)
        expected = ['startTest', 'addFailure', 'stopTest']
        self.assertEqual(events, expected)

    # "The default implementation does nothing."
    def test_setUp(self):
        class Foo(unittest2.TestCase):

            def runTest(self):
                pass

        # ... and nothing should happen
        Foo().setUp()

    # "The default implementation does nothing."
    def test_tearDown(self):
        class Foo(unittest2.TestCase):

            def runTest(self):
                pass

        # ... and nothing should happen
        Foo().tearDown()

    # "Return a string identifying the specific test case."
    #
    # Because of the vague nature of the docs, I'm not going to lock this
    # test down too much. Really all that can be asserted is that the id()
    # will be a string (either 8-byte or unicode -- again, because the docs
    # just say "string")
    def test_id(self):
        class Foo(unittest2.TestCase):

            def runTest(self):
                pass

        self.assertIsInstance(Foo().id(), six.string_types)

    # "If result is omitted or None, a temporary result object is created
    # and used, but is not made available to the caller. As TestCase owns the
    # temporary result startTestRun and stopTestRun are called.

    def test_run__uses_defaultTestResult(self):
        events = []

        class Foo(unittest2.TestCase):

            def test(self):
                events.append('test')

            def defaultTestResult(self):
                return LoggingResult(events)

        # Make run() find a result object on its own
        Foo('test').run()

        expected = ['startTestRun', 'startTest', 'test', 'addSuccess',
                    'stopTest', 'stopTestRun']
        self.assertEqual(events, expected)

    def testShortDescriptionWithoutDocstring(self):
        self.assertIsNone(self.shortDescription())

    def testShortDescriptionWithOneLineDocstring(self):
        """Tests shortDescription() for a method with a docstring."""
        self.assertEqual(
            self.shortDescription(),
            'Tests shortDescription() for a method with a docstring.')

    def testShortDescriptionWithMultiLineDocstring(self):
        """Tests shortDescription() for a method with a longer docstring.

        This method ensures that only the first line of a docstring is
        returned used in the short description, no matter how long the
        whole thing is.
        """
        self.assertEqual(
            self.shortDescription(),
            'Tests shortDescription() for a method with a longer '
            'docstring.')

    def testAddTypeEqualityFunc(self):
        class SadSnake(object):
            """Dummy class for test_addTypeEqualityFunc."""
        s1, s2 = SadSnake(), SadSnake()
        self.assertNotEqual(s1, s2)

        def AllSnakesCreatedEqual(a, b, msg=None):
            return type(a) is type(b) is SadSnake
        self.addTypeEqualityFunc(SadSnake, AllSnakesCreatedEqual)
        self.assertEqual(s1, s2)
        # No this doesn't clean up and remove the SadSnake equality func
        # from this TestCase instance but since its a local nothing else
        # will ever notice that.

    def testAssertIs(self):
        thing = object()
        self.assertIs(thing, thing)
        self.assertRaises(
            self.failureException,
            self.assertIs,
            thing,
            object())

    def testAssertIsNot(self):
        thing = object()
        self.assertIsNot(thing, object())
        self.assertRaises(
            self.failureException,
            self.assertIsNot,
            thing,
            thing)

    def testAssertIsInstance(self):
        thing = []
        self.assertIsInstance(thing, list)
        self.assertRaises(self.failureException, self.assertIsInstance,
                          thing, dict)

    def testAssertNotIsInstance(self):
        thing = []
        self.assertNotIsInstance(thing, dict)
        self.assertRaises(self.failureException, self.assertNotIsInstance,
                          thing, list)

    def testAssertIn(self):
        animals = {'monkey': 'banana', 'cow': 'grass', 'seal': 'fish'}

        self.assertIn('a', 'abc')
        self.assertIn(2, [1, 2, 3])
        self.assertIn('monkey', animals)

        self.assertNotIn('d', 'abc')
        self.assertNotIn(0, [1, 2, 3])
        self.assertNotIn('otter', animals)

        self.assertRaises(self.failureException, self.assertIn, 'x', 'abc')
        self.assertRaises(self.failureException, self.assertIn, 4, [1, 2, 3])
        self.assertRaises(self.failureException, self.assertIn, 'elephant',
                          animals)

        self.assertRaises(self.failureException, self.assertNotIn, 'c', 'abc')
        self.assertRaises(
            self.failureException, self.assertNotIn, 1, [
                1, 2, 3])
        self.assertRaises(self.failureException, self.assertNotIn, 'cow',
                          animals)

    def testAssertDictContainsSubset(self):
        self.assertDictContainsSubset({}, {})
        self.assertDictContainsSubset({}, {'a': 1})
        self.assertDictContainsSubset({'a': 1}, {'a': 1})
        self.assertDictContainsSubset({'a': 1}, {'a': 1, 'b': 2})
        self.assertDictContainsSubset({'a': 1, 'b': 2}, {'a': 1, 'b': 2})

        self.assertRaises(unittest2.TestCase.failureException,
                          self.assertDictContainsSubset, {'a': 2}, {'a': 1},
                          '.*Mismatched values:.*')

        self.assertRaises(unittest2.TestCase.failureException,
                          self.assertDictContainsSubset, {'c': 1}, {'a': 1},
                          '.*Missing:.*')

        self.assertRaises(unittest2.TestCase.failureException,
                          self.assertDictContainsSubset, {'a': 1, 'c': 1},
                          {'a': 1}, '.*Missing:.*')

        self.assertRaises(unittest2.TestCase.failureException,
                          self.assertDictContainsSubset, {'a': 1, 'c': 1},
                          {'a': 1}, '.*Missing:.*Mismatched values:.*')

        self.assertRaises(self.failureException,
                          self.assertDictContainsSubset, {1: "one"}, {})

    def testAssertEqual(self):
        equal_pairs = [
            ((), ()),
            ({}, {}),
            ([], []),
            (set(), set()),
            (frozenset(), frozenset())]
        for a, b in equal_pairs:
            # This mess of try excepts is to test the assertEqual behavior
            # itself.
            try:
                self.assertEqual(a, b)
            except self.failureException:
                self.fail('assertEqual(%r, %r) failed' % (a, b))
            try:
                self.assertEqual(a, b, msg='foo')
            except self.failureException:
                self.fail('assertEqual(%r, %r) with msg= failed' % (a, b))
            try:
                self.assertEqual(a, b, 'foo')
            except self.failureException:
                self.fail('assertEqual(%r, %r) with third parameter failed' %
                          (a, b))

        unequal_pairs = [
            ((), []),
            ({}, set()),
            (set([4, 1]), frozenset([4, 2])),
            (frozenset([4, 5]), set([2, 3])),
            (set([3, 4]), set([5, 4]))]
        for a, b in unequal_pairs:
            self.assertRaises(self.failureException, self.assertEqual, a, b)
            self.assertRaises(self.failureException, self.assertEqual, a, b,
                              'foo')
            self.assertRaises(self.failureException, self.assertEqual, a, b,
                              msg='foo')

    def testEquality(self):
        self.assertListEqual([], [])
        self.assertTupleEqual((), ())
        self.assertSequenceEqual([], ())

        a = [0, 'a', []]
        b = []
        self.assertRaises(unittest2.TestCase.failureException,
                          self.assertListEqual, a, b)
        self.assertRaises(unittest2.TestCase.failureException,
                          self.assertListEqual, tuple(a), tuple(b))
        self.assertRaises(unittest2.TestCase.failureException,
                          self.assertSequenceEqual, a, tuple(b))

        b.extend(a)
        self.assertListEqual(a, b)
        self.assertTupleEqual(tuple(a), tuple(b))
        self.assertSequenceEqual(a, tuple(b))
        self.assertSequenceEqual(tuple(a), b)

        self.assertRaises(self.failureException, self.assertListEqual,
                          a, tuple(b))
        self.assertRaises(self.failureException, self.assertTupleEqual,
                          tuple(a), b)
        self.assertRaises(self.failureException, self.assertListEqual, None, b)
        self.assertRaises(self.failureException, self.assertTupleEqual, None,
                          tuple(b))
        self.assertRaises(self.failureException, self.assertSequenceEqual,
                          None, tuple(b))
        self.assertRaises(self.failureException, self.assertListEqual, 1, 1)
        self.assertRaises(self.failureException, self.assertTupleEqual, 1, 1)
        self.assertRaises(self.failureException, self.assertSequenceEqual,
                          1, 1)

        self.assertDictEqual({}, {})

        c = {'x': 1}
        d = {}
        self.assertRaises(unittest2.TestCase.failureException,
                          self.assertDictEqual, c, d)

        d.update(c)
        self.assertDictEqual(c, d)

        d['x'] = 0
        self.assertRaises(unittest2.TestCase.failureException,
                          self.assertDictEqual, c, d, 'These are unequal')

        self.assertRaises(self.failureException, self.assertDictEqual, None, d)
        self.assertRaises(self.failureException, self.assertDictEqual, [], d)
        self.assertRaises(self.failureException, self.assertDictEqual, 1, 1)

    def testAssertItemsEqual(self):
        self.assertItemsEqual([1, 2, 3], [3, 2, 1])
        self.assertItemsEqual(['foo', 'bar', 'baz'], ['bar', 'baz', 'foo'])
        self.assertRaises(self.failureException, self.assertItemsEqual,
                          [10], [10, 11])
        self.assertRaises(self.failureException, self.assertItemsEqual,
                          [10, 11], [10])
        self.assertRaises(self.failureException, self.assertItemsEqual,
                          [10, 11, 10], [10, 11])

        # Test that sequences of unhashable objects can be tested for sameness:
        self.assertItemsEqual([[1, 2], [3, 4]], [[3, 4], [1, 2]])

        self.assertItemsEqual([{'a': 1}, {'b': 2}], [{'b': 2}, {'a': 1}])
        self.assertRaises(self.failureException, self.assertItemsEqual,
                          [[1]], [[2]])

        # Test unsortable objects
        self.assertItemsEqual([2j, None], [None, 2j])
        self.assertRaises(self.failureException, self.assertItemsEqual,
                          [2j, None], [None, 3j])

    def testAssertSetEqual(self):
        set1 = set()
        set2 = set()
        self.assertSetEqual(set1, set2)

        self.assertRaises(
            self.failureException,
            self.assertSetEqual,
            None,
            set2)
        self.assertRaises(self.failureException, self.assertSetEqual, [], set2)
        self.assertRaises(
            self.failureException,
            self.assertSetEqual,
            set1,
            None)
        self.assertRaises(self.failureException, self.assertSetEqual, set1, [])

        set1 = set(['a'])
        set2 = set()
        self.assertRaises(
            self.failureException,
            self.assertSetEqual,
            set1,
            set2)

        set1 = set(['a'])
        set2 = set(['a'])
        self.assertSetEqual(set1, set2)

        set1 = set(['a'])
        set2 = set(['a', 'b'])
        self.assertRaises(
            self.failureException,
            self.assertSetEqual,
            set1,
            set2)

        set1 = set(['a'])
        set2 = frozenset(['a', 'b'])
        self.assertRaises(
            self.failureException,
            self.assertSetEqual,
            set1,
            set2)

        set1 = set(['a', 'b'])
        set2 = frozenset(['a', 'b'])
        self.assertSetEqual(set1, set2)

        set1 = set()
        set2 = "foo"
        self.assertRaises(
            self.failureException,
            self.assertSetEqual,
            set1,
            set2)
        self.assertRaises(
            self.failureException,
            self.assertSetEqual,
            set2,
            set1)

        # make sure any string formatting is tuple-safe
        set1 = set([(0, 1), (2, 3)])
        set2 = set([(4, 5)])
        self.assertRaises(
            self.failureException,
            self.assertSetEqual,
            set1,
            set2)

    def testInequality(self):
        # Try ints
        self.assertGreater(2, 1)
        self.assertGreaterEqual(2, 1)
        self.assertGreaterEqual(1, 1)
        self.assertLess(1, 2)
        self.assertLessEqual(1, 2)
        self.assertLessEqual(1, 1)
        self.assertRaises(self.failureException, self.assertGreater, 1, 2)
        self.assertRaises(self.failureException, self.assertGreater, 1, 1)
        self.assertRaises(self.failureException, self.assertGreaterEqual, 1, 2)
        self.assertRaises(self.failureException, self.assertLess, 2, 1)
        self.assertRaises(self.failureException, self.assertLess, 1, 1)
        self.assertRaises(self.failureException, self.assertLessEqual, 2, 1)

        # Try Floats
        self.assertGreater(1.1, 1.0)
        self.assertGreaterEqual(1.1, 1.0)
        self.assertGreaterEqual(1.0, 1.0)
        self.assertLess(1.0, 1.1)
        self.assertLessEqual(1.0, 1.1)
        self.assertLessEqual(1.0, 1.0)
        self.assertRaises(self.failureException, self.assertGreater, 1.0, 1.1)
        self.assertRaises(self.failureException, self.assertGreater, 1.0, 1.0)
        self.assertRaises(
            self.failureException,
            self.assertGreaterEqual,
            1.0,
            1.1)
        self.assertRaises(self.failureException, self.assertLess, 1.1, 1.0)
        self.assertRaises(self.failureException, self.assertLess, 1.0, 1.0)
        self.assertRaises(
            self.failureException,
            self.assertLessEqual,
            1.1,
            1.0)

        # Try Strings
        self.assertGreater('bug', 'ant')
        self.assertGreaterEqual('bug', 'ant')
        self.assertGreaterEqual('ant', 'ant')
        self.assertLess('ant', 'bug')
        self.assertLessEqual('ant', 'bug')
        self.assertLessEqual('ant', 'ant')
        self.assertRaises(
            self.failureException,
            self.assertGreater,
            'ant',
            'bug')
        self.assertRaises(
            self.failureException,
            self.assertGreater,
            'ant',
            'ant')
        self.assertRaises(
            self.failureException,
            self.assertGreaterEqual,
            'ant',
            'bug')
        self.assertRaises(self.failureException, self.assertLess, 'bug', 'ant')
        self.assertRaises(self.failureException, self.assertLess, 'ant', 'ant')
        self.assertRaises(
            self.failureException,
            self.assertLessEqual,
            'bug',
            'ant')

        # Try Unicode
        self.assertGreater(u'bug', u'ant')
        self.assertGreaterEqual(u'bug', u'ant')
        self.assertGreaterEqual(u'ant', u'ant')
        self.assertLess(u'ant', u'bug')
        self.assertLessEqual(u'ant', u'bug')
        self.assertLessEqual(u'ant', u'ant')
        self.assertRaises(
            self.failureException,
            self.assertGreater,
            u'ant',
            u'bug')
        self.assertRaises(
            self.failureException,
            self.assertGreater,
            u'ant',
            u'ant')
        self.assertRaises(
            self.failureException,
            self.assertGreaterEqual,
            u'ant',
            u'bug')
        self.assertRaises(
            self.failureException,
            self.assertLess,
            u'bug',
            u'ant')
        self.assertRaises(
            self.failureException,
            self.assertLess,
            u'ant',
            u'ant')
        self.assertRaises(
            self.failureException,
            self.assertLessEqual,
            u'bug',
            u'ant')

        # Try Mixed String/Unicode
        self.assertGreater('bug', u'ant')
        self.assertGreater(u'bug', 'ant')
        self.assertGreaterEqual('bug', u'ant')
        self.assertGreaterEqual(u'bug', 'ant')
        self.assertGreaterEqual('ant', u'ant')
        self.assertGreaterEqual(u'ant', 'ant')
        self.assertLess('ant', u'bug')
        self.assertLess(u'ant', 'bug')
        self.assertLessEqual('ant', u'bug')
        self.assertLessEqual(u'ant', 'bug')
        self.assertLessEqual('ant', u'ant')
        self.assertLessEqual(u'ant', 'ant')
        self.assertRaises(
            self.failureException,
            self.assertGreater,
            'ant',
            u'bug')
        self.assertRaises(
            self.failureException,
            self.assertGreater,
            u'ant',
            'bug')
        self.assertRaises(
            self.failureException,
            self.assertGreater,
            'ant',
            u'ant')
        self.assertRaises(
            self.failureException,
            self.assertGreater,
            u'ant',
            'ant')
        self.assertRaises(
            self.failureException,
            self.assertGreaterEqual,
            'ant',
            u'bug')
        self.assertRaises(
            self.failureException,
            self.assertGreaterEqual,
            u'ant',
            'bug')
        self.assertRaises(
            self.failureException,
            self.assertLess,
            'bug',
            u'ant')
        self.assertRaises(
            self.failureException,
            self.assertLess,
            u'bug',
            'ant')
        self.assertRaises(
            self.failureException,
            self.assertLess,
            'ant',
            u'ant')
        self.assertRaises(
            self.failureException,
            self.assertLess,
            u'ant',
            'ant')
        self.assertRaises(
            self.failureException,
            self.assertLessEqual,
            'bug',
            u'ant')
        self.assertRaises(
            self.failureException,
            self.assertLessEqual,
            u'bug',
            'ant')

    def testAssertMultiLineEqual(self):
        sample_text = """\
http://www.python.org/doc/2.3/lib/module-unittest.html
test case
    A test case is the smallest unit of testing. [...]
"""
        revised_sample_text = """\
http://www.python.org/doc/2.4.1/lib/module-unittest.html
test case
    A test case is the smallest unit of testing. [...] You may provide your
    own implementation that does not subclass from TestCase, of course.
"""
        sample_text_error = """\
- http://www.python.org/doc/2.3/lib/module-unittest.html
?                             ^
+ http://www.python.org/doc/2.4.1/lib/module-unittest.html
?                             ^^^
  test case
-     A test case is the smallest unit of testing. [...]
+     A test case is the smallest unit of testing. [...] You may provide your
?                                                       +++++++++++++++++++++
+     own implementation that does not subclass from TestCase, of course.
"""
        self.maxDiff = None
        for type_changer in (lambda x: x, lambda x: x.decode('utf8')):
            try:
                self.assertMultiLineEqual(type_changer(sample_text),
                                          type_changer(revised_sample_text))
            except self.failureException as e:
                # need to remove the first line of the error message
                error = str(e).encode('utf8').split('\n', 1)[1]

                # assertMultiLineEqual is hooked up as the default for
                # unicode strings - so we can't use it for this check
                self.assertTrue(sample_text_error == error)

    def testAssertSequenceEqualMaxDiff(self):
        self.assertEqual(self.maxDiff, 80 * 8)
        seq1 = 'a' + 'x' * 80**2
        seq2 = 'b' + 'x' * 80**2
        diff = '\n'.join(difflib.ndiff(pprint.pformat(seq1).splitlines(),
                                       pprint.pformat(seq2).splitlines()))
        # the +1 is the leading \n added by assertSequenceEqual
        omitted = unittest2.case.DIFF_OMITTED % (len(diff) + 1,)

        self.maxDiff = len(diff) // 2
        try:
            self.assertSequenceEqual(seq1, seq2)
        except self.failureException as e:
            msg = e.args[0]
        else:
            self.fail('assertSequenceEqual did not fail.')
        self.assertTrue(len(msg) < len(diff))
        self.assertIn(omitted, msg)

        self.maxDiff = len(diff) * 2
        try:
            self.assertSequenceEqual(seq1, seq2)
        except self.failureException as e:
            msg = e.args[0]
        else:
            self.fail('assertSequenceEqual did not fail.')
        self.assertTrue(len(msg) > len(diff))
        self.assertNotIn(omitted, msg)

        self.maxDiff = None
        try:
            self.assertSequenceEqual(seq1, seq2)
        except self.failureException as e:
            msg = e.args[0]
        else:
            self.fail('assertSequenceEqual did not fail.')
        self.assertTrue(len(msg) > len(diff))
        self.assertNotIn(omitted, msg)

    def testTruncateMessage(self):
        self.maxDiff = 1
        message = self._truncateMessage('foo', 'bar')
        omitted = unittest2.case.DIFF_OMITTED % len('bar')
        self.assertEqual(message, 'foo' + omitted)

        self.maxDiff = None
        message = self._truncateMessage('foo', 'bar')
        self.assertEqual(message, 'foobar')

        self.maxDiff = 4
        message = self._truncateMessage('foo', 'bar')
        self.assertEqual(message, 'foobar')

    def testAssertDictEqualTruncates(self):
        test = unittest2.TestCase('assertEqual')

        def truncate(msg, diff):
            return 'foo'
        test._truncateMessage = truncate
        try:
            test.assertDictEqual({}, {1: 0})
        except self.failureException as e:
            self.assertEqual(str(e), 'foo')
        else:
            self.fail('assertDictEqual did not fail')

    def testAssertMultiLineEqualTruncates(self):
        test = unittest2.TestCase('assertEqual')

        def truncate(msg, diff):
            return 'foo'
        test._truncateMessage = truncate
        try:
            test.assertMultiLineEqual('foo', 'bar')
        except self.failureException as e:
            self.assertEqual(str(e), 'foo')
        else:
            self.fail('assertMultiLineEqual did not fail')

    def testAssertIsNone(self):
        self.assertIsNone(None)
        self.assertRaises(self.failureException, self.assertIsNone, False)
        self.assertIsNotNone('DjZoPloGears on Rails')
        self.assertRaises(self.failureException, self.assertIsNotNone, None)

    def testAssertRegexpMatches(self):
        self.assertRegexpMatches('asdfabasdf', r'ab+')
        self.assertRaises(self.failureException, self.assertRegexpMatches,
                          'saaas', r'aaaa')

    def testAssertRaisesRegexp(self):
        class ExceptionMock(Exception):
            pass

        def Stub():
            raise ExceptionMock('We expect')

        self.assertRaisesRegexp(ExceptionMock, re.compile('expect$'), Stub)
        self.assertRaisesRegexp(ExceptionMock, 'expect$', Stub)
        self.assertRaisesRegexp(ExceptionMock, u'expect$', Stub)

    def testAssertNotRaisesRegexp(self):
        self.assertRaisesRegexp(
            self.failureException, '^Exception not raised$',
            self.assertRaisesRegexp, Exception, re.compile('x'),
            lambda: None)
        self.assertRaisesRegexp(
            self.failureException, '^Exception not raised$',
            self.assertRaisesRegexp, Exception, 'x',
            lambda: None)
        self.assertRaisesRegexp(
            self.failureException, '^Exception not raised$',
            self.assertRaisesRegexp, Exception, u'x',
            lambda: None)

    def testAssertRaisesRegexpMismatch(self):
        def Stub():
            raise Exception('Unexpected')

        self.assertRaisesRegexp(
            self.failureException,
            r'"\^Expected\$" does not match "Unexpected"',
            self.assertRaisesRegexp, Exception, '^Expected$',
            Stub)
        self.assertRaisesRegexp(
            self.failureException,
            r'"\^Expected\$" does not match "Unexpected"',
            self.assertRaisesRegexp, Exception, u'^Expected$',
            Stub)
        self.assertRaisesRegexp(
            self.failureException,
            r'"\^Expected\$" does not match "Unexpected"',
            self.assertRaisesRegexp, Exception,
            re.compile('^Expected$'), Stub)

    def testSynonymAssertMethodNames(self):
        """Test undocumented method name synonyms.

        Please do not use these methods names in your own code.

        This test confirms their continued existence and functionality
        in order to avoid breaking existing code.
        """
        self.assertNotEquals(3, 5)
        self.assertEquals(3, 3)
        self.assertAlmostEquals(2.0, 2.0)
        self.assertNotAlmostEquals(3.0, 5.0)
        self.assert_(True)

    def testDeepcopy(self):
        # Issue: 5660
        class TestableTest(unittest2.TestCase):

            def testNothing(self):
                pass

        test = TestableTest('testNothing')

        # This shouldn't blow up
        deepcopy(test)


if __name__ == "__main__":
    unittest2.main()
