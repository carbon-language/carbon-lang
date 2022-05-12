"""
unittest2

unittest2 is a backport of the new features added to the unittest testing
framework in Python 2.7. It is tested to run on Python 2.4 - 2.6.

To use unittest2 instead of unittest simply replace ``import unittest`` with
``import unittest2``.


Copyright (c) 1999-2003 Steve Purcell
Copyright (c) 2003-2010 Python Software Foundation
This module is free software, and you may redistribute it and/or modify
it under the same terms as Python itself, so long as this copyright message
and disclaimer are retained in their original form.

IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF
THIS CODE, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE.  THE CODE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THERE IS NO OBLIGATION WHATSOEVER TO PROVIDE MAINTENANCE,
SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""

import sys

if sys.version_info[0] >= 3:
    # Python 3 doesn't have the builtin `cmp` function anymore
    cmp_ = lambda x, y: (x > y) - (x < y)
else:
    cmp_ = cmp

reversed_cmp_ = lambda x, y: -cmp_(x, y)

__all__ = ['TestResult', 'TestCase', 'TestSuite',
           'TextTestRunner', 'TestLoader', 'FunctionTestCase', 'main',
           'defaultTestLoader', 'SkipTest', 'skip', 'skipIf', 'skipUnless',
           'expectedFailure', 'TextTestResult', '__version__', 'collector']

__version__ = '0.5.1'

# Expose obsolete functions for backwards compatibility
__all__.extend(['getTestCaseNames', 'makeSuite', 'findTestCases'])


from unittest2.collector import collector
from unittest2.result import TestResult
from unittest2.case import (
    TestCase, FunctionTestCase, SkipTest, skip, skipIf,
    skipUnless, expectedFailure
)
from unittest2.suite import BaseTestSuite, TestSuite
from unittest2.loader import (
    TestLoader, defaultTestLoader, makeSuite, getTestCaseNames,
    findTestCases
)
from unittest2.main import TestProgram, main, main_
from unittest2.runner import TextTestRunner, TextTestResult

try:
    from unittest2.signals import (
        installHandler, registerResult, removeResult, removeHandler
    )
except ImportError:
    # Compatibility with platforms that don't have the signal module
    pass
else:
    __all__.extend(['installHandler', 'registerResult', 'removeResult',
                    'removeHandler'])

# deprecated
_TextTestResult = TextTestResult

__unittest = True
