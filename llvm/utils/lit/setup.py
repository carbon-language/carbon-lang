import lit

# FIXME: Support distutils?
from setuptools import setup, find_packages
setup(
    name = "Lit",
    version = lit.__version__,

    author = lit.__author__,
    author_email = lit.__email__,
    url = 'http://llvm.org',
    license = 'BSD',

    description = "A Software Testing Tool",
    keywords = 'test C++ automatic discovery',
    long_description = """\
Lit
+++

About
=====

Lit is a portable tool for executing LLVM and Clang style test suites,
summarizing their results, and providing indication of failures. Lit is designed
to be a lightweight testing tool with as simple a user interface as possible.


Features
========

 * Portable!
 * Flexible test discovery.
 * Parallel test execution.
 * Support for multiple test formats and test suite designs.


Documentation
=============

The offical Lit documentation is in the man page, available online in the `LLVM
Command Guide http://llvm.org/cmds/lit.html`_.


Source
======

The Lit source is available as part of LLVM, in the `LLVM SVN repository
<http://llvm.org/svn/llvm-project/llvm/trunk/utils/lit`_.
""",

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: University of Illinois/NCSA Open Source License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Progamming Language :: Python',
        'Topic :: Software Development :: Testing',
        ],

    zip_safe = False,
    packages = find_packages(),
    entry_points = {
        'console_scripts': [
            'lit = lit:main',
            ],
        }
)
