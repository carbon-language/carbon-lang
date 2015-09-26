"""
                     The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.

Sync lldb and related source from a local machine to a remote machine.

This facilitates working on the lldb sourcecode on multiple machines
and multiple OS types, verifying changes across all.

Provides helper support for adding lldb test paths to the python path.
"""
import os
import sys


def add_lldb_test_paths(check_dir):
    """Adds lldb test-related paths to the python path.

    Starting with the given directory and working upward through
    each parent directory up to the root, it looks for the lldb
    test directory.  When found, the lldb test directory and its
    child test_runner/lib directory will be added to the python
    system path.

    Instructions for use:

    This method supports a simple way of getting pylint to be able
    to reliably lint lldb python test scripts (including the test
    infrastructure itself).  To do so, add the following to a
    .pylintrc file in your home directory:

    [Master]
    init-hook='import os; import sys; sys.path.append(os.path.expanduser("~/path/to/lldb/test")); import lldb_pylint_helper; lldb_pylint_helper.add_lldb_test_paths(os.getcwd()); print("sys.path={}\n".format(sys.path))'

    Replace ~/path/to/lldb/test with a valid path to your local lldb source
    tree.  Note you can have multiple lldb source trees on your system, and
    this will work just fine.  The path in your .pylintrc is just needed to
    find the paths needed for pylint in whatever lldb source tree you're in.
    pylint will use the python files in whichever tree it is run from.

    Note it is critical that the init-hook line be contained on a single line.
    You can remove the print line at the end once you know the pythonpath is
    getting set up the way you expect.

    With these changes, you will be able to run the following, for example.

    cd lldb/sourcetree/1-of-many/test/lang/c/anonymous
    pylint TestAnonymous.py

    This will work, and include all the lldb/sourcetree/1-of-many lldb-specific
    python directories to your path.

    You can then run it in another lldb source tree on the same machine like
    so:

    cd lldb/sourcetree/2-of-many/test/functionalities/inferior-assert
    pyline TestInferiorAssert.py

    and this will properly lint that file, using the lldb-specific python
    directories from the 2-of-many source tree.

    Note at the time I'm writing this, our tests are in pretty sad shape
    as far as a stock pylint setup goes.  But we need to start somewhere :-)

    @param check_dir specifies a directory that will be used to start
    looking for the lldb test infrastructure python library paths.
    """
    check_dir = os.path.realpath(check_dir)
    while check_dir and len(check_dir) > 0:
        # If the current directory is test, it might be the lldb/test
        # directory. If so, we've found an anchor that will allow us
        # to add the relevant lldb-sourcetree-relative python lib
        # dirs.
        if os.path.basename(check_dir) == 'test':
            # If this directory has a dotest.py file in it,
            # then this is an lldb test tree.  Add the
            # test directories to the python path.
            if os.path.exists(os.path.join(check_dir, "dotest.py")):
                sys.path.insert(0, check_dir)
                sys.path.insert(0, os.path.join(
                    check_dir, "test_runner", "lib"))
                break
        # Continue looking up the parent chain until we have no more
        # directories to check.
        new_check_dir = os.path.dirname(check_dir)
        # We're done when the new check dir is not different
        # than the current one.
        if new_check_dir == check_dir:
            break
        check_dir = new_check_dir
