"""
If the build* function is passed the compiler argument, for example, 'llvm-gcc',
it is passed as a make variable to the make command.  Otherwise, we check the
LLDB_CC environment variable; if it is defined, it is passed as a make variable
to the make command.

If neither the compiler keyword argument nor the LLDB_CC environment variable is
specified, no CC make variable is passed to the make command.  The Makefile gets
to define the default CC being used.
"""

import os
import lldbtest

#print "Hello, darwin plugin!"

def getCCSpec(compiler):
    """
    Helper function to return the key-value pair string to specify the compiler
    used for the make system.
    """
    cc = compiler if compiler else None
    if not cc and "LLDB_CC" in os.environ:
        cc = os.environ["LLDB_CC"]

    # Note the leading space character.
    return (" CC=" + cc) if cc else ""


def buildDefault(compiler=None):
    """Build the binaries the default way."""
    lldbtest.system(["/bin/sh", "-c",
                     "make clean; make" + getCCSpec(compiler)])

    # True signifies that we can handle building default.
    return True

def buildDsym(compiler=None):
    """Build the binaries with dsym debug info."""
    lldbtest.system(["/bin/sh", "-c",
                     "make clean; make MAKE_DSYM=YES" + getCCSpec(compiler)])

    # True signifies that we can handle building dsym.
    return True

def buildDwarf(compiler=None):
    """Build the binaries with dwarf debug info."""
    lldbtest.system(["/bin/sh", "-c",
                     "make clean; make MAKE_DSYM=NO" + getCCSpec(compiler)])

    # True signifies that we can handle building dsym.
    return True
