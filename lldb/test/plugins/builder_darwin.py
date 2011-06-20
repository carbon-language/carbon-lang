import os
import lldbtest

from builder_base import *

#print "Hello, darwin plugin!"

def buildDsym(sender=None, architecture=None, compiler=None, dictionary=None):
    """Build the binaries with dsym debug info."""
    lldbtest.system(["/bin/sh", "-c",
                     "make clean" + getCmdLine(dictionary)
                     + "; make MAKE_DSYM=YES"
                     + getArchSpec(architecture) + getCCSpec(compiler)
                     + getCmdLine(dictionary)],
                    sender=sender)

    # True signifies that we can handle building dsym.
    return True
