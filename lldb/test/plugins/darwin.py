import lldbtest

#print "Hello, darwin plugin!"

def buildDefault():
    lldbtest.system(["/bin/sh", "-c", "make clean; make"])

    # True signifies that we can handle building default.
    return True

def buildDsym():
    lldbtest.system(["/bin/sh", "-c", "make clean; make MAKE_DSYM=YES"])

    # True signifies that we can handle building dsym.
    return True

def buildDwarf():
    lldbtest.system(["/bin/sh", "-c", "make clean; make MAKE_DSYM=NO"])

    # True signifies that we can handle building dsym.
    return True
