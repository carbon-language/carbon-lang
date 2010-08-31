import lldbtest

#print "Hello, darwin plugin!"

def buildDsym():
    lldbtest.system(["/bin/sh", "-c", "make clean; make MAKE_DSYM=YES"])

    # True signifies that we can handle building dsym.
    return True

def buildDwarf():
    lldbtest.system(["/bin/sh", "-c", "make clean; make MAKE_DSYM=NO"])

    # True signifies that we can handle building dsym.
    return True
