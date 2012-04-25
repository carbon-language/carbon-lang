#----------------------------------------------------------------------
# Fill in the source files to build
#----------------------------------------------------------------------
# Uncomment line below for debugging shell commands
# SHELL = /bin/sh -x

#----------------------------------------------------------------------
# Change any build/tool options needed
#----------------------------------------------------------------------
DS := /usr/bin/dsymutil
CFLAGS ?=-arch x86_64 -arch i386 -gdwarf-2 -O0
CPLUSPLUSFLAGS +=$(CFLAGS)
CPPFLAGS +=$(CFLAGS)
LDFLAGS = $(CFLAGS) -install_name "@executable_path/libheap.dylib" -dynamiclib
CXX := $(shell xcrun -find clang++)
LD := $(CXX)
TEMPS =
EXE=libheap.dylib
DSYM=$(EXE).dSYM

#----------------------------------------------------------------------
# Make the dSYM file from the executable
#----------------------------------------------------------------------
$(DSYM) : $(EXE)
	$(DS) -o "$(DSYM)" "$(EXE)"

#----------------------------------------------------------------------
# Compile the executable from all the objects (default rule) with no
# dsym file.
#----------------------------------------------------------------------
$(EXE) : heap_find.o
	$(LD) $(LDFLAGS) heap_find.o -o "$(EXE)"

heap_find.o : heap_find.cpp
	$(CXX) $(CFLAGS) -c heap_find.cpp

#----------------------------------------------------------------------
# Include all of the makefiles for each source file so we don't have
# to manually track all of the prerequisites for each source file.
#----------------------------------------------------------------------
.PHONY: clean
dsym:	$(DSYM)
all:	$(EXE) $(DSYM)
clean:
	rm -rf "$(EXE)" "$(DSYM)" heap_find.o $(TEMPS)



