LEVEL = ../../..
TOOLNAME = tblgen
USEDLIBS = support.a

.PRECIOUS: FileLexer.cpp FileParser.cpp

include $(LEVEL)/Makefile.common

#
# Make the source file depend on the header file.  In this way, dependencies
# (which depend on the source file) won't get generated until bison is done
# generating the C source and header files for the parser.
#
FileLexer.cpp: FileParser.h

clean::
	-rm -f FileParser.cpp FileParser.h FileLexer.cpp CommandLine.cpp
	-rm -f FileParser.output
