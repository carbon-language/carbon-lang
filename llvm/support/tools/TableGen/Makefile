LEVEL = ../..
TOOLNAME = tblgen
USEDLIBS = support.a

include $(LEVEL)/Makefile.common

clean::
	-rm -f FileParser.cpp FileParser.h FileLexer.cpp CommandLine.cpp
	-rm -f FileParser.output


dump::
	$(TOOLEXENAME_G) < X86.td

parse:
	$(TOOLEXENAME_G) < X86.td -parse

test::

	@echo "enum {"
	@$(TOOLEXENAME_G) < X86.td -class=Register
	@echo 
	@echo "};"

	@echo "enum {"
	@$(TOOLEXENAME_G) < X86.td -class=Instruction
	@echo 
	@echo "};"
