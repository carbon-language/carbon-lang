##===- utils/Burg/Makefile ---------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = burg
BUILT_SOURCES = gram.tab.c gram.tab.h

EXTRA_DIST = gram.yc gram.tab.c gram.tab.h sample.gr

include $(LEVEL)/Makefile.common

gram.tab.c gram.tab.h: gram.yc
	$(Verb) $(BISON) -o gram.tab.c -d $<

$(ObjDir)/lex.o : gram.tab.h

clean::
	$(Verb) $(RM) -rf gram.tab.h gram.tab.c core* *.aux *.log *.dvi sample sample.c tmp

doc.dvi: doc.tex
	$(Verb) latex doc; latex doc

check:: $(ToolBuildPath) $(BUILD_SRC_DIR)/sample.gr
	$(ToolBuildPath) -I <$(BUILD_SRC_DIR)/sample.gr   >sample.c \
	  && $(CC) $(CFLAGS) -o sample sample.c && ./sample
	$(ToolBuildPath) -I      $(BUILD_SRC_DIR)/sample.gr   >tmp \
	  && cmp tmp sample.c
	$(ToolBuildPath) -I     <$(BUILD_SRC_DIR)/sample.gr -o tmp \
	  && cmp tmp sample.c
	$(ToolBuildPath) -I      $(BUILD_SRC_DIR)/sample.gr -o tmp \
	  && cmp tmp sample.c
	$(ToolBuildPath) -I -O0 <$(BUILD_SRC_DIR)/sample.gr   >tmp \
	  && cmp tmp sample.c
	$(ToolBuildPath) -I -=  <$(BUILD_SRC_DIR)/sample.gr   >tmp \
	  && cmp tmp sample.c
	$(RM) -f tmp sample.c
