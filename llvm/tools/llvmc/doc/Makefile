##===- tools/llvmc/doc/Makefile ----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL=../../..
include $(LEVEL)/Makefile.config

RST_CSS=llvm-rst.css
DOC_DIR=../../../docs
RST2HTML=rst2html --stylesheet=$(RST_CSS) --link-stylesheet

all : LLVMC-Reference.html LLVMC-Tutorial.html $(RST_CSS)
	$(CP) $(RST_CSS) $(DOC_DIR)/$(RST_CSS)
	$(CP) LLVMC-Reference.html $(DOC_DIR)/CompilerDriver.html
	$(CP) LLVMC-Reference.html $(DOC_DIR)/CompilerDriverTutorial.html

LLVMC-Tutorial.html : LLVMC-Tutorial.rst $(RST_CSS)
	$(RST2HTML) $< $@

LLVMC-Reference.html : LLVMC-Reference.rst $(RST_CSS)
	$(RST2HTML) $< $@

clean :
	$(RM) LLVMC-Tutorial.html LLVMC-Reference.html
