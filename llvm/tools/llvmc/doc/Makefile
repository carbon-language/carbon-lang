##===- tools/llvmc/doc/Makefile ----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

RST2HTML=rst2html --stylesheet=llvm.css --link-stylesheet

all : LLVMC-Reference.html LLVMC-Tutorial.html

LLVMC-Tutorial.html : LLVMC-Tutorial.rst llvm.css
	$(RST2HTML) $< $@

LLVMC-Reference.html : LLVMC-Reference.rst llvm.css
	$(RST2HTML) $< $@

clean :
	rm *.html
