##===- docs/Makefile ---------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
doxygen-files:
	doxygen doxygen.cfg

doxygen.tar.gz: doxygen/index.html
	rm -rf $@
	gtar czvf $@ doxygen
