##===- tools/extra/Makefile --------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..

include $(CLANG_LEVEL)/../../Makefile.config

PARALLEL_DIRS := remove-cstr-calls tool-template clang-format cpp11-migrate
DIRS := test

include $(CLANG_LEVEL)/Makefile

# Custom target. Pass request to test/Makefile that knows what to do. To access
# this target you'd issue:
#
# make -C <build_dir>/tools/clang/tools/extra test
test::
	@ $(MAKE) -C test test

.PHONY: test
