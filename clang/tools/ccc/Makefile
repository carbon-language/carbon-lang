##===- tools/ccc/Makefile ----------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../../..

include $(LEVEL)/Makefile.common

install-local:: $(PROJ_bindir)/ccc $(PROJ_bindir)/ccclib

Extra := $(wildcard $(PROJ_SRC_ROOT)/tools/clang/tools/ccc/ccclib/*.py)

$(PROJ_bindir)/ccclib : $(Extra)
	$(Echo) Installing ccclib.
	$(Verb) mkdir -p "$(PROJ_bindir)/ccclib"
	$(Verb) cp -p $? "$(PROJ_bindir)/ccclib"
	$(Verb) python -m compileall -d "$(PROJ_prefix)/bin/ccclib" "$(PROJ_bindir)/ccclib"
	$(Verb) touch "$(PROJ_bindir)/ccclib"

$(PROJ_bindir)/ccc : ccc
	$(Echo) Installing $< shell script.
	$(Verb) cat $< > "$@"
	$(Verb) chmod 0755 "$@"
