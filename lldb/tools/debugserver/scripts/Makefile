##===- tools/debugserver/scripts/Makefile ------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../../..

include $(LLDB_LEVEL)/Makefile

DBCONFIG_FILE = "$(PROJ_OBJ_DIR)/../DNBConfig.h"

$(DBCONFIG_FILE):
	SCRIPT_OUTPUT_FILE_0=$(DBCONFIG_FILE) perl $(PROJ_SRC_DIR)/dbgnub-config.pl

all:: $(DBCONFIG_FILE)

.PHONY: all