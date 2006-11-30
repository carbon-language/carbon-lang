##===- tools/llvm-upgrade/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
TOOLNAME = llvm-upgrade
LINK_COMPONENTS := support system

include $(LEVEL)/Makefile.common

# Make the object code file for the lexer depend upon the header file generated
# by the Bison parser.  This prevents the Lexer from being compiled before the
# header file it needs is built.
$(ObjDir)/upgradeLexer.o: $(PROJ_SRC_DIR)/UpgradeParser.h

test:
	../../Debug/bin/llvm-upgrade -o - ../../test/Feature/basictest.ll
