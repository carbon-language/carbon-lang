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
LINK_COMPONENTS := Core support system
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common

# Make the object code file for the lexer depend upon the header file generated
# by the Bison parser.  This prevents the Lexer from being compiled before the
# header file it needs is built.
$(ObjDir)/upgradeLexer.o: $(PROJ_SRC_DIR)/UpgradeParser.h

TESTCASE=../../test/Regression/Assembler/2004-09-29-VerifierIsReallySlow.llx
test:
	llvm-as $(TESTCASE) -o - | llvm-dis -o source.ll -f
	../../Debug/bin/llvm-upgrade -o - $(TESTCASE) 2>err.out | llvm-as | \
	  llvm-dis > upgrade.ll -f
	diff source.ll upgrade.ll > diff.out

valgrind:
	valgrind ../../Debug/bin/llvm-upgrade -o /dev/null -f $(TESTCASE)

$(ObjDir)/UpgradeLexer.o: $(PROJ_SRC_DIR)/UpgradeParser.y $(PROJ_SRC_DIR)/UpgradeParser.h
