##===- utils/Makefile --------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ..
DIRS = Burg TableGen fpcmp

EXTRA_DIST := cgiplotNLT.pl check-each-file codegen-diff countloc.sh cvsupdate \
              DSAclean.py DSAextract.py emacs findsym.pl GenLibDeps.pl \
	      getsrcs.sh importNLT.pl llvmdo llvmgrep llvm-native-gcc \
	      llvm-native-gxx makellvm NightlyTest.gnuplot NightlyTest.pl \
	      NightlyTestTemplate.html NLT.schema parseNLT.pl plotNLT.pl \
	      profile.pl RegressionFinder.pl userloc.pl webNLT.pl \
	      vim llvm-config

include $(LEVEL)/Makefile.common

# Only include llvm-config if we have Perl to build it with.
ifeq ($(HAVE_PERL),1)
  DIRS += llvm-config
endif
