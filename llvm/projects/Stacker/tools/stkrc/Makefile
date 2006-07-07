##===- projects/Stacker/lib/stkrc/Makefile -----------------*- Makefile -*-===##

#
# Indicate where we are relative to the top of the source tree.
#
LEVEL=../..

#
# Give the name of a library.  This will build a dynamic version.
#
TOOLNAME = stkrc
LLVMLIBS = LLVMAsmParser.a LLVMBCWriter.a LLVMipo.a LLVMScalarOpts.a \
	  LLVMTransforms.a LLVMTransformUtils.a LLVMipa.a LLVMAnalysis.a \
	  LLVMTarget.a LLVMCore.a LLVMSupport.a LLVMbzip2.a LLVMSystem.a 
CONFIG_FILES = st
EXTRA_DIST = st
USEDLIBS=stkr_compiler 
REQUIRES_EH := 1


ifdef PARSE_DEBUG
CPPFLAGS = -DPARSE_DEBUG=1
endif

ifdef FLEX_DEBUG
CPPFLAGS += -DFLEX_DEBUG=1
endif

#
# Include Makefile.common so we know what to do.
#
include $(LEVEL)/Makefile.common

