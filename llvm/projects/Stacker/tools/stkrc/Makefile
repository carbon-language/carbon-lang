##===- projects/Stacker/lib/stkrc/Makefile -----------------*- Makefile -*-===##

#
# Indicate where we are relative to the top of the source tree.
#
LEVEL=../..

#
# Give the name of a library.  This will build a dynamic version.
#
TOOLNAME = stkrc
LLVMLIBS = LLVMAsmParser LLVMBCWriter LLVMTransforms.a LLVMipo.a LLVMipa.a \
	   LLVMScalarOpts.a LLVMAnalysis.a LLVMTarget.a LLVMTransformUtils.a \
	   LLVMCore LLVMSupport.a LLVMbzip2 LLVMSystem.a 
CONFIG_FILES = st
EXTRA_DIST = st
USEDLIBS=stkr_compiler 


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

