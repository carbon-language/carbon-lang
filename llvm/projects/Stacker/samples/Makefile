##===- projects/sample/Makefile ----------------------------*- Makefile -*-===##
#
# This is a sample Makefile for a project that uses LLVM.
#
##===----------------------------------------------------------------------===##

#
# Indicates our relative path to the top of the project's root directory.
#
LEVEL = ../../..

#
# Directories that needs to be built.
#
DIRS = 

TESTS = fibonacci hello prime

all :: $(TESTS)

ifdef OPTIMIZE
%.bc : %.st 
	stkrc -e -o - $< | opt -stats -q -f -o $*.bc \
	    -aa-eval -adce -branch-combine -cee -constmerge -constprop -dce -die -ds-aa \
	    -ds-opt -gcse -globaldce -indvars -inline -instcombine \
	    -ipconstprop -licm -loopsimplify -mem2reg -pre -sccp -simplifycfg \
	    -tailcallelim -verify
else
%.bc : %.st
	stkrc -e -f -o $*.bc $< 
endif

%.s : %.bc
	llc -f -o $*.s $<

% : %.s
	gcc -g -L$(BUILD_OBJ_ROOT)/lib/Debug -lstkr_runtime -o $* $*.s

%.ll : %.bc
	llvm-dis -f -o $*.ll $<

%.bc :  $(BUILD_OBJ_ROOT)/tools/Debug/stkrc

.PRECIOUS: %.bc %.s %.ll %.st
#
# Include the Master Makefile that knows how to build all.
#
include $(LEVEL)/Makefile.common
