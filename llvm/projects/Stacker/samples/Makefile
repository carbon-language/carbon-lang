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

SAMPLES = fibonacci hello prime

LLC_EXEC = $(BUILD_OBJ_ROOT)/tools/$(CONFIGURATION)/llc
OPT_EXEC = $(BUILD_OBJ_ROOT)/tools/$(CONFIGURATION)/opt
STKRC_EXEC = $(BUILD_OBJ_ROOT)/tools/$(CONFIGURATION)/stkrc
LLVMDIS_EXEC = $(BUILD_OBJ_ROOT)/tools/$(CONFIGURATION)/llvm-dis

all :: $(SAMPLES)

ifdef OPTIMIZE
%.bc : %.st 
	@$(ECHO) "Compiling and Optimizing $< to $*.bc"
	$(VERB)$(STKRC_EXEC) -e -o - $< | opt -stats -q -f -o $*.bc \
	    -aa-eval -adce -branch-combine -cee -constmerge -constprop -dce -die -ds-aa \
	    -ds-opt -gcse -globaldce -indvars -inline -instcombine \
	    -ipconstprop -licm -loopsimplify -mem2reg -pre -sccp -simplifycfg \
	    -tailcallelim -verify
else
%.bc : %.st
	@$(ECHO) "Compiling $< to $*.bc"
	$(VERB)$(STKRC_EXEC) -e -f -o $*.bc $< 
endif

%.s : %.bc
	@$(ECHO) "Compiling $< to $*.s"
	$(VERB)$(LLC_EXEC) -f -o $*.s $<

% : %.s
	@$(ECHO) "Compiling and Linking $< to $*"
	$(VERB)gcc -g -L$(BUILD_OBJ_ROOT)/lib/$(CONFIGURATION) -lstkr_runtime -o $* $*.s

%.ll : %.bc
	@$(ECHO) "Disassembling $< to $*.ll"
	$(VERB)$(LLVMDIS_EXEC) -f -o $*.ll $<

%.bc :  $(STKRC_EXEC)

.PRECIOUS: %.bc %.s %.ll %.st

SAMPLES_LL = $(SAMPLES:%=%.ll)
SAMPLES_BC = $(SAMPLES:%=%.bc)
SAMPLES_S  = $(SAMPLES:%=%.s)

clean ::
	$(VERB)rm -f gmon.out $(SAMPLES_LL) $(SAMPLES_BC) $(SAMPLES_S) $(SAMPLES)
#
# Include the Master Makefile that knows how to build all.
#
include $(LEVEL)/Makefile.common
