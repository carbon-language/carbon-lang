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

#
# Include the Master Makefile that knows how to build all.
#
include $(LEVEL)/Makefile.common

LOGIC_TESTS = eq ne le ge gt lt false true
BITWISE_TESTS = shl shr xor or and
ARITHMETIC_TESTS = abs neg add sub mul div mod star_slash incr decr min max 
STACK_TESTS = drop drop2 nip nip2 dup dup2 swap swap2 over over2 rot rot2 \
	      rrot rrot2 tuck tuck2 roll pick select
MEMORY_TESTS = memory
CONTROL_TESTS = while return 
IO_TESTS = space tab out_chr out_num out_str

TESTS = $(LOGIC_TESTS) $(ARITHMETIC_TESTS) $(BITWISE_TESTS) $(STACK_TESTS) \
	$(MEMORY_TESTS) $(CONTROL_TESTS) $(IO_TESTS)

LLC_EXEC = $(BUILD_OBJ_ROOT)/tools/$(CONFIGURATION)/llc
OPT_EXEC = $(BUILD_OBJ_ROOT)/tools/$(CONFIGURATION)/opt
STKRC_EXEC = $(BUILD_OBJ_ROOT)/tools/$(CONFIGURATION)/stkrc
LLVMDIS_EXEC = $(BUILD_OBJ_ROOT)/tools/$(CONFIGURATION)/llvm-dis

all :: test_each

test_each: $(TESTS)
	@$(ECHO) "Running Tests..."
	$(VERB) LD_LIBRARY_PATH=$(BUILD_OBJ_ROOT)/lib/$(CONFIGURATION) $(BUILD_SRC_DIR)/runtests $(BUILD_OBJ_DIR) $(TESTS) 

% : %.s testing.s
	@$(ECHO) "Compiling and Linking $< to $*"
	$(VERB)$(CC) -ggdb -L$(BUILD_OBJ_ROOT)/lib/$(CONFIGURATION) testing.s -lstkr_runtime -o $* $*.s

%.s : %.bc
	@$(ECHO) "Compiling $< to $*.s"
	$(VERB)$(LLC_EXEC) -f -o $*.s $<

ifdef OPTIMIZE
%.bc : %.st $(STKRC_EXEC)
	@$(ECHO) "Compiling and Optimizing $< to $*.bc"
	$(VERB)$(STKRC_EXEC) -e -o - $< | opt -stats -q -f -o $*.bc -adce -branch-combine -cee -constmerge -constprop -dce -die -gcse -globaldce -instcombine -pre
else
%.bc : %.st $(STKRC_EXEC)
	@$(ECHO) "Compiling $< to $*.bc"
	$(VERB)$(STKRC_EXEC) -e -f -o $*.bc $< 
endif

%.ll : %.bc
	@$(ECHO) "Disassembling $< to $*.ll"
	$(VERB)$(LLVMDIS_EXEC) -dis -o $*.ll $<

TESTS_LL = $(TESTS:%=%.ll)
TESTS_BC = $(TESTS:%=%.bc)
TESTS_S  = $(TESTS:%=%.s)

clean :: 
	$(VERB)rm -f gmon.out $(TESTS_LL) $(TESTS_BC) $(TESTS_S) $(TESTS) testing.bc testing.s testing.ll

.SUFFIXES: .st .s .ll .bc
.PRECIOUS: %.s %.ll %.bc %.st
.PHONY: test_each
