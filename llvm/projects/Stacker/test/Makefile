##===- projects/Stacker/test/Makefile ----------------------*- Makefile -*-===##
#
# This is the makefile that tests the various facilities of the Stacker language
#
##===----------------------------------------------------------------------===##

#
# Indicates our relative path to the top of the project's root directory.
#
LEVEL = ../

#
# Directories that need to be built.
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

STKRC_EXEC = $(BUILD_OBJ_ROOT)/tools/$(CONFIGURATION)/stkrc

all :: test_each

test_each: $(TESTS)
	$(Echo) "Running Tests..."
	$(Verb) LD_LIBRARY_PATH=$(BUILD_OBJ_ROOT)/lib/$(CONFIGURATION) $(BUILD_SRC_DIR)/runtests $(BUILD_OBJ_DIR) $(TESTS) 

%.bc : %.st Makefile
	$(Echo) "Compiling And Optimizing $< to $*.bc"
	$(Verb)$(STKRC_EXEC) -f -O4 -s 2048 -o $*.bc $(BUILD_SRC_DIR)/$*.st

%.s : %.bc testing.bc
	$(Echo) "Assembling $< to $*.s"
	$(Verb)$(LLINK) $*.bc testing.bc -o $*.bcl
	$(Verb)$(LLC) -f -o $*.s $*.bcl

% : %.s
	$(Echo) "Linking $*"
	$(Verb)$(CC) -ggdb -L$(BUILD_OBJ_ROOT)/lib/$(CONFIGURATION) -lstkr_runtime -o $* $*.s

%.ll : %.bc
	$(Echo) "Disassembling $< to $*.ll"
	$(Verb)$(LDIS) -dis -o $*.ll $<

TESTS_LL = $(TESTS:%=%.ll)
TESTS_BC = $(TESTS:%=%.bc)
TESTS_S  = $(TESTS:%=%.s)

clean :: 
	$(Verb)rm -f gmon.out $(TESTS_LL) $(TESTS_BC) $(TESTS_S) $(TESTS) testing.bc testing.s testing.ll

.SUFFIXES: .st .s .ll .bc
.PRECIOUS: %.s %.ll %.bc %.st
.PHONY: test_each
