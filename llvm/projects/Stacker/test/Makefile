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

LLVMC = $(LLVMToolDir)/llvmc

all :: test_each

test_each: $(TESTS)
	$(Echo) "Running Tests..."
	$(Verb) LD_LIBRARY_PATH=$(PROJ_OBJ_ROOT)/lib/$(CONFIGURATION) \
	  $(PROJ_SRC_DIR)/runtests $(PROJ_OBJ_DIR) $(TESTS) 

% : %.st Makefile testing.bc
	$(Echo) "Building $*" 
	$(Verb)$(LLVMC) -O4 -o $@ $< testing.bc $(LibDir)/stkr_runtime.bc -lcrtend

testing.bc : testing.st Makefile
	$(Echo) "Compiling $*"
	$(Verb)$(LLVMC) -O3 -c -o $@ $<

TESTS_LL = $(TESTS:%=%.ll)
TESTS_BC = $(TESTS:%=%.bc)
TESTS_S  = $(TESTS:%=%.s)

clean :: 
	$(Verb)rm -f gmon.out $(TESTS_LL) $(TESTS_BC) $(TESTS_S) $(TESTS) \
	  testing.bc testing.s testing.ll

.SUFFIXES: .st .s .ll .bc
.PRECIOUS: %.s %.ll %.bc %.st
.PHONY: test_each
