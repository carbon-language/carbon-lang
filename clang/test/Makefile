LEVEL = ../../..
include $(LEVEL)/Makefile.common

# Test in all immediate subdirectories if unset.
ifdef TESTSUITE
TESTDIRS := $(TESTSUITE:%=$(PROJ_SRC_DIR)/%)
else
TESTDIRS ?= $(PROJ_SRC_DIR)
endif

ifndef TESTARGS
ifdef VERBOSE
TESTARGS = -v
else
TESTARGS = -s
endif
endif

ifdef VG
  VGARG="--vg"
else
  VGARG=
endif

all::
	@ echo '--- Running clang tests for $(TARGET_TRIPLE) ---'
	@ $(PROJ_SRC_DIR)/../utils/test/MultiTestRunner.py \
	  --root $(PROJ_SRC_DIR) \
	  --path $(ToolDir) \
	  --path $(LLVM_SRC_ROOT)/test/Scripts \
	  $(TESTARGS) $(TESTDIRS) $(VGARG)

clean::
	@ rm -rf Output/

.PHONY: all report clean
