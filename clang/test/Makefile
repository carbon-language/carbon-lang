LEVEL = ../../..
include $(LEVEL)/Makefile.common

# Test in all immediate subdirectories if unset.
TESTDIRS ?= $(shell echo $(PROJ_SRC_DIR)/*/)

ifndef TESTARGS
ifdef VERBOSE
TESTARGS = -v
else
TESTARGS = -s
endif
endif

all::
	@ echo '--- Running clang tests for $(TARGET_TRIPLE) ---'
	@ PATH=$(ToolDir):$(LLVM_SRC_ROOT)/test/Scripts:$$PATH VG=$(VG) $(PROJ_SRC_DIR)/../utils/test/MultiTestRunner.py $(TESTARGS) $(TESTDIRS)

clean::
	@ rm -rf Output/

.PHONY: all report clean
