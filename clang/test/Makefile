LEVEL = ../../..
include $(LEVEL)/Makefile.common

# Test in all immediate subdirectories if unset (except for the .svn and Output
# directories).
TESTDIRS ?= $(shell find $(PROJ_SRC_DIR) -depth 1 -type d -not -name .svn -prune -and -not -name Output -prune)

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
	@ PATH=$(ToolDir):$(LLVM_SRC_ROOT)/test/Scripts:$$PATH $(PROJ_SRC_DIR)/../utils/test/MultiTestRunner.py $(TESTARGS) $(TESTDIRS) $(VGARG)

clean::
	@ rm -rf Output/

.PHONY: all report clean
