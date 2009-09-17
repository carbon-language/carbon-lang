LEVEL = ../../..
include $(LEVEL)/Makefile.common

# Test in all immediate subdirectories if unset.
ifdef TESTSUITE
TESTDIRS := $(TESTSUITE:%=$(PROJ_SRC_DIR)/%)
else
TESTDIRS ?= $(PROJ_SRC_DIR)
endif

# LIT2 wants objdir paths, so it will pick up the lit.site.cfg.
LIT2_TESTDIRS := $(TESTDIRS:$(PROJ_SRC_DIR)%=$(PROJ_OBJ_DIR)%)

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

ifdef LIT2
all:: lit.site.cfg
	@ echo '--- Running clang tests for $(TARGET_TRIPLE) ---'
	@ $(LLVM_SRC_ROOT)/utils/lit/lit.py \
	  $(TESTARGS) $(LIT2_TESTDIRS) $(VGARG)
else
all::
	@ echo '--- Running clang tests for $(TARGET_TRIPLE) ---'
	@ $(PROJ_SRC_DIR)/../utils/test/MultiTestRunner.py \
	  --root $(PROJ_SRC_DIR) \
	  --path $(ToolDir) \
	  --path $(LLVM_SRC_ROOT)/test/Scripts \
	  $(TESTARGS) $(TESTDIRS) $(VGARG)
endif

FORCE:

lit.site.cfg: FORCE
	@echo "Making 'lit.site.cfg' file..."
	@sed -e "s#@CLANG_SOURCE_DIR@#$(PROJ_SRC_DIR)/..#g" \
	     -e "s#@CLANG_BINARY_DIR@#$(PROJ_OBJ_DIR)/..#g" \
	     -e "s#@LLVM_TOOLS_DIR@#$(ToolDir)#g" \
	     $(PROJ_SRC_DIR)/lit.site.cfg.in > $@

clean::
	@ rm -rf Output/

.PHONY: all report clean
