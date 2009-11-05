LEVEL = ../../..
include $(LEVEL)/Makefile.common

# Test in all immediate subdirectories if unset.
ifdef TESTSUITE
TESTDIRS := $(TESTSUITE:%=$(PROJ_SRC_DIR)/%)
else
TESTDIRS ?= $(PROJ_SRC_DIR)
endif

# 'lit' wants objdir paths, so it will pick up the lit.site.cfg.
TESTDIRS := $(TESTDIRS:$(PROJ_SRC_DIR)%=$(PROJ_OBJ_DIR)%)

# Allow EXTRA_TESTDIRS to provide additional test directories.
TESTDIRS += $(EXTRA_TESTDIRS)

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

all:: lit.site.cfg
	@ echo '--- Running clang tests for $(TARGET_TRIPLE) ---'
	@ $(PYTHON) $(LLVM_SRC_ROOT)/utils/lit/lit.py \
	  $(TESTARGS) $(TESTDIRS) $(VGARG)

FORCE:

lit.site.cfg: FORCE
	@echo "Making Clang 'lit.site.cfg' file..."
	@sed -e "s#@LLVM_SOURCE_DIR@#$(LLVM_SRC_ROOT)#g" \
	     -e "s#@LLVM_BINARY_DIR@#$(LLVM_OBJ_ROOT)#g" \
	     -e "s#@LLVM_TOOLS_DIR@#$(ToolDir)#g" \
	     -e "s#@LLVM_LIBS_DIR@#$(LibDir)#g" \
	     -e "s#@CLANG_SOURCE_DIR@#$(PROJ_SRC_DIR)/..#g" \
	     -e "s#@CLANG_BINARY_DIR@#$(PROJ_OBJ_DIR)/..#g" \
	     -e "s#@TARGET_TRIPLE@#$(TARGET_TRIPLE)#g" \
	     $(PROJ_SRC_DIR)/lit.site.cfg.in > $@

clean::
	@ find . -name Output | xargs rm -fr

.PHONY: all report clean
