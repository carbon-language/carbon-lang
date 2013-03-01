##===- tools/extra/test/cpp11-migrate/Makefile -------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

# This Makefile is responsible for creating a LIT testsuite for auto-generated
# tests. The recipe that runs llvm-lit needs to provide the following path of
# the lit.site.cfg for the generated testsuite: $(PROJ_OBJ_DIR)/generated_tests.

CLANG_LEVEL := ../../../..
include $(CLANG_LEVEL)/Makefile

# List of generator scripts. Generator scripts must:
# * Be written in python
# * Output their result to standard out.
# * Be named as gen_X.py where X will be the name of the auto-generated file.
GENERATOR_SCRIPTS := \
  UseAuto/gen_basic_std_iterator_tests.cpp.py \
  UseAuto/Inputs/gen_my_std.h.py

# macro to be used with $(call) that generates a rule and recipe that causes a
# file to be auto-generated from a generator script. Generator scripts must
# follow scheme above. The resulting file is placed in:
# $(PROJ_OBJ_DIR)/generated_tests/$(dir <script name>).
#
# One argument required: the name of generator script relative to
# $(PROJ_SRC_DIR).
define autogenerate-file
OUTFILE := $(addprefix generated_tests/$(dir $1), $(patsubst gen_%.py,%,$(notdir $1)))
GENERATED_FILES := $$(GENERATED_FILES) $$(OUTFILE)
$$(OUTFILE): $1
	@echo "Autogenerating $$@"
	@$(MKDIR) $(addprefix generated_tests/, $(dir $1))
	@$(PYTHON) $$< > $$@
endef
$(foreach script, $(GENERATOR_SCRIPTS), $(eval $(call autogenerate-file,$(script))))

FORCE:

GENERATED_TESTS_ROOT := $(PROJ_OBJ_DIR)/generated_tests

# Recipe to create lit.site.cfg for the auto-generated tests suite. Sources
# for these tests are auto-generated into the build directory.
GEN_LIT_TMP := generated_tests/lit.tmp
generated_tests/lit.site.cfg: FORCE
	@$(MKDIR) $(GENERATED_TESTS_ROOT)
	@echo "Making lit.site.cfg for cpp11-migrate Auto-Generated Tests..."
	@$(ECHOPATH) s=@LLVM_SOURCE_DIR@=$(LLVM_SRC_ROOT)=g > $(GEN_LIT_TMP)
	@$(ECHOPATH) s=@LLVM_BINARY_DIR@=$(LLVM_OBJ_ROOT)=g >> $(GEN_LIT_TMP)
	@$(ECHOPATH) s=@LLVM_TOOLS_DIR@=$(ToolDir)=g >> $(GEN_LIT_TMP)
	@$(ECHOPATH) s=@LLVM_LIBS_DIR@=$(LibDir)=g >> $(GEN_LIT_TMP)
	@$(ECHOPATH) s=@CLANG_TOOLS_SOURCE_DIR@=$(PROJ_SRC_DIR)/../..=g >> $(GEN_LIT_TMP)
	@$(ECHOPATH) s=@CLANG_TOOLS_BINARY_DIR@=$(PROJ_OBJ_DIR)/../..=g >> $(GEN_LIT_TMP)
	@$(ECHOPATH) s=@TARGET_TRIPLE@=$(TARGET_TRIPLE)=g >> $(GEN_LIT_TMP)
	@$(ECHOPATH) s=@TEST_SOURCE_ROOT@=$(GENERATED_TESTS_ROOT)=g >> $(GEN_LIT_TMP)
	@$(ECHOPATH) s=@TEST_EXEC_ROOT@=$(PROJ_OBJ_DIR)=g >> $(GEN_LIT_TMP)
	@$(ECHOPATH) s=@TESTSUITE_NAME@=cpp11-migrate Auto-Generated Tests=g >> $(GEN_LIT_TMP)
	@$(ECHOPATH) s=@GENERATED_TESTS_ROOT@=$(GENERATED_TESTS_ROOT)=g >> $(GEN_LIT_TMP)
	@sed -f $(GEN_LIT_TMP) $(PROJ_SRC_DIR)/lit.site.cfg.in > $@
	@-rm -f $(GEN_LIT_TMP)

# Recipe to create the lit.site.cfg for the regular cpp11-migrate test suite.
lit.site.cfg: FORCE
	@echo "Making lit.site.cfg for cpp11-migrate Tests..."
	@$(ECHOPATH) s=@LLVM_SOURCE_DIR@=$(LLVM_SRC_ROOT)=g > lit.tmp
	@$(ECHOPATH) s=@LLVM_BINARY_DIR@=$(LLVM_OBJ_ROOT)=g >> lit.tmp
	@$(ECHOPATH) s=@LLVM_TOOLS_DIR@=$(ToolDir)=g >> lit.tmp
	@$(ECHOPATH) s=@LLVM_LIBS_DIR@=$(LibDir)=g >> lit.tmp
	@$(ECHOPATH) s=@CLANG_TOOLS_SOURCE_DIR@=$(PROJ_SRC_DIR)/../..=g >> lit.tmp
	@$(ECHOPATH) s=@CLANG_TOOLS_BINARY_DIR@=$(PROJ_OBJ_DIR)/../..=g >> lit.tmp
	@$(ECHOPATH) s=@TARGET_TRIPLE@=$(TARGET_TRIPLE)=g >> lit.tmp
	@$(ECHOPATH) s=@TEST_SOURCE_ROOT@=$(PROJ_SRC_DIR)=g >> lit.tmp
	@$(ECHOPATH) s=@TEST_EXEC_ROOT@=$(PROJ_OBJ_DIR)=g >> lit.tmp
	@$(ECHOPATH) s=@TESTSUITE_NAME@=cpp11-migrate Tests=g >> lit.tmp
	@$(ECHOPATH) s=@GENERATED_TESTS_ROOT@=$(GENERATED_TESTS_ROOT)=g >> lit.tmp
	@sed -f lit.tmp $(PROJ_SRC_DIR)/lit.site.cfg.in > $@
	@-rm -f lit.tmp

all:: $(GENERATED_FILES) lit.site.cfg generated_tests/lit.site.cfg

clean::
	@-rm -r generated_tests

.PHONY: all clean
