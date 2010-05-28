##===- clang/lib/Headers/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../../../..

BUILT_SOURCES = arm_neon.h.inc
TABLEGEN_INC_FILES_COMMON = 1

include $(LEVEL)/Makefile.common

CLANG_VERSION := $(shell cat $(PROJ_SRC_DIR)/../../VER)

HeaderDir := $(PROJ_OBJ_ROOT)/$(BuildMode)/lib/clang/$(CLANG_VERSION)/include

HEADERS := $(notdir $(wildcard $(PROJ_SRC_DIR)/*.h))

OBJHEADERS := $(addprefix $(HeaderDir)/, $(HEADERS))


$(OBJHEADERS): $(HeaderDir)/%.h: $(PROJ_SRC_DIR)/%.h $(HeaderDir)/.dir $(HeaderDir)/arm_neon.h
	$(Verb) cp $< $@
	$(Echo) Copying $(notdir $<) to build dir

$(HeaderDir)/arm_neon.h: $(BUILT_SOURCES)
	$(Verb) cp $< $@
	$(Echo) Copying $(notdir $<) to build dir
	
# Hook into the standard Makefile rules.
all-local:: $(OBJHEADERS)

PROJ_headers := $(DESTDIR)$(PROJ_prefix)/lib/clang/$(CLANG_VERSION)/include

INSTHEADERS := $(addprefix $(PROJ_headers)/, $(HEADERS))

$(PROJ_headers):
	$(Verb) $(MKDIR) $@

$(INSTHEADERS): $(PROJ_headers)/%.h: $(HeaderDir)/%.h | $(PROJ_headers)
	$(Verb) $(DataInstall) $< $(PROJ_headers)
	$(Echo) Installing compiler include file: $(notdir $<)

install-local:: $(INSTHEADERS)

$(ObjDir)/arm_neon.h.inc.tmp : arm_neon.td $(TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang arm_neon.h.inc with tblgen"
	$(Verb) $(TableGen) -gen-arm-neon-header -o $(call SYSPATH, $@) $<
