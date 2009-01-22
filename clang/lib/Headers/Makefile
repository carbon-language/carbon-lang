##===- clang/lib/Headers/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../../../..
include $(LEVEL)/Makefile.common

HeaderDir := $(PROJ_OBJ_ROOT)/$(BuildMode)/Headers

HEADERS := $(notdir $(wildcard $(PROJ_SRC_DIR)/*.h))

OBJHEADERS := $(addprefix $(HeaderDir)/, $(HEADERS))


$(OBJHEADERS): $(HeaderDir)/%.h: $(PROJ_SRC_DIR)/%.h $(HeaderDir)/.dir
	$(Verb) cp $< $@
	@echo Copying $(notdir $<) to build dir
        
printit:
	echo $(OBJHEADERS)
	echo  $(PROJ_SRC_DIR)
 
# Hook into the standard Makefile rules.
all-local:: $(OBJHEADERS)

PROJ_headers := $(DESTDIR)$(PROJ_prefix)/Headers

INSTHEADERS := $(addprefix $(PROJ_headers)/, $(HEADERS))

$(PROJ_headers):
	$(Verb) $(MKDIR) $@

$(INSTHEADERS): $(PROJ_headers)/%.h: $(HeaderDir)/%.h | $(PROJ_headers)
	$(Verb) $(DataInstall) $< $(PROJ_headers)

install-local:: $(INSTHEADERS)

