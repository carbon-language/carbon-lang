##===- source/Host/Makefile --------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../..
LEVEL := $(LLDB_LEVEL)/../..

include $(LEVEL)/Makefile.config

define DIR_SOURCES
SOURCES += $$(addprefix $(1)/,$$(notdir $$(wildcard $$(PROJ_SRC_DIR)/$(1)/*.cpp \
					 $$(PROJ_SRC_DIR)/*.cc $$(PROJ_SRC_DIR)/$(1)/*.c $$(PROJ_SRC_DIR)/$(1)/*.mm)))
endef

$(eval $(call DIR_SOURCES,common))

ifeq ($(HOST_OS),Darwin)
$(eval $(call DIR_SOURCES,posix))
$(eval $(call DIR_SOURCES,macosx))
CFCPP_SOURCES = \
          $(addprefix macosx/cfcpp/,$(notdir $(wildcard $(PROJ_SRC_DIR)/macosx/cfcpp/*.cpp)))
SOURCES += $(CFCPP_SOURCES)

CFCPP_BaseNameSources := $(sort $(basename $(CFCPP_SOURCES)))
CFCPP_OBJECTS := $(CFCPP_BaseNameSources:%=$(ObjDir)/%.o)

# Make sure the cfcpp output directory exists
$(CFCPP_OBJECTS): $(ObjDir)/cfcpp/.dir
endif

ifeq ($(HOST_OS),Linux)
$(eval $(call DIR_SOURCES,posix))
$(eval $(call DIR_SOURCES,linux))
endif

ifneq (,$(filter $(HOST_OS), FreeBSD GNU/kFreeBSD))
$(eval $(call DIR_SOURCES,posix))
$(eval $(call DIR_SOURCES,freebsd))
endif

ifeq ($(HOST_OS),MingW)
$(eval $(call DIR_SOURCES,windows))
endif

ifeq ($(HOST_OS),Android)
$(eval $(call DIR_SOURCES,posix))
$(eval $(call DIR_SOURCES,linux))
$(eval $(call DIR_SOURCES,android))
endif

LIBRARYNAME := lldbHost
BUILD_ARCHIVE = 1

include $(LLDB_LEVEL)/Makefile
