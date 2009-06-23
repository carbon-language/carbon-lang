##===- llvmc/example/Skeleton/driver/Makefile --------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open
# Source License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = $(LLVMC_BASE_LEVEL)/..
include $(LEVEL)/Makefile.config

TOOLNAME = $(LLVMC_BASED_DRIVER_NAME)
LLVMLIBS = CompilerDriver.a
LINK_COMPONENTS = support system
REQUIRES_EH := 1

# Preprocessor magic that generates references to static variables in built-in
# plugins.
# TODO: Move this to Makefile.rules? (also used by examples/{Skeleton, mcc16})
ifneq ($(LLVMC_BUILTIN_PLUGINS),)

USEDLIBS += $(patsubst %,plugin_llvmc_%.a,$(LLVMC_BUILTIN_PLUGINS))

LLVMC_BUILTIN_PLUGIN_1 = $(word 1, $(LLVMC_BUILTIN_PLUGINS))
LLVMC_BUILTIN_PLUGIN_2 = $(word 2, $(LLVMC_BUILTIN_PLUGINS))
LLVMC_BUILTIN_PLUGIN_3 = $(word 3, $(LLVMC_BUILTIN_PLUGINS))
LLVMC_BUILTIN_PLUGIN_4 = $(word 4, $(LLVMC_BUILTIN_PLUGINS))
LLVMC_BUILTIN_PLUGIN_5 = $(word 5, $(LLVMC_BUILTIN_PLUGINS))

ifneq ($(LLVMC_BUILTIN_PLUGIN_1),)
CPP.Flags += -DLLVMC_BUILTIN_PLUGIN_1=$(LLVMC_BUILTIN_PLUGIN_1)
endif

ifneq ($(LLVMC_BUILTIN_PLUGIN_2),)
CPP.Flags += -DLLVMC_BUILTIN_PLUGIN_2=$(LLVMC_BUILTIN_PLUGIN_2)
endif

ifneq ($(LLVMC_BUILTIN_PLUGIN_3),)
CPP.Flags += -DLLVMC_BUILTIN_PLUGIN_3=$(LLVMC_BUILTIN_PLUGIN_3)
endif

ifneq ($(LLVMC_BUILTIN_PLUGIN_4),)
CPP.Flags += -DLLVMC_BUILTIN_PLUGIN_4=$(LLVMC_BUILTIN_PLUGIN_4)
endif

ifneq ($(LLVMC_BUILTIN_PLUGIN_5),)
CPP.Flags += -DLLVMC_BUILTIN_PLUGIN_5=$(LLVMC_BUILTIN_PLUGIN_5)
endif

endif

include $(LEVEL)/Makefile.common
