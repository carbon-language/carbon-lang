##===- tools/llvmc2/Makefile -------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open
# Source License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

# Compiled-in plugins
##### FIXME: This breaks the build.
#####BUILTIN_PLUGINS = Base

LEVEL = ../..
TOOLNAME = llvmc2
LINK_COMPONENTS = support system
REQUIRES_EH := 1

ifneq ($(BUILTIN_PLUGINS),)

export BUILTIN_LLVMC_PLUGIN=1
USEDLIBS = $(patsubst %,LLVMC%,$(BUILTIN_PLUGINS))

endif

include $(LEVEL)/Makefile.common

TD_COMMON = $(wildcard $(LLVM_SRC_ROOT)/include/llvm/CompilerDriver/*.td)

# There is probably a better way to do this: currently we enter the
# subdirectory 2 times - the second time is not needed.
# This probably also needs to be integrated into Makefile.rules.
define PLUGIN_template
PLUGIN_$(1)_SOURCES=$$(wildcard plugins/$(1)/*.cpp)
PLUGIN_$(1)_TD_SOURCES=$$(wildcard plugins/$(1)/*.cpp)

$$(LibDir)/LLVMC$(1).o: $$(PLUGIN_$(1)_SOURCES) $$(PLUGIN_$(1)_TD_SOURCES) \
			$$(TD_COMMON)
	@$$(MAKE) -C plugins/$(1)

$$(RecursiveTargets) ::
	@$$(MAKE) -C plugins/$(1) $$@
endef

$(foreach plugin,$(BUILTIN_PLUGINS),$(eval $(call PLUGIN_template,$(plugin))))
