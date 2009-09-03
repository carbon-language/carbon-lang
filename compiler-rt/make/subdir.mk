# This file is intended to be included from each subdirectory
# makefile.

ifeq ($(Dir),)
  $(error "No Dir variable defined.")
endif

ifeq ($(DebugMake),1)
  $(info MAKE: $(Dir): Processing subdirectory)
endif

# Expand template for each configuration and architecture.
# FIXME: This level of logic should be in primary Makefile?
ifeq ($(OnlyConfigs),)
  ConfigsToTraverse := $(Configs)
else
  ConfigsToTraverse := $(OnlyConfigs)
endif

ifeq ($(OnlyArchs),)
  ArchsToTraverse := $(Archs)
else
  ArchsToTraverse := $(OnlyArchs)
endif

# If we are only targetting a single arch, only traverse that.
ifneq ($(TargetArch),)
  ArchsToTraverse := $(filter $(TargetArch), $(ArchsToTraverse))
endif

$(foreach config,$(ConfigsToTraverse), \
  $(foreach arch,$(ArchsToTraverse), \
    $(eval $(call CNA_subdir_template,$(config),$(arch),$(Dir)))))

###
# Include child makefile fragments

# Evaluate this now so we do not have to worry about order of
# evaluation.
SubDirsList := $(SubDirs:%=$(Dir)/%)
ifeq ($(SubDirsList),)
else
  ifeq ($(DebugMake),1)
    $(info MAKE: Descending into subdirs: $(SubDirsList))
  endif
  $(foreach subdir,$(SubDirsList),$(eval include $(subdir)/Makefile.mk))
endif

