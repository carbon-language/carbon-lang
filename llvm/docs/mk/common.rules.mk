##===- docs/mk/common.rules.mk -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

# Emit an error for any missing variables which are required to be defined
# before including this makefile-module.
#
_DOCS.vars.missing := $(foreach v,$(DOCS.vars.mandatory),$(if $($v),,$v))

ifneq (,$(strip $(_DOCS.vars.missing)))
$(call fn.DUMP.errorn,undefined variable,$(_DOCS.vars.missing))
endif

##===----------------------------------------------------------------------===##

# Basic target to build directory of output files.
# Opaque file lists not visible to target rules need not be added here.
#
$(sort $(dir $(DOCS.mkdir.files))):
	$(Echo) Creating directory $(@:$(PROJ_OBJ_ROOT)/%=%)
	$(Verb) $(MKDIR) $@

##===----------------------------------------------------------------------===##

# Print help defined by variables added to the help list.
#
.PHONY:
help:
	$(foreach h,$(HELP.sections),$(info $())$(info $($h)))
	$(info $())

##===----------------------------------------------------------------------===##

printvars:: $(DOCS.vars.mandatory:%=%.print.var)
printvars:: $(DOCS.vars.print:%=%.print.var)

.PHONY: %.printvar
%.print.var:
	@echo '$($*)' | awk -v name='$*' '{ printf("llvm[$(MAKELEVEL)]: %-13s:  %s\n",name,$$0) }'

##===----------------------------------------------------------------------===##

.PHONY: vars
vars: $(sort $(foreach n,$(filter-out \
    .VARIABLES $(HELP.sections) HELP.%,$(.VARIABLES)),$n.print2.var))

.PHONY: %.print2.var
%.print2.var:
	@echo "$* = $($*)"
