##===- docs/mk/api.mk --------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Generated API documentation support module.
#
# The following variables must be defined before including this makefile:
#
#       API.Project  formal project name. eg. LLVM or Clang.
#       API.project  filesystem project name. eg. llvm or clang.
#       API.srcdir   top-most source dir used in doxygen .cfg file.
#
##===----------------------------------------------------------------------===##

include $(LLVM_SRC_ROOT)/docs/mk/common.defs.mk

API.in/  = $(PROJ_SRC_DIR)/
API.out/ = ./

API.html/    = $(API.out/)html/api/
API.html.tar = $(API.out/)html-api.tar.gz

API.doxygen                     = $(DOXYGEN)
API.doxygen.target              = $(API.html/)index.html
API.doxygen.extradeps           = $(foreach x,html xml h,$(wildcard $(API.in/)*.$(x)))
API.doxygen.cfg                 = $(API.out/)doxygen.cfg
API.doxygen.cfg.srcdir          = $(API.srcdir)
API.doxygen.cfg.objdir          = .
API.doxygen.cfg.output_dir      = $(API.out/)
API.doxygen.cfg.dot             = # blank and doxygen will search path for 'dot'
API.doxygen.cfg.opts            = $(call SELECT,API.doxygen.cfg.opts,$(HOST_OS))
API.doxygen.cfg.opts.__.Darwin  = DOT_FONTNAME=Monaco DOT_FONTSIZE=8
API.doxygen.cfg.opts.__.Linux   = DOT_FONTNAME=FreeSans DOT_FONTSIZE=9
API.doxygen.cfg.opts.__.default = # values from file
API.doxygen.cfg.version         = $(call fn.SVN.LCREV,$(API.srcdir))

API.doxygen.css = $(API.html/)api.css \
                  $(API.html/)api.ie.css

ifdef VERBOSE
API.doxygen.cfg.opts := QUIET=NO WARN_IF_DOC_ERROR=YES $(API.doxygen.cfg.opts)
endif

API.files += $(API.html.tar)
API.files += $(API.doxygen.cfg)
API.files += $(API.doxygen.css)
API.files += $(API.doxygen.target)

DOCS.mkdir.files += $(API.files)

##===----------------------------------------------------------------------===##

INSTALL.out/        = $(PROJ_prefix)/share/
INSTALL.doc/        = $(INSTALL.out/)doc/$(API.project)/

INSTALL.html/       = $(INSTALL.doc/)html/api/
INSTALL.html.target = $(API.doxygen.target:$(API.html/)%=$(INSTALL.html/)%)
INSTALL.html.tar    = $(INSTALL.doc/)$(notdir $(API.html.tar))

INSTALL.files += $(INSTALL.html.tar)

DOCS.mkdir.files += $(INSTALL.doc/)file-placebo
DOCS.mkdir.files += $(INSTALL.files)

##===----------------------------------------------------------------------===##

clean-local::
	-$(Verb) $(RM) -f $(API.files)
	-$(Verb) $(call fn.RMRF,$(API.html/))

ifeq ($(ENABLE_DOXYGEN),1)
all:: docs docs-tar
install-local:: install-docs
uninstall-local:: uninstall-docs
endif

##===----------------------------------------------------------------------===##

.PHONY: docs-tar
docs-tar: $(API.html.tar)

.PHONY: docs
docs: $(API.doxygen.target)

$(API.html.tar): | $(dir $(API.html.tar))
$(API.html.tar): $(API.doxygen.target)
	$(Echo) Creating $(API.Project) API documentation tarball
	$(Verb) (set -e; cd $(API.out/); \
	  $(TAR) cf - --exclude='*.md5' --exclude='*.map' html/api) \
	  $(call fn.PIPE.COMP,$@) > $@

$(API.doxygen.cfg): | $(dir $(API.doxygen.cfg))
$(API.doxygen.cfg): $(notdir $(API.doxygen.cfg)).in
	$(Echo) Generating $(API.Project) doxygen config
	$(Verb) $(CAT) $< | $(SED) \
	  -e 's,@srcdir@,$(API.doxygen.cfg.srcdir),g' \
	  -e 's,@objdir@,$(API.doxygen.cfg.objdir),g' \
	  -e 's,@output_dir@,$(API.doxygen.cfg.output_dir),g' \
	  -e 's,@dot@,$(API.doxygen.cfg.dot),g' \
	  -e 's,@version@,$(API.doxygen.cfg.version),g' \
	  > $@

$(API.html/)api.css: | $(dir $(API.html/)api.css)
$(API.html/)api.css: api.css
	$(Echo) Copying $(API.Project) doxygen stylesheet
	$(Verb) $(CP) $< $@

# IE misbehaves when browser-specific constructs are used.
# This target strips them out to create an IE-specific css file.
# The following is an example of setting background to an extension.
# With IE instead of skipping an unrecognized extension it resets
# the background:
#
#   background: -webkit-gradient(...)
#
# Note this simple approach assumes source has strippable single-lines.
#
$(API.html/)api.ie.css: | $(dir $(API.html/)api.ie.css)
$(API.html/)api.ie.css: api.css
	$(Echo) Generating $(API.Project) doxygen stylesheet for IE
	$(Verb) $(CAT) $< | egrep -v -e '-(moz|webkit)' > $@

# Generate API docs.
#
# Define API.nodot=1 to not use 'dot' tool even if available.
# In this mode doxygen has built-in support to generate only class-diagrams
# and thus all other diagrams are skipped. Shaves 80% off generate time.
#
# We pipe (doxygen.cfg + overrides) to doxygen. This allows us to override
# almost any setting in doxygen.cfg file without having to edit it.
#
ifneq (undefined,$(origin API.nodot))
$(API.doxygen.target): API.doxygen.cfg.opts += HAVE_DOT=NO
$(API.doxygen.target): API.doxygen.target.msg = " (FAST)"
endif
$(API.doxygen.target): | $(dir $(API.doxygen.target))
$(API.doxygen.target): $(API.doxygen.cfg)
$(API.doxygen.target): $(API.doxygen.css)
$(API.doxygen.target): $(API.doxygen.extradeps)
	$(Echo) Generating $(API.Project) API documentation$(API.doxygen.target.msg)
	$(Verb) ($(CAT) $(API.doxygen.cfg)$(foreach n,$(API.doxygen.cfg.opts),; echo '$n')) | \
	  $(API.doxygen) -

##===----------------------------------------------------------------------===##

.PHONY: install-docs
install-docs: $(INSTALL.html.tar)
install-docs: $(INSTALL.html.target)

$(INSTALL.html.target): | $(INSTALL.doc/)
$(INSTALL.html.target): $(API.html.tar)
	$(Echo) Installing $(API.Project) API documentation
	$(Verb) $(CAT) $(API.html.tar) $(call fn.PIPE.DECOMP,$(API.html.tar)) | \
	  (set -e; cd $(INSTALL.doc/); $(TAR) xf -)
	@touch $@

$(INSTALL.html.tar): | $(dir $(INSTALL.html.tar))
$(INSTALL.html.tar): $(API.html.tar)
	$(Echo) Installing $(API.Project) API documentation tarball
	$(Verb) $(DataInstall) $< $@

uninstall-docs:
	$(Echo) Uninstalling $(API.Project) API documentation
	-$(Verb) $(RM) -f $(INSTALL.files)
	-$(Verb) $(call fn.RMRF,$(INSTALL.html/))

##===----------------------------------------------------------------------===##

DOCS.vars.mandatory += API.Project API.project API.srcdir
DOCS.vars.print     += $(sort $(filter INSTALL.%/,$(.VARIABLES)))

HELP.sections += API.help

define API.help
  API Documentation Module. This module is srcdir/objdir build-friendly.

  WARNING: The following directories are recursively deleted during cleanup
  procedures. Be sure not to mix files therein or bad things will happen.

    $(API.html/)
    $(INSTALL.html/)

  ------------------------------------------------------------------------------
  TARGET          NOTES
  ------------------------------------------------------------------------------
 *all             invokes target docs
 *install         invokes target install-docs
 *uninstall       invokes target uninstall-docs
  ------------------------------------------------------------------------------
  docs            generate API docs from sources using doxygen
  ------------------------------------------------------------------------------
  docs-tar        create docs tarball $(API.html.tar)
  clean           remove built files
  install-docs    install to $(INSTALL.doc/)
  uninstall-docs  remove installed files
  ------------------------------------------------------------------------------
  (targets marked with '*' require configure --enable-doxygen)
endef

include $(LLVM_SRC_ROOT)/docs/mk/common.rules.mk
