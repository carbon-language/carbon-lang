##===- docs/mk/main.mk -------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Main HTML documentation support module.
#
# The following variables must be defined before including this makefile:
#
#       HTML.Project  formal project name. eg. LLVM or Clang.
#       HTML.project  filesystem project name. eg. llvm or clang.
#
##===----------------------------------------------------------------------===##

include $(LLVM_SRC_ROOT)/docs/mk/common.defs.mk

MAIN.in/   = $(PROJ_SRC_DIR)/
MAIN.out/  = ./

MAIN.html/        = $(MAIN.out/)html/
MAIN.html.tar     = $(MAIN.out/)html.tar.gz
MAIN.html.cp.ext  = .html .css .txt .png .jpg .gif
MAIN.html.cp.in   = $(patsubst $(MAIN.in/)%,%, \
  $(sort $(wildcard $(foreach x,$(MAIN.html.cp.ext), $(MAIN.in/)*$(x)     ))) \
  $(sort $(wildcard $(foreach x,$(MAIN.html.cp.ext), $(MAIN.in/)*/*$(x)   ))) \
  $(sort $(wildcard $(foreach x,$(MAIN.html.cp.ext), $(MAIN.in/)*/*/*$(x) ))) )
MAIN.html.cp.out  = $(MAIN.html.cp.in:%=$(MAIN.html/)%)
MAIN.html.pod.in  = $(patsubst $(MAIN.in/)%,%, \
                                    $(sort $(wildcard $(MAIN.in/)*.pod     )) \
                                    $(sort $(wildcard $(MAIN.in/)*.pod     )) \
                                    $(sort $(wildcard $(MAIN.in/)*/*.pod   )) \
                                    $(sort $(wildcard $(MAIN.in/)*/*/*.pod )) )
MAIN.html.pod.out = $(MAIN.html.pod.in:%.pod=$(MAIN.html/)%.html)

MAIN.html.files   = $(MAIN.html.cp.out) \
                    $(MAIN.html.pod.out)

MAIN.man/    = $(MAIN.out/)man/man1/
MAIN.man.in  = $(sort $(wildcard $(MAIN.in/)*.pod     )) \
               $(sort $(wildcard $(MAIN.in/)*/*.pod   )) \
               $(sort $(wildcard $(MAIN.in/)*/*/*.pod ))
MAIN.man.out = $(patsubst %.pod,$(MAIN.man/)%.1,$(notdir $(MAIN.man.in)))

MAIN.ps/     = $(MAIN.out/)ps/
MAIN.ps.out  = $(patsubst %.1,$(MAIN.ps/)%.ps,$(notdir $(MAIN.man.out)))

MAIN.pdf/    = $(MAIN.out/)pdf/
MAIN.pdf.out = $(patsubst %.1,$(MAIN.pdf/)%.pdf,$(notdir $(MAIN.man.out)))

MAIN.files += $(MAIN.html.tar)
MAIN.files += $(MAIN.html.cp.out)
MAIN.files += $(MAIN.html.pod.out)
MAIN.files += $(MAIN.man.out)
MAIN.files += $(MAIN.ps.out)
MAIN.files += $(MAIN.pdf.out)

DOCS.mkdir.files += $(MAIN.files)

##===----------------------------------------------------------------------===##

INSTALL.out/     = $(PROJ_prefix)/share/
INSTALL.doc/     = $(INSTALL.out/)doc/$(MAIN.project)/

INSTALL.html/    = $(INSTALL.doc/)html/
INSTALL.html.tar = $(INSTALL.doc/)$(notdir $(MAIN.html.tar))
INSTALL.html.out = $(MAIN.html.files:$(MAIN.html/)%=$(INSTALL.html/)%)

INSTALL.man/     = $(PROJ_mandir)/man1/
INSTALL.man.out  = $(MAIN.man.out:$(MAIN.man/)%=$(INSTALL.man/)%)

INSTALL.ps/      = $(INSTALL.doc/)ps/
INSTALL.ps.out   = $(MAIN.ps.out:$(MAIN.ps/)%=$(INSTALL.ps/)%)

INSTALL.pdf/     = $(INSTALL.doc/)pdf/
INSTALL.pdf.out  = $(MAIN.pdf.out:$(MAIN.pdf/)%=$(INSTALL.pdf/)%)

INSTALL.files += $(INSTALL.html.tar)
INSTALL.files += $(INSTALL.html.out)
INSTALL.files += $(INSTALL.man.out)
INSTALL.files += $(INSTALL.ps.out)
INSTALL.files += $(INSTALL.pdf.out)

DOCS.mkdir.files += $(INSTALL.out/)file-placebo
DOCS.mkdir.files += $(INSTALL.files)

##===----------------------------------------------------------------------===##

clean-local::
	-$(Verb) $(RM) -f $(MAIN.files)

all:: docs docs-tar
install-local:: install-docs
uninstall-local:: uninstall-docs

##===----------------------------------------------------------------------===##

.PHONY: docs-tar
docs-tar: $(MAIN.html.tar)

.PHONY: docs
docs: $(MAIN.html.files)
docs: $(MAIN.man.out)
ifeq ($(ENABLE_DOXYGEN),1)
ifneq (,$(GROFF))
docs: $(MAIN.ps.out)
endif
ifneq (,$(PDFROFF))
docs: $(MAIN.pdf.out)
endif
endif

$(MAIN.html.tar): | $(dir $(MAIN.html.tar))
$(MAIN.html.tar): $(MAIN.html.files)
	$(Echo) Creating $(MAIN.Project) MAIN documentation tarball
	$(Verb) (set -e; cd $(MAIN.out/); $(TAR) cf - html) \
	$(call fn.PIPE.COMP,$@) > $@

$(MAIN.html.cp.out): | $(dir $(MAIN.html.cp.out))
$(MAIN.html.cp.out): $(MAIN.html/)%: %
	$(Echo) Copying to $(@:$(PROJ_OBJ_ROOT)/%=%)
	$(Verb) $(CP) $< $@

$(MAIN.html.pod.out): | $(dir $(MAIN.html.pod.out))
$(MAIN.html.pod.out): $(MAIN.html/)%.html: %.pod
	$(Echo) Converting pod to $(@:$(PROJ_OBJ_ROOT)/%=%)
	$(Verb) $(CAT) $< | (set -e; cd $(MAIN.in/); $(POD2HTML) --title=$(*F) \
	  --noindex --css=manpage.css \
	  --htmlroot=. --podpath=. --podroot=$(<D)) > $@

$(MAIN.man.out): | $(dir $(MAIN.man.out))
$(MAIN.man.out): $(MAIN.man/)%.1: %.pod
	$(Echo) Converting pod to $(@:$(PROJ_OBJ_ROOT)/%=%)
	$(Verb) $(POD2MAN) --release=$(PROJ_VERSION) --center="$(MAIN.man.center)" $< $@

$(MAIN.ps.out): | $(dir $(MAIN.ps.out))
$(MAIN.ps.out): $(MAIN.ps/)%.ps: $(MAIN.man/)%.1
	$(Echo) Converting man to $(@:$(PROJ_OBJ_ROOT)/%=%)
	$(Verb) $(GROFF) -Tps -man $< > $@

$(MAIN.pdf.out): | $(dir $(MAIN.pdf.out))
$(MAIN.pdf.out): $(MAIN.pdf/)%.pdf: $(MAIN.man/)%.1
	$(Echo) Converting man to $(@:$(PROJ_OBJ_ROOT)/%=%)
	$(Verb) $(PDFROFF) -man --no-toc-relocation $< > $@

# Mapping of src pod files is not always direct so we need a search vpath.
# This solution works because no man page filenames (without dir) collide.
vpath %.pod $(sort $(dir $(MAIN.man.in)))

##===----------------------------------------------------------------------===##

.PHONY: install-docs
install-docs: $(INSTALL.html.tar)
install-docs: $(INSTALL.html.out)
install-docs: $(INSTALL.man.out)
ifeq ($(ENABLE_DOXYGEN),1)
ifneq (,$(GROFF))
install-docs: $(INSTALL.ps.out)
endif
ifneq (,$(PDFROFF))
install-docs: $(INSTALL.pdf.out)
endif
endif

$(INSTALL.html.tar): | $(dir $(INSTALL.html.tar))
$(INSTALL.html.tar): $(MAIN.html.tar)
	$(Echo) Installing $(MAIN.Project) MAIN documentation tarball
	$(Verb) $(DataInstall) $< $@

$(INSTALL.html.out): | $(dir $(INSTALL.html.out))
$(INSTALL.html.out): $(INSTALL.html/)%: $(MAIN.html/)%
	$(Echo) Installing $(@:$(INSTALL.out/)%=%)
	$(Verb) $(DataInstall) $< $@

$(INSTALL.man.out): | $(dir $(INSTALL.man.out))
$(INSTALL.man.out): $(INSTALL.man/)%: $(MAIN.man/)%
	$(Echo) Installing $(@:$(INSTALL.out/)%=%)
	$(Verb) $(DataInstall) $< $@

$(INSTALL.ps.out): | $(dir $(INSTALL.ps.out))
$(INSTALL.ps.out): $(INSTALL.ps/)%: $(MAIN.ps/)%
	$(Echo) Installing $(@:$(INSTALL.out/)%=%)
	$(Verb) $(DataInstall) $< $@

$(INSTALL.pdf.out): | $(dir $(INSTALL.pdf.out))
$(INSTALL.pdf.out): $(INSTALL.pdf/)%: $(MAIN.pdf/)%
	$(Echo) Installing $(@:$(INSTALL.out/)%=%)
	$(Verb) $(DataInstall) $< $@

uninstall-docs:
	$(Echo) Uninstalling $(MAIN.Project) MAIN documentation
	-$(Verb) $(RM) -f $(INSTALL.files)

##===----------------------------------------------------------------------===##

DOCS.vars.mandatory += MAIN.Project MAIN.project MAIN.man.center
DOCS.vars.print     += $(sort $(filter INSTALL.%/,$(.VARIABLES)))

HELP.sections += MAIN.help

define MAIN.help
  MAIN Documentation Module. This module is objdir build-friendly.

  ------------------------------------------------------------------------------
  TARGET          NOTES
  ------------------------------------------------------------------------------
 *all             invokes target docs
 *install         invokes target install-docs
 *uninstall       invokes target uninstall-docs
  ------------------------------------------------------------------------------
  docs            copy MAIN docs from
                  $(MAIN.in/)
                  with extensions { $(MAIN.html.cp.ext) }

                  and perform conversions:

                  FROM     TO    DESTINATION
                  --------------------------------------------------------------
                  html ->  html  $(MAIN.html/)
                  pod  ->  man   $(MAIN.man/)
                  man  ->  ps    $(MAIN.ps/)
                  man  ->  pdf   $(MAIN.pdf/)
  ------------------------------------------------------------------------------
  docs-tar        create docs tarball $(MAIN.html.tar)
  clean           remove built files
  install-docs    install MAIN docs:

                  FROM     TO    DESTINATION
                  --------------------------------------------------------------
                  html ->  html  $(INSTALL.html/)
                  pod  ->  man   $(INSTALL.man/)
                  man  ->  ps    $(INSTALL.ps/)
                  man  ->  pdf   $(INSTALL.pdf/)
  ------------------------------------------------------------------------------
  uninstall-docs  remove installed files
  ------------------------------------------------------------------------------
  (targets marked with '*' require configure --enable-doxygen)
endef

include $(LLVM_SRC_ROOT)/docs/mk/common.rules.mk
