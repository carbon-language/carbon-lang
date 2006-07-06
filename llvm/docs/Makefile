##===- docs/Makefile ---------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL      := ..
DIRS       := CommandGuide

ifdef BUILD_FOR_WEBSITE
PROJ_OBJ_DIR = .
DOXYGEN = doxygen

doxygen.cfg: doxygen.cfg.in
	cat $< | sed 's/@abs_top_srcdir@/../g' | sed 's/@DOT@/dot/g' | \
        sed 's/@PACKAGE_VERSION@/CVS/g' | sed 's/@abs_top_builddir@/../g' > $@
endif

include $(LEVEL)/Makefile.common

HTML       := $(wildcard $(PROJ_SRC_DIR)/*.html) \
              $(wildcard $(PROJ_SRC_DIR)/*.css)
IMAGES     := $(wildcard $(PROJ_SRC_DIR)/img/*.*)
DOXYFILES  := doxygen.cfg.in doxygen.css doxygen.footer doxygen.header \
              doxygen.intro
EXTRA_DIST := $(HTML) $(DOXYFILES) llvm.css CommandGuide img

.PHONY: install-html install-doxygen doxygen

ifeq ($(ENABLE_DOXYGEN),1)
install-local:: install-html install-doxygen
else
install-local:: install-html
endif

install-html: $(PROJ_OBJ_DIR)/html.tar.gz
	$(Echo) Installing HTML documentation
	$(Verb) $(MKDIR) $(PROJ_docsdir)/html
	$(Verb) $(MKDIR) $(PROJ_docsdir)/html/img
	$(Verb) $(DataInstall) $(HTML) $(PROJ_docsdir)/html
	$(Verb) $(DataInstall) $(IMAGES) $(PROJ_docsdir)/html/img
	$(Verb) $(DataInstall) $(PROJ_OBJ_DIR)/html.tar.gz $(PROJ_docsdir)

$(PROJ_OBJ_DIR)/html.tar.gz: $(HTML)
	$(Echo) Packaging HTML documentation
	$(Verb) $(RM) -rf $@ $(PROJ_OBJ_DIR)/html.tar
	$(Verb) cd $(PROJ_SRC_DIR) && \
	  $(TAR) cf $(PROJ_OBJ_DIR)/html.tar *.html
	$(Verb) $(GZIP) $(PROJ_OBJ_DIR)/html.tar

install-doxygen: doxygen
	$(Echo) Installing doxygen documentation
	$(Echo) Installing doxygen documentation
	$(Verb) $(MKDIR) $(PROJ_docsdir)/html/doxygen
	$(Verb) $(DataInstall) $(PROJ_OBJ_DIR)/doxygen.tar.gz $(PROJ_docsdir)
	$(Verb) cd $(PROJ_OBJ_DIR)/doxygen && \
	  $(FIND) . -type f -exec \
	    $(DataInstall) {} $(PROJ_docsdir)/html/doxygen \;

doxygen: $(PROJ_OBJ_DIR)/doxygen.tar.gz

$(PROJ_OBJ_DIR)/doxygen.tar.gz: $(DOXYFILES) $(PROJ_OBJ_DIR)/doxygen.cfg
	$(Echo) Building doxygen documentation
	$(Verb) if test -e $(PROJ_OBJ_DIR)/doxygen ; then \
	  $(RM) -rf $(PROJ_OBJ_DIR)/doxygen ; \
	fi
	$(Verb) $(DOXYGEN) $(PROJ_OBJ_DIR)/doxygen.cfg
	$(Echo) Packaging doxygen documentation
	$(Verb) $(RM) -rf $@ $(PROJ_OBJ_DIR)/doxygen.tar
	$(Verb) $(TAR) cf $(PROJ_OBJ_DIR)/doxygen.tar doxygen
	$(Verb) $(GZIP) $(PROJ_OBJ_DIR)/doxygen.tar
	$(Verb) $(CP) $(PROJ_OBJ_DIR)/doxygen.tar.gz $(PROJ_OBJ_DIR)/doxygen/html/

userloc: $(LLVM_SRC_ROOT)/docs/userloc.html

$(LLVM_SRC_ROOT)/docs/userloc.html:
	$(Echo) Making User LOC Table
	$(Verb) cd $(LLVM_SRC_ROOT) ; ./utils/userloc.pl -details -recurse \
	  -html lib include tools runtime utils examples autoconf test > docs/userloc.html
	  
