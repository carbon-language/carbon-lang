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

include $(LEVEL)/Makefile.common

HTML       := $(wildcard $(BUILD_SRC_DIR)/*.html) \
              $(wildcard $(BUILD_SRC_DIR)/*.css)
IMAGES     := $(wildcard $(BUILD_SRC_DIR)/img/*.*)
DOXYFILES  := doxygen.cfg.in doxygen.css doxygen.footer doxygen.header \
              doxygen.intro
EXTRA_DIST := $(HTML) $(DOXYFILES) llvm.css CommandGuide img

.PHONY: install-html install-doxygen doxygen

ifeq ($(ENABLE_DOXYGEN),1)
install-local:: install-html install-doxygen
else
install-local:: install-html
endif

install-html: $(BUILD_OBJ_DIR)/html.tar.gz
	$(Echo) Installing HTML documentation
	$(Verb) $(INSTALL) -d $(LLVM_DOCSDIR)/html
	$(Verb) $(INSTALL) -d $(LLVM_DOCSDIR)/html/img
	$(Verb) $(INSTALL) -C $(HTML) $(LLVM_DOCSDIR)/html
	$(Verb) $(INSTALL) -C $(IMAGES) $(LLVM_DOCSDIR)/html/img
	$(Verb) $(INSTALL) -C $(BUILD_OBJ_DIR)/html.tar.gz $(LLVM_DOCSDIR)

$(BUILD_OBJ_DIR)/html.tar.gz: $(HTML)
	$(Echo) Packaging HTML documentation
	$(Verb) $(RM) -rf $@ $(BUILD_OBJ_DIR)/html.tar
	$(Verb) cd $(BUILD_SRC_DIR) && \
	  $(TAR) cf $(BUILD_OBJ_DIR)/html.tar *.html
	$(Verb) $(GZIP) $(BUILD_OBJ_DIR)/html.tar

install-doxygen: doxygen
	$(Echo) Installing doxygen documentation
	$(Echo) Installing doxygen documentation
	$(Verb) $(INSTALL) -d $(LLVM_DOCSDIR)/html/doxygen
	$(Verb) $(INSTALL) $(BUILD_OBJ_DIR)/doxygen.tar.gz $(LLVM_DOCSDIR)
	$(Verb) cd $(BUILD_OBJ_DIR)/doxygen && \
	  $(FIND) . -type f -exec \
	    $(INSTALL) -C {} $(LLVM_DOCSDIR)/html/doxygen \;

doxygen: $(BUILD_OBJ_DIR)/doxygen.tar.gz

$(BUILD_OBJ_DIR)/doxygen.tar.gz: $(DOXYFILES) $(BUILD_OBJ_DIR)/doxygen.cfg
	$(Echo) Building doxygen documentation
	$(Verb) if test -e $(BUILD_OBJ_DIR)/doxygen ; then \
	  $(RM) -rf $(BUILD_OBJ_DIR)/doxygen ; \
	fi
	$(Verb) $(DOXYGEN) $(BUILD_OBJ_DIR)/doxygen.cfg
	$(Echo) Packaging doxygen documentation
	$(Verb) $(RM) -rf $@ $(BUILD_OBJ_DIR)/doxygen.tar
	$(Verb) $(TAR) cf $(BUILD_OBJ_DIR)/doxygen.tar doxygen
	$(Verb) $(GZIP) $(BUILD_OBJ_DIR)/doxygen.tar

