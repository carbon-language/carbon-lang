##===- docs/CommandGuide/Makefile --------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../..

include $(LEVEL)/Makefile.common

POD := $(wildcard $(BUILD_SRC_DIR)/*.pod)

EXTRA_DIST := $(POD) index.html

HTML = $(patsubst $(BUILD_SRC_DIR)/%.pod, html/%.html, $(POD))
MAN = $(patsubst $(BUILD_SRC_DIR)/%.pod, man/man1/%.1, $(POD))
PS = $(patsubst $(BUILD_SRC_DIR)/%.pod, ps/%.ps, $(POD))

all-local:: $(HTML) $(MAN) $(PS)

.SUFFIXES:
.SUFFIXES: .html .pod .1 .ps

$(HTML) : html/.dir man/.dir man/man1/.dir ps/.dir

html/%.html: %.pod
	$(POD2HTML) --css=manpage.css --htmlroot=. --podpath=. \
	  --noindex --infile=$< --outfile=$@ --title=$*

man/man1/%.1: %.pod
	$(POD2MAN) --release=$(PACKAGE_VERSION) \
	  --center="LLVM Command Guide" $< $@

ps/%.ps: man/man1/%.1
	$(GROFF) -Tps -man $< > $@

clean-local::
	$(Verb) $(RM) -f pod2htm*.*~~ $(HTML) $(MAN) $(PS)

install-local:: $(HTML) $(MAN) $(PS)
	$(Echo) Installing HTML CommandGuide Documentation
	$(Verb) $(INSTALL) -d $(LLVM_DOCSDIR)/html/CommandGuide
	$(Verb) $(INSTALL) -C $(HTML) $(LLVM_DOCSDIR)/html/CommandGuide
	$(Echo) Installing MAN CommandGuide Documentation
	$(Verb) $(INSTALL) -d $(LLVM_MANDIR)/man1
	$(Verb) $(INSTALL) -C $(MAN) $(LLVM_MANDIR)/man1
	$(Echo) Installing PS CommandGuide Documentation
	$(Verb) $(INSTALL) -d $(LLVM_DOCSDIR)/ps
	$(Verb) $(INSTALL) -C $(PS) $(LLVM_DOCSDIR)/ps

uninstall-local::
	$(Echo) Uninstalling Documentation
	$(Verb) $(RM) -rf $(LLVM_DOCSDIR)

printvars::
	$(Echo) "POD            : " '$(POD)'
	$(Echo) "HTML           : " '$(HTML)'
