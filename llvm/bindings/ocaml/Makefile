##===- bindings/ocaml/Makefile -----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ../..
DIRS = llvm bitreader bitwriter irreader analysis target executionengine \
       transforms linker backends all_backends
ExtraMakefiles = $(PROJ_OBJ_DIR)/Makefile.ocaml

ocamldoc:
	$(Verb) for i in $(DIRS) ; do \
		$(MAKE) -C $$i ocamldoc; \
	done

include $(LEVEL)/Makefile.common
