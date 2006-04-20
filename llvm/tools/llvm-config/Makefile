##===- tools/llvm-config/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and Eric Kidd and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..

EXTRA_DIST = LibDeps.txt llvm-config.in.in find-cycles.pl

include $(LEVEL)/Makefile.common

# Combine preprocessor flags (except for -I) and CXX flags.
SUB_CXXFLAGS = ${CPP.BaseFlags} ${CXX.Flags}

# This is blank for now.  We need to be careful about adding stuff here:
# LDFLAGS tend not to be portable, and we don't currently require the
# user to use libtool when linking against LLVM.
SUB_LDFLAGS = 

FinalLibDeps = $(PROJ_SRC_DIR)/FinalLibDeps.txt
ifdef HAVE_PERL
ifeq ($(HAVE_PERL),1)
LibDeps = $(PROJ_SRC_DIR)/LibDeps.txt
GenLibDeps = $(PROJ_SRC_ROOT)/utils/GenLibDeps.pl
# MANUAL USE ONLY!  GenLibDeps.pl is very non-portable, so LibDeps.txt
# should only be re-built manually.  No other rule in this file should
# depend on LibDeps.txt.
$(LibDeps): $(GenLibDeps) $(LibDir)
	$(Echo) "Regenerating LibDeps.txt"
	$(Verb) $(GenLibDeps) -flat $(LibDir) | sort > $(LibDeps)

# Find all the cyclic dependencies between various LLVM libraries, so we
# don't have to process them at runtime.
$(FinalLibDeps): find-cycles.pl $(LibDeps)
	$(Echo) "Finding cyclic dependencies between LLVM libraries."
	$(Verb) $(PERL) $< < $(PROJ_SRC_DIR)/LibDeps.txt > $@
endif
endif

# Rerun our configure substitutions as needed.
ConfigInIn = $(PROJ_SRC_DIR)/llvm-config.in.in
llvm-config.in: $(ConfigInIn) $(ConfigStatusScript)
	$(Verb) cd $(PROJ_OBJ_ROOT) ; \
		$(ConfigStatusScript) tools/llvm-config/llvm-config.in

# Build our final script.
llvm-config: llvm-config.in $(FinalLibDeps)
	$(Echo) "Building llvm-config script."
	$(Verb) $(ECHO) 's,@LLVM_CXXFLAGS@,$(SUB_CXXFLAGS),' > temp.sed
	$(Verb) $(ECHO) 's,@LLVM_LDFLAGS@,$(SUB_LDFLAGS),' >> temp.sed
	$(Verb) $(ECHO) 's,@CORE_IS_ARCHIVE@,$(CORE_IS_ARCHIVE),' >> temp.sed
	$(Verb) $(SED) -f temp.sed < $< > $@
	$(Verb) $(RM) temp.sed
	$(Verb) cat $(FinalLibDeps) >> $@
	$(Verb) chmod +x llvm-config

# Hook into the standard Makefile rules.
all-local:: llvm-config
clean-local::
	$(Verb) $(RM) -f llvm-config llvm-config.in
install-local:: all-local
	$(Echo) Installing llvm-config
	$(Verb) $(MKDIR) $(PROJ_bindir)
	$(Verb) $(ScriptInstall) llvm-config $(PROJ_bindir)
