##===- utils/llvm-config/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and Eric Kidd and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..


include $(LEVEL)/Makefile.common

# Combine preprocessor flags (except for -I) and CXX flags.
SUB_CXXFLAGS = ${CPP.BaseFlags} ${CXX.Flags}

# This is blank for now.  We need to be careful about adding stuff here:
# LDFLAGS tend not to be portable, and we don't currently require the
# user to use libtool when linking against LLVM.
SUB_LDFLAGS = 

# MANUAL USE ONLY!  GenLibDeps.pl is very non-portable, so LibDeps.txt
# should only be re-built manually.  No other rule in this file should
# depend on LibDeps.txt.
LibDeps.txt: $(LEVEL)/utils/GenLibDeps.pl $(LibDir)
	$(LEVEL)/utils/GenLibDeps.pl -flat $(LibDir) | sort > LibDeps.txt

# Find all the cyclic dependencies between various LLVM libraries, so we
# don't have to process them at runtime.
FinalLibDeps.txt: find-cycles.pl # LibDeps.txt deliberately omitted.
	$(Echo) "Finding cyclic dependencies between LLVM libraries."
	$(Verb) $< < $(PROJ_SRC_DIR)/LibDeps.txt > $@

# Rerun our configure substitutions as needed.
llvm-config.in: llvm-config.in.in $(ConfigStatusScript)
	$(Verb) cd $(PROJ_OBJ_ROOT) ; \
		$(ConfigStatusScript) utils/llvm-config/llvm-config.in

# Build our final script.
llvm-config: llvm-config.in FinalLibDeps.txt
	$(Echo) "Building llvm-config script."
	$(Verb) $(ECHO) 's,@LLVM_CXXFLAGS@,$(SUB_CXXFLAGS),' > temp.sed
	$(Verb) $(ECHO) 's,@LLVM_LDFLAGS@,$(SUB_LDFLAGS),' >> temp.sed
	$(Verb) $(ECHO) 's,@CORE_IS_ARCHIVE@,$(CORE_IS_ARCHIVE),' >> temp.sed
	$(Verb) $(SED) -f temp.sed < $< > $@
	$(Verb) $(RM) temp.sed
	$(Verb) cat FinalLibDeps.txt >> $@
	$(Verb) chmod +x llvm-config

# Hook into the standard Makefile rules.
all-local:: llvm-config
clean-local::
	$(Verb) $(RM) -f FinalLibDeps.txt llvm-config llvm-config.in
install-local:: all-local
	$(Echo) Installing llvm-config
	$(Verb) $(MKDIR) $(PROJ_bindir)
	$(Verb) $(ScriptInstall) llvm-config $(PROJ_bindir)
