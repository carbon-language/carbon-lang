##===- tools/llvm-config/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and Eric Kidd and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..

EXTRA_DIST = LibDeps.txt FinalLibDeps.txt llvm-config.in.in find-cycles.pl
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common

# If we don't have Perl, we can't generate the library dependencies upon which 
# llvm-config depends. Therefore, only if we detect perl will we do anything
# useful.
ifeq ($(HAVE_PERL),1)

# Combine preprocessor flags (except for -I) and CXX flags.
SUB_CXXFLAGS = ${CPP.BaseFlags} ${CXX.Flags}

# This is blank for now.  We need to be careful about adding stuff here:
# LDFLAGS tend not to be portable, and we don't currently require the
# user to use libtool when linking against LLVM.
SUB_LDFLAGS = 

FinalLibDeps = $(PROJ_OBJ_DIR)/FinalLibDeps.txt
LibDeps      = $(PROJ_OBJ_DIR)/LibDeps.txt
GenLibDeps   = $(PROJ_SRC_ROOT)/utils/GenLibDeps.pl

$(LibDeps): $(GenLibDeps) $(LibDir) $(wildcard $(LibDir)/*.a $(LibDir)/*.o)
	$(Echo) "Regenerating LibDeps.txt"
	$(Verb) $(GenLibDeps) -flat $(LibDir) $(NM_PATH) | sort > $(LibDeps)

# Find all the cyclic dependencies between various LLVM libraries, so we
# don't have to process them at runtime.
$(FinalLibDeps): find-cycles.pl $(LibDeps)
	$(Echo) "Finding cyclic dependencies between LLVM libraries."
	$(Verb) $(PERL) $< < $(LibDeps) > $@

# Rerun our configure substitutions as needed.
ConfigInIn = $(PROJ_SRC_DIR)/llvm-config.in.in
llvm-config.in: $(ConfigInIn) $(ConfigStatusScript)
	$(Verb) cd $(PROJ_OBJ_ROOT) ; \
		$(ConfigStatusScript) tools/llvm-config/llvm-config.in

# Build our final script.
$(ToolDir)/llvm-config: llvm-config.in $(FinalLibDeps)
	$(Echo) "Building llvm-config script."
	$(Verb) $(ECHO) 's,@LLVM_CXXFLAGS@,$(SUB_CXXFLAGS),' > temp.sed
	$(Verb) $(ECHO) 's,@LLVM_LDFLAGS@,$(SUB_LDFLAGS),' >> temp.sed
	$(Verb) $(ECHO) 's,@LLVM_BUILDMODE@,$(BuildMode),' >> temp.sed
	$(Verb) $(SED) -f temp.sed < $< > $@
	$(Verb) $(RM) temp.sed
	$(Verb) cat $(FinalLibDeps) >> $@
	$(Verb) chmod +x $@

else
# We don't have perl, just generate a dummy llvm-config
$(ToolDir)/llvm-config:
	$(Echo) "Building place holder llvm-config script."
	$(Verb) $(ECHO) 'echo llvm-config: Perl not found so llvm-config could not be generated' >> $@
	$(Verb) chmod +x $@

endif
# Hook into the standard Makefile rules.
all-local:: $(ToolDir)/llvm-config
clean-local::
	$(Verb) $(RM) -f $(ToolDir)/llvm-config llvm-config.in $(FinalLibDeps) \
	  $(LibDeps) GenLibDeps.out
install-local:: all-local
	$(Echo) Installing llvm-config
	$(Verb) $(MKDIR) $(PROJ_bindir)
	$(Verb) $(ScriptInstall) $(ToolDir)/llvm-config $(PROJ_bindir)

