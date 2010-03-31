
#
# Make rules to build compiler_rt in Apple B&I infrastructure
#

# set ProjSrcRoot appropriately
ProjSrcRoot := $(SRCROOT)
# set ProjObjRoot appropriately
ifdef OBJROOT
  ProjObjRoot := $(OBJROOT)
else
  ProjObjRoot := $(ProjSrcRoot)
endif

# Log full compile lines in B&I logs and omit summary lines.
Verb :=
Summary := @true

# List of functions needed for each architecture.

# Copies any public headers to DSTROOT.
installhdrs:


# Copies source code to SRCROOT.
installsrc:
	cp -r . $(SRCROOT)


# Copy results to DSTROOT.
install:  $(SYMROOT)/libcompiler_rt.dylib
	mkdir -p $(DSTROOT)/usr/lib/system
	strip -S $(SYMROOT)/libcompiler_rt.dylib \
	    -o $(DSTROOT)/usr/lib/system/libcompiler_rt.dylib
	cd $(DSTROOT)/usr/lib/system; \
	    ln -s libcompiler_rt.dylib libcompiler_rt_profile.dylib; \
	    ln -s libcompiler_rt.dylib libcompiler_rt_debug.dylib

# Rule to make each dylib slice
$(OBJROOT)/libcompiler_rt-%.dylib : $(OBJROOT)/darwin_bni/Release/%/libcompiler_rt.a
	echo "const char vers[] = \"@(#) $(RC_ProjectName)-$(RC_ProjectSourceVersion)\"; " > $(OBJROOT)/version.c
	cc $(OBJROOT)/version.c -arch $* -dynamiclib \
	   -install_name /usr/lib/system/libcompiler_rt.dylib \
	   -compatibility_version 1 -current_version $(RC_ProjectSourceVersion) \
	   -nodefaultlibs -lSystem -umbrella System -dead_strip \
	   -Wl,-force_load,$^ -o $@ 

# Rule to make fat dylib
$(SYMROOT)/libcompiler_rt.dylib: $(foreach arch,$(RC_ARCHS), \
									$(OBJROOT)/libcompiler_rt-$(arch).dylib)
	lipo -create $^ -o  $@

