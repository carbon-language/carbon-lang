
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
install:  $(SYMROOT)/usr/local/lib/system/libcompiler_rt.a
	mkdir -p $(DSTROOT)/usr/local/lib/system
	cp $(SYMROOT)/usr/local/lib/system/libcompiler_rt.a \
	   $(DSTROOT)/usr/local/lib/system/libcompiler_rt.a
	cd $(DSTROOT)/usr/local/lib/system; \
	ln -s libcompiler_rt.a libcompiler_rt_profile.a; \
	ln -s libcompiler_rt.a libcompiler_rt_debug.a


# Rule to make fat libcompiler_rt.a.
$(SYMROOT)/usr/local/lib/system/libcompiler_rt.a : $(foreach arch,$(RC_ARCHS), \
                                                    $(OBJROOT)/$(arch)-pruned.a)
	mkdir -p $(SYMROOT)/usr/local/lib/system
	lipo -create $^ -o  $@


# Rule to add project info so that "what /usr/lib/libSystem.B.dylib" will work.
$(OBJROOT)/%-pruned.a : $(OBJROOT)/darwin_bni/Release/%/libcompiler_rt.a
	mkdir -p $(OBJROOT)/$*.tmp
	cd $(OBJROOT)/$*.tmp; \
	/Developer/Makefiles/bin/version.pl $(RC_ProjectName) > $(OBJROOT)/version.c; \
	gcc -arch $* -c ${OBJROOT}/version.c -o version.o; \
	ar -x $<; \
	libtool -static *.o -o $@
