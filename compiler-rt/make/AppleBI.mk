
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

# We override this with RC_ARCHS because B&I may want to build on an
# ARCH we haven't explicitly defined support for. If all goes well,
# this will just work and the resulting lib will just have generic
# versions for anything unknown.
Archs := $(RC_ARCHS)

# log full compile lines in B&I logs and omit summary lines
Verb := 
Summary := @true

# list of functions needed for each architecture
Funcs_all = absvdi2.o absvsi2.o addvdi3.o addvsi3.o ashldi3.o ashrdi3.o \
            clzdi2.o clzsi2.o cmpdi2.o ctzdi2.o ctzsi2.o \
            divdc3.o divdi3.o divsc3.o ffsdi2.o \
            fixdfdi.o fixsfdi.o fixunsdfdi.o fixunsdfsi.o fixunssfdi.o \
            fixunssfsi.o floatdidf.o floatdisf.o floatundidf.o floatundisf.o \
            gcc_personality_v0.o lshrdi3.o moddi3.o muldc3.o muldi3.o \
            mulsc3.o mulvdi3.o mulvsi3.o negdi2.o negvdi2.o negvsi2.o \
            paritydi2.o paritysi2.o popcountdi2.o popcountsi2.o powidf2.o \
            powisf2.o subvdi3.o subvsi3.o ucmpdi2.o udivdi3.o \
            udivmoddi4.o umoddi3.o apple_versioning.o eprintf.o 
Funcs_i386    = divxc3.o fixunsxfdi.o fixunsxfsi.o fixxfdi.o floatdixf.o \
                floatundixf.o mulxc3.o powixf2.o clear_cache.o \
				enable_execute_stack.o 
Funcs_ppc     = divtc3.o fixtfdi.o fixunstfdi.o floatditf.o floatunditf.o \
                gcc_qadd.o gcc_qdiv.o gcc_qmul.o gcc_qsub.o multc3.o \
                powitf2.o restFP.o saveFP.o trampoline_setup.o \
				clear_cache.o enable_execute_stack.o
Funcs_x86_64  = absvti2.o addvti3.o ashlti3.o ashrti3.o clzti2.o cmpti2.o \
                ctzti2.o divti3.o divxc3.o ffsti2.o fixdfti.o fixsfti.o \
                fixunsdfti.o fixunssfti.o fixunsxfdi.o fixunsxfsi.o \
                fixunsxfti.o fixxfdi.o fixxfti.o floatdixf.o floattidf.o \
                floattisf.o floattixf.o floatundixf.o floatuntidf.o \
                floatuntisf.o floatuntixf.o lshrti3.o modti3.o multi3.o \
                mulvti3.o mulxc3.o negti2.o negvti2.o parityti2.o \
                popcountti2.o powixf2.o subvti3.o ucmpti2.o udivmodti4.o \
                udivti3.o umodti3.o clear_cache.o enable_execute_stack.o
Funcs_armv6 =	adddf3vfp.o addsf3vfp.o bswapdi2.o bswapsi2.o divdf3vfp.o \
				divsf3vfp.o eqdf2vfp.o eqsf2vfp.o extendsfdf2vfp.o \
				fixdfsivfp.o fixsfsivfp.o fixunsdfsivfp.o fixunssfsivfp.o \
				floatsidfvfp.o floatsisfvfp.o floatunssidfvfp.o floatunssisfvfp.o \
				gedf2vfp.o gesf2vfp.o gtdf2vfp.o gtsf2vfp.o \
				ledf2vfp.o lesf2vfp.o ltdf2vfp.o ltsf2vfp.o \
				muldf3vfp.o mulsf3vfp.o \
				nedf2vfp.o negdf2vfp.o negsf2vfp.o nesf2vfp.o \
				subdf3vfp.o subsf3vfp.o truncdfsf2vfp.o unorddf2vfp.o unordsf2vfp.o \
				modsi3.o umodsi3.o udivsi3.o divsi3.o switch.o save_restore_d8_d15.o


# copies any public headers to DSTROOT
installhdrs:


# copies source code to SRCROOT
installsrc:
	cp -r . $(SRCROOT)


# copy results to DSTROOT
install:  $(SYMROOT)/usr/local/lib/system/libcompiler_rt.a
	mkdir -p $(DSTROOT)/usr/local/lib/system
	cp $(SYMROOT)/usr/local/lib/system/libcompiler_rt.a \
				$(DSTROOT)/usr/local/lib/system/libcompiler_rt.a
	cd $(DSTROOT)/usr/local/lib/system; \
	ln -s libcompiler_rt.a libcompiler_rt_profile.a; \
	ln -s libcompiler_rt.a libcompiler_rt_debug.a


# rule to make fat libcompiler_rt.a
$(SYMROOT)/usr/local/lib/system/libcompiler_rt.a : $(foreach arch,$(Archs), \
                                                    $(OBJROOT)/$(arch)-pruned.a)
	mkdir -p $(SYMROOT)/usr/local/lib/system
	lipo -create $^ -o  $@


# rule to make filter each architecture of libcompiler_rt.a
# adds project info so that "what /usr/lib/libSystem.B.dylib" will work
$(OBJROOT)/%-pruned.a : $(OBJROOT)/Release/%/libcompiler_rt.Optimized.a
	mkdir -p $(OBJROOT)/$*.tmp	
	cd $(OBJROOT)/$*.tmp; \
	/Developer/Makefiles/bin/version.pl $(RC_ProjectName) > $(OBJROOT)/version.c; \
	gcc -arch $* -c ${OBJROOT}/version.c -o version.o; \
	ar -x $<  $(Funcs_all) $(Funcs_$*); \
	libtool -static *.o -o $@
