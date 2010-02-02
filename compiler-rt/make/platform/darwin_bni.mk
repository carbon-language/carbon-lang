
Description := Target for Darwin using an Apple-style build.

Configs := Debug Release Profile

# We override this with RC_ARCHS because B&I may want to build on an ARCH we
# haven't explicitly defined support for. If all goes well, this will just work
# and the resulting lib will just have generic versions for anything unknown.
UniversalArchs := $(RC_ARCHS)


CFLAGS := -Wall -O3 -fomit-frame-pointer


FUNCTIONS := absvdi2 absvsi2 addvdi3 addvsi3 ashldi3 ashrdi3 \
             clzdi2 clzsi2 cmpdi2 ctzdi2 ctzsi2 \
             divdc3 divdi3 divsc3 ffsdi2 \
             fixdfdi fixsfdi fixunsdfdi fixunsdfsi fixunssfdi \
             fixunssfsi floatdidf floatdisf floatundidf floatundisf \
             gcc_personality_v0 lshrdi3 moddi3 muldc3 muldi3 \
             mulsc3 mulvdi3 mulvsi3 negdi2 negvdi2 negvsi2 \
             paritydi2 paritysi2 popcountdi2 popcountsi2 powidf2 \
             powisf2 subvdi3 subvsi3 ucmpdi2 udivdi3 \
             udivmoddi4 umoddi3 apple_versioning eprintf

FUNCTIONS.i386 := $(FUNCTIONS) \
                divxc3 fixunsxfdi fixunsxfsi fixxfdi floatdixf \
                floatundixf mulxc3 powixf2 clear_cache \
                enable_execute_stack
FUNCTIONS.ppc := $(FUNCTIONS) \
                divtc3 fixtfdi fixunstfdi floatditf floatunditf \
                gcc_qadd gcc_qdiv gcc_qmul gcc_qsub multc3 \
                powitf2 restFP saveFP trampoline_setup \
                clear_cache enable_execute_stack
FUNCTIONS.x86_64 := $(FUNCTIONS) \
                absvti2 addvti3 ashlti3 ashrti3 clzti2 cmpti2 \
                ctzti2 divti3 divxc3 ffsti2 fixdfti fixsfti \
                fixunsdfti fixunssfti fixunsxfdi fixunsxfsi \
                fixunsxfti fixxfdi fixxfti floatdixf floattidf \
                floattisf floattixf floatundixf floatuntidf \
                floatuntisf floatuntixf lshrti3 modti3 multi3 \
                mulvti3 mulxc3 negti2 negvti2 parityti2 \
                popcountti2 powixf2 subvti3 ucmpti2 udivmodti4 \
                udivti3 umodti3 clear_cache enable_execute_stack
FUNCTIONS.armv6 := $(FUNCTIONS) \
                adddf3vfp addsf3vfp bswapdi2 bswapsi2 divdf3vfp \
                divsf3vfp eqdf2vfp eqsf2vfp extendsfdf2vfp \
                fixdfsivfp fixsfsivfp fixunsdfsivfp fixunssfsivfp \
                floatsidfvfp floatsisfvfp floatunssidfvfp floatunssisfvfp \
                gedf2vfp gesf2vfp gtdf2vfp gtsf2vfp \
                ledf2vfp lesf2vfp ltdf2vfp ltsf2vfp \
                muldf3vfp mulsf3vfp \
                nedf2vfp negdf2vfp negsf2vfp nesf2vfp \
                subdf3vfp subsf3vfp truncdfsf2vfp unorddf2vfp unordsf2vfp \
                modsi3 umodsi3 udivsi3 divsi3 \
				switch8 switchu8 switch16 switch32 \
				restore_vfp_d8_d15_regs save_vfp_d8_d15_regs \
				sync_synchronize 

VISIBILITY_HIDDEN := 0
