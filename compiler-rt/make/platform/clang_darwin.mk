# These are the functions which clang needs when it is targetting a previous
# version of the OS. The issue is that the backend may use functions which were
# not present in the libgcc that shipped on the platform. In such cases, we link
# with a version of the library which contains private_extern definitions of all
# the extra functions which might be referenced.

Description := Static runtime libraries for clang/Darwin.

Configs :=
UniversalArchs :=

# Configuration solely for providing access to an eprintf symbol, which may
# still be referenced from Darwin system headers. This symbol is only ever
# needed on i386.
Configs += eprintf
UniversalArchs.eprintf := i386

# Configuration for targetting 10.4. We need a few functions missing from
# libgcc_s.10.4.dylib. We only build x86 slices since clang doesn't really
# support targetting PowerPC.
Configs += 10.4
UniversalArchs.10.4 := i386 x86_64

# Configuration for targetting armv6. We need a few additional functions which
# must be in the same linkage unit.
Configs += armv6
UniversalArchs.armv6 := armv6

# Configuration for use with kernel/kexts.
Configs += cc_kext
UniversalArchs.cc_kext := armv6 armv7 i386 x86_64

###

CC := gcc

# Forcibly strip off any -arch, as that totally breaks our universal support.
override CC := $(subst -arch ,-arch_,$(CC))
override CC := $(patsubst -arch_%,,$(CC))

# Note that although we use -fno-builtin here, the backend may still synthesize
# calls to runtime functions. Unfortunately, we currently have no way to
# guarantee that we won't be creating a cycle in the runtime library, aside from
# explicit runtime testing.
CFLAGS := -Wall -Werror -O3 -fomit-frame-pointer -fno-builtin

FUNCTIONS.eprintf := eprintf
FUNCTIONS.10.4 := eprintf floatundidf floatundisf floatundixf
FUNCTIONS.armv6 := switch16 switch32 switch8 switchu8 \
                   save_vfp_d8_d15_regs restore_vfp_d8_d15_regs

CCKEXT_COMMON_FUNCTIONS := \
	absvdi2 \
	absvsi2 \
	addvdi3 \
	addvsi3 \
	ashldi3 \
	ashrdi3 \
	bswapdi2 \
	bswapsi2 \
	clear_cache \
	clzdi2 \
	clzsi2 \
	cmpdi2 \
	ctzdi2 \
	ctzsi2 \
	divdc3 \
	divdi3 \
	divsc3 \
	do_global_dtors \
	eprintf \
	ffsdi2 \
	fixdfdi \
	fixsfdi \
	fixunsdfdi \
	fixunsdfsi \
	fixunssfdi \
	fixunssfsi \
	floatdidf \
	floatdisf \
	floatundidf \
	floatundisf \
	gcc_bcmp \
	lshrdi3 \
	moddi3 \
	muldc3 \
	muldi3 \
	mulsc3 \
	mulvdi3 \
	mulvsi3 \
	negdi2 \
	negvdi2 \
	negvsi2 \
	paritydi2 \
	paritysi2 \
	popcountdi2 \
	popcountsi2 \
	powidf2 \
	powisf2 \
	subvdi3 \
	subvsi3 \
	ucmpdi2 \
	udiv_w_sdiv \
	udivdi3 \
	udivmoddi4 \
	umoddi3

CCKEXT_ARM_FUNCTIONS := $(CCKEXT_COMMON_FUNCTIONS) \
	adddf3 \
	addsf3 \
	aeabi_cdcmpeq \
	aeabi_cdrcmple \
	aeabi_cfcmpeq \
	aeabi_cfrcmple \
	aeabi_dcmpeq \
	aeabi_dcmpge \
	aeabi_dcmpgt \
	aeabi_dcmple \
	aeabi_dcmplt \
	aeabi_drsub \
	aeabi_fcmpeq \
	aeabi_fcmpge \
	aeabi_fcmpgt \
	aeabi_fcmple \
	aeabi_fcmplt \
	aeabi_frsub \
	aeabi_idivmod \
	aeabi_uidivmod \
	cmpdf2 \
	cmpsf2 \
	div0 \
	divdf3 \
	divsf3 \
	divsi3 \
	extendsfdf2 \
	ffssi2 \
	fixdfsi \
	fixsfsi \
	floatsidf \
	floatsisf \
	floatunsidf \
	floatunsisf \
	gtdf2 \
	gtsf2 \
	ltdf2 \
	ltsf2 \
	modsi3 \
	muldf3 \
	mulsf3 \
	negdf2 \
	negsf2 \
	subdf3 \
	subsf3 \
	switch16 \
	switch32 \
	switch8 \
	switchu8 \
	truncdfsf2 \
	udivsi3 \
	umodsi3 \
	unorddf2 \
	unordsf2

FUNCTIONS.cc_kext.armv6 := $(CCKEXT_ARM_FUNCTIONS)
FUNCTIONS.cc_kext.armv7 := $(CCKEXT_ARM_FUNCTIONS)

CFLAGS.cc_kext.armv6 := $(CFLAGS) -mthumb
CFLAGS.cc_kext.armv7 := $(CFLAGS) -mthumb

CCKEXT_X86_FUNCTIONS := $(CCKEXT_COMMON_FUNCTIONS) \
	divxc3 \
	fixunsxfdi \
	fixunsxfsi \
	fixxfdi \
	floatdixf \
	floatundixf \
	mulxc3 \
	powixf2

FUNCTIONS.cc_kext.i386 := $(CCKEXT_X86_FUNCTIONS) \
	ffssi2 \
	i686.get_pc_thunk.eax \
	i686.get_pc_thunk.ebp \
	i686.get_pc_thunk.ebx \
	i686.get_pc_thunk.ecx \
	i686.get_pc_thunk.edi \
	i686.get_pc_thunk.edx \
	i686.get_pc_thunk.esi

FUNCTIONS.cc_kext.x86_64 := $(CCKEXT_X86_FUNCTIONS) \
	absvti2 \
	addvti3 \
	ashlti3 \
	ashrti3 \
	clzti2 \
	cmpti2 \
	ctzti2 \
	divti3 \
	ffsti2 \
	fixdfti \
	fixsfti \
	fixunsdfti \
	fixunssfti \
	fixunsxfti \
	fixxfti \
	floattidf \
	floattisf \
	floattixf \
	floatuntidf \
	floatuntisf \
	floatuntixf \
	lshrti3 \
	modti3 \
	multi3 \
	mulvti3 \
	negti2 \
	negvti2 \
	parityti2 \
	popcountti2 \
	subvti3 \
	ucmpti2 \
	udivmodti4 \
	udivti3 \
	umodti3

# FIXME: Currently, compiler-rt is missing implementations for a number of the
# functions that need to go into libcc_kext.a. Filter them out for now.
CCKEXT_MISSING_FUNCTIONS := \
	adddf3 addsf3 cmpdf2 cmpsf2 div0 divdf3 divsf3 \
	extendsfdf2 ffssi2 fixdfsi fixsfsi floatsidf floatsisf \
	floatunsidf floatunsisf gtdf2 gtsf2 ltdf2 ltsf2 \
	muldf3 mulsf3 negdf2 negsf2 subdf3 subsf3 \
	truncdfsf2 udiv_w_sdiv unorddf2 unordsf2 bswapdi2 \
	bswapsi2 \
	gcc_bcmp \
	do_global_dtors \
	i686.get_pc_thunk.eax i686.get_pc_thunk.ebp i686.get_pc_thunk.ebx \
	i686.get_pc_thunk.ecx i686.get_pc_thunk.edi i686.get_pc_thunk.edx \
	i686.get_pc_thunk.esi \
	aeabi_cdcmpeq aeabi_cdrcmple aeabi_cfcmpeq aeabi_cfrcmple aeabi_dcmpeq \
	aeabi_dcmpge aeabi_dcmpgt aeabi_dcmple aeabi_dcmplt aeabi_drsub aeabi_fcmpeq \
	aeabi_fcmpge aeabi_fcmpgt aeabi_fcmple aeabi_fcmplt aeabi_frsub aeabi_idivmod \
	aeabi_uidivmod

FUNCTIONS.cc_kext.armv6 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.armv6))
FUNCTIONS.cc_kext.armv7 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.armv7))
FUNCTIONS.cc_kext.i386 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.i386))
FUNCTIONS.cc_kext.x86_64 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.x86_64))

KERNEL_USE.cc_kext := 1

VISIBILITY_HIDDEN := 1
