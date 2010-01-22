# These are the functions which clang needs when it is targetting a previous
# version of the OS. The issue is that the backend may use functions which were
# not present in the libgcc that shipped on the platform. In such cases, we link
# with a version of the library which contains private_extern definitions of all
# the extra functions which might be referenced.

Description := Static runtime libraries for clang/Darwin.

Configs :=
UniversalArchs :=

# Configuration for targetting 10.4. We need a few functions missing from
# libgcc_s.10.4.dylib. We only build x86 slices since clang doesn't really
# support targetting PowerPC.
Configs += 10.4
UniversalArchs.10.4 := i386 x86_64

# Configuration for targetting armv6. We need a few additional functions which
# must be in the same linkage unit.
Configs += armv6
UniversalArchs.armv6 := armv6

CC := gcc

# Forcibly strip off any -arch, as that totally breaks our universal support.
override CC := $(subst -arch ,-arch_,$(CC))
override CC := $(patsubst -arch_%,,$(CC))

CFLAGS := -Wall -Werror -O3 -fomit-frame-pointer

FUNCTIONS.10.4 := eprintf floatundidf floatundisf floatundixf
FUNCTIONS.armv6 := switch16 switch32 switch8 switchu8 \
                   save_vfp_d8_d15_regs restore_vfp_d8_d15_regs

VISIBILITY_HIDDEN := 1
