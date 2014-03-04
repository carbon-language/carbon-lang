Description := Static runtime libraries for clang/Linux.

###

CC := clang
Arch := unknown
Configs :=

# We don't currently have any general purpose way to target architectures other
# than the compiler defaults (because there is no generalized way to invoke
# cross compilers). For now, we just find the target architecture of the
# compiler and only define configurations we know that compiler can generate.
CompilerTargetTriple := $(shell \
	LANG=C $(CC) -v 2>&1 | grep 'Target:' | cut -d' ' -f2)
ifeq ($(CompilerTargetTriple),)
$(error "unable to infer compiler target triple for $(CC)")
endif

# Only define configs if we detected a linux target.
ifneq ($(findstring -linux-,$(CompilerTargetTriple)),)

# Define configs only if arch in triple is i386 or x86_64
CompilerTargetArch := $(firstword $(subst -, ,$(CompilerTargetTriple)))
ifeq ($(call contains,i386 x86_64,$(CompilerTargetArch)),true)

# TryCompile compiler source flags
# Returns exit code of running a compiler invocation.
TryCompile = \
  $(shell \
    cflags=""; \
    for flag in $(3); do \
      cflags="$$cflags $$flag"; \
    done; \
    $(1) $$cflags $(2) -o /dev/null > /dev/null 2> /dev/null ; \
    echo $$?)

test_source = $(ProjSrcRoot)/make/platform/clang_linux_test_input.c
ifeq ($(CompilerTargetArch),i386)
  SupportedArches := i386
  ifeq ($(call TryCompile,$(CC),$(test_source),-m64),0)
    SupportedArches += x86_64
  endif
else
  SupportedArches := x86_64
  ifeq ($(call TryCompile,$(CC),$(test_source),-m32),0)
    SupportedArches += i386
  endif
endif

# Build runtime libraries for i386.
ifeq ($(call contains,$(SupportedArches),i386),true)
Configs += full-i386 profile-i386 san-i386 asan-i386 ubsan-i386 ubsan_cxx-i386
Arch.full-i386 := i386
Arch.profile-i386 := i386
Arch.san-i386 := i386
Arch.asan-i386 := i386
Arch.ubsan-i386 := i386
Arch.ubsan_cxx-i386 := i386
endif

# Build runtime libraries for x86_64.
ifeq ($(call contains,$(SupportedArches),x86_64),true)
Configs += full-x86_64 profile-x86_64 san-x86_64 asan-x86_64 tsan-x86_64 \
           msan-x86_64 ubsan-x86_64 ubsan_cxx-x86_64 dfsan-x86_64 lsan-x86_64
Arch.full-x86_64 := x86_64
Arch.profile-x86_64 := x86_64
Arch.san-x86_64 := x86_64
Arch.asan-x86_64 := x86_64
Arch.tsan-x86_64 := x86_64
Arch.msan-x86_64 := x86_64
Arch.ubsan-x86_64 := x86_64
Arch.ubsan_cxx-x86_64 := x86_64
Arch.dfsan-x86_64 := x86_64
Arch.lsan-x86_64 := x86_64
endif

endif

ifneq ($(LLVM_ANDROID_TOOLCHAIN_DIR),)
Configs += asan-arm-android
Arch.asan-arm-android := arm-android
endif

endif

###

CFLAGS := -Wall -Werror -O3 -fomit-frame-pointer
SANITIZER_CFLAGS := -fPIE -fno-builtin -gline-tables-only

CFLAGS.full-i386 := $(CFLAGS) -m32
CFLAGS.full-x86_64 := $(CFLAGS) -m64
CFLAGS.profile-i386 := $(CFLAGS) -m32
CFLAGS.profile-x86_64 := $(CFLAGS) -m64
CFLAGS.san-i386 := $(CFLAGS) -m32 $(SANITIZER_CFLAGS) -fno-rtti
CFLAGS.san-x86_64 := $(CFLAGS) -m64 $(SANITIZER_CFLAGS) -fno-rtti
CFLAGS.asan-i386 := $(CFLAGS) -m32 $(SANITIZER_CFLAGS) -fno-rtti
CFLAGS.asan-x86_64 := $(CFLAGS) -m64 $(SANITIZER_CFLAGS) -fno-rtti
CFLAGS.tsan-x86_64 := $(CFLAGS) -m64 $(SANITIZER_CFLAGS) -fno-rtti
CFLAGS.msan-x86_64 := $(CFLAGS) -m64 $(SANITIZER_CFLAGS) -fno-rtti
CFLAGS.ubsan-i386 := $(CFLAGS) -m32 $(SANITIZER_CFLAGS) -fno-rtti
CFLAGS.ubsan-x86_64 := $(CFLAGS) -m64 $(SANITIZER_CFLAGS) -fno-rtti
CFLAGS.ubsan_cxx-i386 := $(CFLAGS) -m32 $(SANITIZER_CFLAGS)
CFLAGS.ubsan_cxx-x86_64 := $(CFLAGS) -m64 $(SANITIZER_CFLAGS)
CFLAGS.dfsan-x86_64 := $(CFLAGS) -m64 $(SANITIZER_CFLAGS)
CFLAGS.lsan-x86_64 := $(CFLAGS) -m64 $(SANITIZER_CFLAGS) -fno-rtti

SHARED_LIBRARY.asan-arm-android := 1
ANDROID_COMMON_FLAGS := -target arm-linux-androideabi \
	--sysroot=$(LLVM_ANDROID_TOOLCHAIN_DIR)/sysroot \
	-B$(LLVM_ANDROID_TOOLCHAIN_DIR)
CFLAGS.asan-arm-android := $(CFLAGS) -fPIC -fno-builtin \
	$(ANDROID_COMMON_FLAGS) -fno-rtti \
	-I$(ProjSrcRoot)/third_party/android/include
LDFLAGS.asan-arm-android := $(LDFLAGS) $(ANDROID_COMMON_FLAGS) -ldl -lm -llog \
	-lstdc++ -Wl,-soname=libclang_rt.asan-arm-android.so -Wl,-z,defs

# Use our stub SDK as the sysroot to support more portable building. For now we
# just do this for the core module, because the stub SDK doesn't have
# enough support to build the sanitizers or profile runtimes.
CFLAGS.full-i386 += --sysroot=$(ProjSrcRoot)/SDKs/linux
CFLAGS.full-x86_64 += --sysroot=$(ProjSrcRoot)/SDKs/linux

FUNCTIONS.full-i386 := $(CommonFunctions) $(ArchFunctions.i386)
FUNCTIONS.full-x86_64 := $(CommonFunctions) $(ArchFunctions.x86_64)
FUNCTIONS.profile-i386 := GCDAProfiling
FUNCTIONS.profile-x86_64 := GCDAProfiling
FUNCTIONS.san-i386 := $(SanitizerCommonFunctions)
FUNCTIONS.san-x86_64 := $(SanitizerCommonFunctions)
FUNCTIONS.asan-i386 := $(AsanFunctions) $(InterceptionFunctions) \
                                        $(SanitizerCommonFunctions)
FUNCTIONS.asan-x86_64 := $(AsanFunctions) $(InterceptionFunctions) \
                         $(SanitizerCommonFunctions) $(LsanCommonFunctions)
FUNCTIONS.asan-arm-android := $(AsanFunctions) $(InterceptionFunctions) \
                                          $(SanitizerCommonFunctions)
FUNCTIONS.tsan-x86_64 := $(TsanFunctions) $(InterceptionFunctions) \
                                          $(SanitizerCommonFunctions)
FUNCTIONS.msan-x86_64 := $(MsanFunctions) $(InterceptionFunctions) \
                                          $(SanitizerCommonFunctions)
FUNCTIONS.ubsan-i386 := $(UbsanFunctions)
FUNCTIONS.ubsan-x86_64 := $(UbsanFunctions)
FUNCTIONS.ubsan_cxx-i386 := $(UbsanCXXFunctions)
FUNCTIONS.ubsan_cxx-x86_64 := $(UbsanCXXFunctions)
FUNCTIONS.dfsan-x86_64 := $(DfsanFunctions) $(SanitizerCommonFunctions)
FUNCTIONS.lsan-x86_64 := $(LsanFunctions) $(InterceptionFunctions) \
                                          $(SanitizerCommonFunctions)

# Always use optimized variants.
OPTIMIZED := 1

# We don't need to use visibility hidden on Linux.
VISIBILITY_HIDDEN := 0

SHARED_LIBRARY_SUFFIX := so
