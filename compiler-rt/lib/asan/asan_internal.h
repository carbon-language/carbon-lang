//===-- asan_internal.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan-private header which defines various general utilities.
//===----------------------------------------------------------------------===//
#ifndef ASAN_INTERNAL_H
#define ASAN_INTERNAL_H

#include "asan_flags.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_libc.h"

#if !defined(__linux__) && !defined(__APPLE__) && !defined(_WIN32)
# error "This operating system is not supported by AddressSanitizer"
#endif

#define ASAN_DEFAULT_FAILURE_EXITCODE 1

#if defined(__linux__)
# define ASAN_LINUX   1
#else
# define ASAN_LINUX   0
#endif

#if defined(__APPLE__)
# define ASAN_MAC     1
#else
# define ASAN_MAC     0
#endif

#if defined(_WIN32)
# define ASAN_WINDOWS 1
#else
# define ASAN_WINDOWS 0
#endif

#if defined(__ANDROID__) || defined(ANDROID)
# define ASAN_ANDROID 1
#else
# define ASAN_ANDROID 0
#endif


#define ASAN_POSIX (ASAN_LINUX || ASAN_MAC)

#if __has_feature(address_sanitizer)
# error "The AddressSanitizer run-time should not be"
        " instrumented by AddressSanitizer"
#endif

// Build-time configuration options.

// If set, asan will install its own SEGV signal handler.
#ifndef ASAN_NEEDS_SEGV
# if ASAN_ANDROID == 1
#  define ASAN_NEEDS_SEGV 0
# else
#  define ASAN_NEEDS_SEGV 1
# endif
#endif

// If set, asan will intercept C++ exception api call(s).
#ifndef ASAN_HAS_EXCEPTIONS
# define ASAN_HAS_EXCEPTIONS 1
#endif

// If set, asan uses the values of SHADOW_SCALE and SHADOW_OFFSET
// provided by the instrumented objects. Otherwise constants are used.
#ifndef ASAN_FLEXIBLE_MAPPING_AND_OFFSET
# define ASAN_FLEXIBLE_MAPPING_AND_OFFSET 0
#endif

// If set, values like allocator chunk size, as well as defaults for some flags
// will be changed towards less memory overhead.
#ifndef ASAN_LOW_MEMORY
# ifdef ASAN_ANDROID
#  define ASAN_LOW_MEMORY 1
# else
#  define ASAN_LOW_MEMORY 0
# endif
#endif

// All internal functions in asan reside inside the __asan namespace
// to avoid namespace collisions with the user programs.
// Seperate namespace also makes it simpler to distinguish the asan run-time
// functions from the instrumented user code in a profile.
namespace __asan {

class AsanThread;
using __sanitizer::StackTrace;

// asan_rtl.cc
void NORETURN ShowStatsAndAbort();

void ReplaceOperatorsNewAndDelete();
// asan_malloc_linux.cc / asan_malloc_mac.cc
void ReplaceSystemMalloc();

// asan_linux.cc / asan_mac.cc / asan_win.cc
void *AsanDoesNotSupportStaticLinkage();

void GetPcSpBp(void *context, uptr *pc, uptr *sp, uptr *bp);

void MaybeReexec();
bool AsanInterceptsSignal(int signum);
void SetAlternateSignalStack();
void UnsetAlternateSignalStack();
void InstallSignalHandlers();
void ClearShadowMemoryForContext(void *context);
void AsanPlatformThreadInit();

// Wrapper for TLS/TSD.
void AsanTSDInit(void (*destructor)(void *tsd));
void *AsanTSDGet();
void AsanTSDSet(void *tsd);

void AppendToErrorMessageBuffer(const char *buffer);

// asan_poisoning.cc
// Poisons the shadow memory for "size" bytes starting from "addr".
void PoisonShadow(uptr addr, uptr size, u8 value);
// Poisons the shadow memory for "redzone_size" bytes starting from
// "addr + size".
void PoisonShadowPartialRightRedzone(uptr addr,
                                     uptr size,
                                     uptr redzone_size,
                                     u8 value);

// Platfrom-specific options.
#ifdef __APPLE__
bool PlatformHasDifferentMemcpyAndMemmove();
# define PLATFORM_HAS_DIFFERENT_MEMCPY_AND_MEMMOVE \
    (PlatformHasDifferentMemcpyAndMemmove())
#else
# define PLATFORM_HAS_DIFFERENT_MEMCPY_AND_MEMMOVE true
#endif  // __APPLE__

extern int asan_inited;
// Used to avoid infinite recursion in __asan_init().
extern bool asan_init_is_running;
extern void (*death_callback)(void);

// These magic values are written to shadow for better error reporting.
const int kAsanHeapLeftRedzoneMagic = 0xfa;
const int kAsanHeapRightRedzoneMagic = 0xfb;
const int kAsanHeapFreeMagic = 0xfd;
const int kAsanStackLeftRedzoneMagic = 0xf1;
const int kAsanStackMidRedzoneMagic = 0xf2;
const int kAsanStackRightRedzoneMagic = 0xf3;
const int kAsanStackPartialRedzoneMagic = 0xf4;
const int kAsanStackAfterReturnMagic = 0xf5;
const int kAsanInitializationOrderMagic = 0xf6;
const int kAsanUserPoisonedMemoryMagic = 0xf7;
const int kAsanStackUseAfterScopeMagic = 0xf8;
const int kAsanGlobalRedzoneMagic = 0xf9;
const int kAsanInternalHeapMagic = 0xfe;

static const uptr kCurrentStackFrameMagic = 0x41B58AB3;
static const uptr kRetiredStackFrameMagic = 0x45E0360E;

}  // namespace __asan

#endif  // ASAN_INTERNAL_H
