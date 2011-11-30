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

#include <stdint.h>  // for __WORDSIZE
#include <stdlib.h>  // for size_t
#include <unistd.h>  // for _exit

#ifdef ANDROID
#include <sys/atomics.h>
#endif

#ifdef ADDRESS_SANITIZER
# error "The AddressSanitizer run-time should not be"
        " instrumented by AddressSanitizer"
#endif

// All internal functions in asan reside inside the __asan namespace
// to avoid namespace collisions with the user programs.
// Seperate namespace also makes it simpler to distinguish the asan run-time
// functions from the instrumented user code in a profile.
namespace __asan {

class AsanThread;
struct AsanStackTrace;

//  asan_rtl.cc
void CheckFailed(const char *cond, const char *file, int line);
void ShowStatsAndAbort();

//  asan_globals.cc
bool DescribeAddrIfGlobal(uintptr_t addr);

//  asan_malloc_linux.cc / asan_malloc_mac.cc
void ReplaceSystemMalloc();

//  asan_linux.cc / asan_mac.cc
void *AsanDoesNotSupportStaticLinkage();
void *asan_mmap(void *addr, size_t length, int prot, int flags,
                int fd, uint64_t offset);
ssize_t asan_write(int fd, const void *buf, size_t count);

//  asan_printf.cc
void RawWrite(const char *buffer);
int SNPrint(char *buffer, size_t length, const char *format, ...);
void Printf(const char *format, ...);
void Report(const char *format, ...);

extern size_t FLAG_quarantine_size;
extern int    FLAG_demangle;
extern bool   FLAG_symbolize;
extern int    FLAG_v;
extern bool   FLAG_mt;
extern size_t FLAG_redzone;
extern int    FLAG_debug;
extern bool   FLAG_poison_shadow;
extern int    FLAG_report_globals;
extern size_t FLAG_malloc_context_size;
extern bool   FLAG_stats;
extern bool   FLAG_replace_str;
extern bool   FLAG_replace_intrin;
extern bool   FLAG_replace_cfallocator;
extern bool   FLAG_fast_unwind;
extern bool   FLAG_use_fake_stack;
extern size_t FLAG_max_malloc_fill_size;
extern int    FLAG_exitcode;
extern bool   FLAG_allow_user_poisoning;

extern int asan_inited;
// Used to avoid infinite recursion in __asan_init().
extern bool asan_init_is_running;

enum LinkerInitialized { LINKER_INITIALIZED = 0 };

#ifndef ASAN_DIE
#define ASAN_DIE _exit(FLAG_exitcode)
#endif  // ASAN_DIE

#define CHECK(cond) do { if (!(cond)) { \
  CheckFailed(#cond, __FILE__, __LINE__); \
}}while(0)

#define RAW_CHECK_MSG(expr, msg) do { \
  if (!(expr)) { \
    RawWrite(msg); \
    ASAN_DIE; \
  } \
} while (0)

#define RAW_CHECK(expr) RAW_CHECK_MSG(expr, #expr)

#define UNIMPLEMENTED() CHECK("unimplemented" && 0)

#define ASAN_ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))

const size_t kWordSize = __WORDSIZE / 8;
const size_t kWordSizeInBits = 8 * kWordSize;
const size_t kPageSizeBits = 12;
const size_t kPageSize = 1UL << kPageSizeBits;

#define GET_CALLER_PC() (uintptr_t)__builtin_return_address(0)
#define GET_CURRENT_FRAME() (uintptr_t)__builtin_frame_address(0)

#define GET_BP_PC_SP \
  uintptr_t bp = GET_CURRENT_FRAME();              \
  uintptr_t pc = GET_CALLER_PC();                  \
  uintptr_t local_stack;                           \
  uintptr_t sp = (uintptr_t)&local_stack;

// These magic values are written to shadow for better error reporting.
const int kAsanHeapLeftRedzoneMagic = 0xfa;
const int kAsanHeapRightRedzoneMagic = 0xfb;
const int kAsanHeapFreeMagic = 0xfd;
const int kAsanStackLeftRedzoneMagic = 0xf1;
const int kAsanStackMidRedzoneMagic = 0xf2;
const int kAsanStackRightRedzoneMagic = 0xf3;
const int kAsanStackPartialRedzoneMagic = 0xf4;
const int kAsanStackAfterReturnMagic = 0xf5;
const int kAsanUserPoisonedMemoryMagic = 0xf7;
const int kAsanGlobalRedzoneMagic = 0xf9;

static const uintptr_t kCurrentStackFrameMagic = 0x41B58AB3;
static const uintptr_t kRetiredStackFrameMagic = 0x45E0360E;

// Poison the shadow memory which corresponds to 'redzone_size' bytes
// of the original memory, where first 'size' bytes are addressable.
static inline void
PoisonShadowPartialRightRedzone(unsigned char *shadow,
                                uintptr_t size,
                                uintptr_t redzone_size,
                                uintptr_t shadow_granularity,
                                unsigned char magic) {
  for (uintptr_t i = 0; i < redzone_size;
       i+= shadow_granularity, shadow++) {
    if (i + shadow_granularity <= size) {
      *shadow = 0;  // fully addressable
    } else if (i >= size) {
      *shadow = (shadow_granularity == 128) ? 0xff : magic;  // unaddressable
    } else {
      *shadow = size - i;  // first size-i bytes are addressable
    }
  }
}

// -------------------------- Atomic ---------------- {{{1
static inline int AtomicInc(int *a) {
  if (!FLAG_mt) return ++(*a);
#ifdef ANDROID
  return __atomic_inc(a) + 1;
#else
  return __sync_add_and_fetch(a, 1);
#endif
}

static inline int AtomicDec(int *a) {
  if (!FLAG_mt) return --(*a);
#ifdef ANDROID
  return __atomic_dec(a) - 1;
#else
  return __sync_add_and_fetch(a, -1);
#endif
}

}  // namespace __asan

#endif  // ASAN_INTERNAL_H
