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

#include "sanitizer_common/sanitizer_defs.h"
#include "sanitizer_common/sanitizer_libc.h"

#if !defined(__linux__) && !defined(__APPLE__) && !defined(_WIN32)
# error "This operating system is not supported by AddressSanitizer"
#endif

#if defined(_WIN32)
# if defined(__clang__)
typedef int              intptr_t;
typedef unsigned int     uptr;
# endif

// There's no <stdint.h> in Visual Studio 9, so we have to define [u]int*_t.
typedef unsigned __int8  uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
typedef __int8           int8_t;
typedef __int16          int16_t;
typedef __int32          int32_t;
typedef __int64          int64_t;
typedef unsigned long    DWORD;  // NOLINT

extern "C" void* _ReturnAddress(void);
# pragma intrinsic(_ReturnAddress)

# define ALIAS(x)   // TODO(timurrrr): do we need this on Windows?
# define ALIGNED(x) __declspec(align(x))
# define NOINLINE __declspec(noinline)
# define NORETURN __declspec(noreturn)

# define ASAN_INTERFACE_ATTRIBUTE  // TODO(timurrrr): do we need this on Win?
#else  // defined(_WIN32)
# include <stdint.h>  // for __WORDSIZE

# define ALIAS(x) __attribute__((alias(x)))
# define ALIGNED(x) __attribute__((aligned(x)))
# define NOINLINE __attribute__((noinline))
# define NORETURN  __attribute__((noreturn))

# define ASAN_INTERFACE_ATTRIBUTE __attribute__((visibility("default")))
#endif  // defined(_WIN32)

// If __WORDSIZE was undefined by the platform, define it in terms of the
// compiler built-ins __LP64__ and _WIN64.
#ifndef __WORDSIZE
#if __LP64__ || defined(_WIN64)
#define __WORDSIZE 64
#else
#define __WORDSIZE 32
#endif
#endif

// Limits for integral types. We have to redefine it in case we don't
// have stdint.h (like in Visual Studio 9).
#if __WORDSIZE == 64
# define __INT64_C(c)  c ## L
# define __UINT64_C(c) c ## UL
#else
# define __INT64_C(c)  c ## LL
# define __UINT64_C(c) c ## ULL
#endif  // __WORDSIZE == 64
#undef INT32_MIN
#define INT32_MIN              (-2147483647-1)
#undef INT32_MAX
#define INT32_MAX              (2147483647)
#undef UINT32_MAX
#define UINT32_MAX             (4294967295U)
#undef INT64_MIN
#define INT64_MIN              (-__INT64_C(9223372036854775807)-1)
#undef INT64_MAX
#define INT64_MAX              (__INT64_C(9223372036854775807))
#undef UINT64_MAX
#define UINT64_MAX             (__UINT64_C(18446744073709551615))

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

#define ASAN_POSIX (ASAN_LINUX || ASAN_MAC)

#if !defined(__has_feature)
#define __has_feature(x) 0
#endif

#if __has_feature(address_sanitizer)
# error "The AddressSanitizer run-time should not be"
        " instrumented by AddressSanitizer"
#endif

// Build-time configuration options.

// If set, asan will install its own SEGV signal handler.
#ifndef ASAN_NEEDS_SEGV
# define ASAN_NEEDS_SEGV 1
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
# define ASAN_LOW_MEMORY 0
#endif

// All internal functions in asan reside inside the __asan namespace
// to avoid namespace collisions with the user programs.
// Seperate namespace also makes it simpler to distinguish the asan run-time
// functions from the instrumented user code in a profile.
namespace __asan {

class AsanThread;
struct AsanStackTrace;

// asan_rtl.cc
void NORETURN CheckFailed(const char *cond, const char *file, int line);
void NORETURN ShowStatsAndAbort();

// asan_globals.cc
bool DescribeAddrIfGlobal(uptr addr);

void ReplaceOperatorsNewAndDelete();
// asan_malloc_linux.cc / asan_malloc_mac.cc
void ReplaceSystemMalloc();

void OutOfMemoryMessageAndDie(const char *mem_type, uptr size);

// asan_linux.cc / asan_mac.cc / asan_win.cc
void *AsanDoesNotSupportStaticLinkage();
bool AsanShadowRangeIsAvailable();
int AsanOpenReadonly(const char* filename);
const char *AsanGetEnv(const char *name);
void AsanDumpProcessMap();

void *AsanMmapFixedNoReserve(uptr fixed_addr, uptr size);
void *AsanMmapFixedReserve(uptr fixed_addr, uptr size);
void *AsanMprotect(uptr fixed_addr, uptr size);
void *AsanMmapSomewhereOrDie(uptr size, const char *where);
void AsanUnmapOrDie(void *ptr, uptr size);

void AsanDisableCoreDumper();
void GetPcSpBp(void *context, uptr *pc, uptr *sp, uptr *bp);

uptr AsanRead(int fd, void *buf, uptr count);
uptr AsanWrite(int fd, const void *buf, uptr count);
int AsanClose(int fd);

bool AsanInterceptsSignal(int signum);
void SetAlternateSignalStack();
void UnsetAlternateSignalStack();
void InstallSignalHandlers();
int GetPid();
uptr GetThreadSelf();
int AtomicInc(int *a);
uint16_t AtomicExchange(uint16_t *a, uint16_t new_val);

// Wrapper for TLS/TSD.
void AsanTSDInit(void (*destructor)(void *tsd));
void *AsanTSDGet();
void AsanTSDSet(void *tsd);

// Opens the file 'file_name" and reads up to 'max_len' bytes.
// The resulting buffer is mmaped and stored in '*buff'.
// The size of the mmaped region is stored in '*buff_size',
// Returns the number of read bytes or 0 if file can not be opened.
uptr ReadFileToBuffer(const char *file_name, char **buff,
                        uptr *buff_size, uptr max_len);

// asan_printf.cc
void RawWrite(const char *buffer);
int SNPrintf(char *buffer, uptr length, const char *format, ...);
void Printf(const char *format, ...);
int SScanf(const char *str, const char *format, ...);
void Report(const char *format, ...);

// Don't use std::min and std::max, to minimize dependency on libstdc++.
template<class T> T Min(T a, T b) { return a < b ? a : b; }
template<class T> T Max(T a, T b) { return a > b ? a : b; }

void SortArray(uptr *array, uptr size);

// asan_poisoning.cc
// Poisons the shadow memory for "size" bytes starting from "addr".
void PoisonShadow(uptr addr, uptr size, uint8_t value);
// Poisons the shadow memory for "redzone_size" bytes starting from
// "addr + size".
void PoisonShadowPartialRightRedzone(uptr addr,
                                     uptr size,
                                     uptr redzone_size,
                                     uint8_t value);

// Platfrom-specific options.
#ifdef __APPLE__
bool PlatformHasDifferentMemcpyAndMemmove();
# define PLATFORM_HAS_DIFFERENT_MEMCPY_AND_MEMMOVE \
    (PlatformHasDifferentMemcpyAndMemmove())
#else
# define PLATFORM_HAS_DIFFERENT_MEMCPY_AND_MEMMOVE true
#endif  // __APPLE__

extern uptr  FLAG_quarantine_size;
extern int64_t FLAG_demangle;
extern bool    FLAG_symbolize;
extern int64_t FLAG_v;
extern uptr  FLAG_redzone;
extern int64_t FLAG_debug;
extern bool    FLAG_poison_shadow;
extern int64_t FLAG_report_globals;
extern uptr  FLAG_malloc_context_size;
extern bool    FLAG_replace_str;
extern bool    FLAG_replace_intrin;
extern bool    FLAG_replace_cfallocator;
extern bool    FLAG_fast_unwind;
extern bool    FLAG_use_fake_stack;
extern uptr  FLAG_max_malloc_fill_size;
extern int64_t FLAG_exitcode;
extern bool    FLAG_allow_user_poisoning;
extern int64_t FLAG_sleep_before_dying;
extern bool    FLAG_handle_segv;
extern bool    FLAG_use_sigaltstack;
extern bool    FLAG_check_malloc_usable_size;

extern int asan_inited;
// Used to avoid infinite recursion in __asan_init().
extern bool asan_init_is_running;

enum LinkerInitialized { LINKER_INITIALIZED = 0 };

void NORETURN AsanDie();
void SleepForSeconds(int seconds);
void NORETURN Exit(int exitcode);
void NORETURN Abort();
int Atexit(void (*function)(void));

#define CHECK(cond) do { if (!(cond)) { \
  CheckFailed(#cond, __FILE__, __LINE__); \
}}while(0)

#define RAW_CHECK_MSG(expr, msg) do { \
  if (!(expr)) { \
    RawWrite(msg); \
    AsanDie(); \
  } \
} while (0)

#define RAW_CHECK(expr) RAW_CHECK_MSG(expr, #expr)

#define UNIMPLEMENTED() CHECK("unimplemented" && 0)

#define ASAN_ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))

const uptr kWordSize = __WORDSIZE / 8;
const uptr kWordSizeInBits = 8 * kWordSize;
const uptr kPageSizeBits = 12;
const uptr kPageSize = 1UL << kPageSizeBits;

#if !defined(_WIN32) || defined(__clang__)
# define GET_CALLER_PC() (uptr)__builtin_return_address(0)
# define GET_CURRENT_FRAME() (uptr)__builtin_frame_address(0)
#else
# define GET_CALLER_PC() (uptr)_ReturnAddress()
// CaptureStackBackTrace doesn't need to know BP on Windows.
// FIXME: This macro is still used when printing error reports though it's not
// clear if the BP value is needed in the ASan reports on Windows.
# define GET_CURRENT_FRAME() (uptr)0xDEADBEEF
#endif

#ifndef _WIN32
const uptr kMmapGranularity = kPageSize;
# define THREAD_CALLING_CONV
typedef void* thread_return_t;
#else
const uptr kMmapGranularity = 1UL << 16;
# define THREAD_CALLING_CONV __stdcall
typedef DWORD thread_return_t;

# ifndef ASAN_USE_EXTERNAL_SYMBOLIZER
#  define ASAN_USE_EXTERNAL_SYMBOLIZER __asan_WinSymbolize
bool __asan_WinSymbolize(const void *addr, char *out_buffer, int buffer_size);
# endif
#endif

typedef thread_return_t (THREAD_CALLING_CONV *thread_callback_t)(void* arg);

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
const int kAsanInternalHeapMagic = 0xfe;

static const uptr kCurrentStackFrameMagic = 0x41B58AB3;
static const uptr kRetiredStackFrameMagic = 0x45E0360E;

// --------------------------- Bit twiddling ------- {{{1
inline bool IsPowerOfTwo(uptr x) {
  return (x & (x - 1)) == 0;
}

inline uptr RoundUpTo(uptr size, uptr boundary) {
  CHECK(IsPowerOfTwo(boundary));
  return (size + boundary - 1) & ~(boundary - 1);
}

// -------------------------- LowLevelAllocator ----- {{{1
// A simple low-level memory allocator for internal use.
class LowLevelAllocator {
 public:
  explicit LowLevelAllocator(LinkerInitialized) {}
  // 'size' must be a power of two.
  // Requires an external lock.
  void *Allocate(uptr size);
 private:
  char *allocated_end_;
  char *allocated_current_;
};

}  // namespace __asan

#endif  // ASAN_INTERNAL_H
