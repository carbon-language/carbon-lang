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

#if !defined(__linux__) && !defined(__APPLE__) && !defined(_WIN32)
# error "This operating system is not supported by AddressSanitizer"
#endif

#include <stddef.h>  // for size_t, uintptr_t, etc.

#if defined(_WIN32)
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

#if defined(__has_feature) && __has_feature(address_sanitizer)
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
bool DescribeAddrIfGlobal(uintptr_t addr);

// asan_malloc_linux.cc / asan_malloc_mac.cc
void ReplaceSystemMalloc();

void OutOfMemoryMessageAndDie(const char *mem_type, size_t size);

// asan_linux.cc / asan_mac.cc / asan_win.cc
void *AsanDoesNotSupportStaticLinkage();
bool AsanShadowRangeIsAvailable();
int AsanOpenReadonly(const char* filename);
const char *AsanGetEnv(const char *name);
void AsanDumpProcessMap();

void *AsanMmapFixedNoReserve(uintptr_t fixed_addr, size_t size);
void *AsanMmapFixedReserve(uintptr_t fixed_addr, size_t size);
void *AsanMprotect(uintptr_t fixed_addr, size_t size);
void *AsanMmapSomewhereOrDie(size_t size, const char *where);
void AsanUnmapOrDie(void *ptr, size_t size);

void AsanDisableCoreDumper();
void GetPcSpBp(void *context, uintptr_t *pc, uintptr_t *sp, uintptr_t *bp);

size_t AsanRead(int fd, void *buf, size_t count);
size_t AsanWrite(int fd, const void *buf, size_t count);
int AsanClose(int fd);

bool AsanInterceptsSignal(int signum);
void InstallSignalHandlers();
int GetPid();
uintptr_t GetThreadSelf();
int AtomicInc(int *a);

// Wrapper for TLS/TSD.
void AsanTSDInit(void (*destructor)(void *tsd));
void *AsanTSDGet();
void AsanTSDSet(void *tsd);

// Opens the file 'file_name" and reads up to 'max_len' bytes.
// The resulting buffer is mmaped and stored in '*buff'.
// The size of the mmaped region is stored in '*buff_size',
// Returns the number of read bytes or 0 if file can not be opened.
size_t ReadFileToBuffer(const char *file_name, char **buff,
                        size_t *buff_size, size_t max_len);

// asan_printf.cc
void RawWrite(const char *buffer);
int SNPrintf(char *buffer, size_t length, const char *format, ...);
void Printf(const char *format, ...);
int SScanf(const char *str, const char *format, ...);
void Report(const char *format, ...);

// Don't use std::min and std::max, to minimize dependency on libstdc++.
template<class T> T Min(T a, T b) { return a < b ? a : b; }
template<class T> T Max(T a, T b) { return a > b ? a : b; }

void SortArray(uintptr_t *array, size_t size);

// asan_poisoning.cc
// Poisons the shadow memory for "size" bytes starting from "addr".
void PoisonShadow(uintptr_t addr, size_t size, uint8_t value);
// Poisons the shadow memory for "redzone_size" bytes starting from
// "addr + size".
void PoisonShadowPartialRightRedzone(uintptr_t addr,
                                     uintptr_t size,
                                     uintptr_t redzone_size,
                                     uint8_t value);

// Platfrom-specific options.
#ifdef __APPLE__
bool PlatformHasDifferentMemcpyAndMemmove();
# define PLATFORM_HAS_DIFFERENT_MEMCPY_AND_MEMMOVE \
    (PlatformHasDifferentMemcpyAndMemmove())
#else
# define PLATFORM_HAS_DIFFERENT_MEMCPY_AND_MEMMOVE true
#endif  // __APPLE__

extern size_t FLAG_quarantine_size;
extern int    FLAG_demangle;
extern bool   FLAG_symbolize;
extern int    FLAG_v;
extern size_t FLAG_redzone;
extern int    FLAG_debug;
extern bool   FLAG_poison_shadow;
extern int    FLAG_report_globals;
extern size_t FLAG_malloc_context_size;
extern bool   FLAG_replace_str;
extern bool   FLAG_replace_intrin;
extern bool   FLAG_replace_cfallocator;
extern bool   FLAG_fast_unwind;
extern bool   FLAG_use_fake_stack;
extern size_t FLAG_max_malloc_fill_size;
extern int    FLAG_exitcode;
extern bool   FLAG_allow_user_poisoning;
extern int    FLAG_sleep_before_dying;
extern bool   FLAG_handle_segv;

extern int asan_inited;
// Used to avoid infinite recursion in __asan_init().
extern bool asan_init_is_running;

enum LinkerInitialized { LINKER_INITIALIZED = 0 };

void NORETURN AsanDie();
void SleepForSeconds(int seconds);
void NORETURN Exit(int exitcode);
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

const size_t kWordSize = __WORDSIZE / 8;
const size_t kWordSizeInBits = 8 * kWordSize;
const size_t kPageSizeBits = 12;
const size_t kPageSize = 1UL << kPageSizeBits;

#ifndef _WIN32
const size_t kMmapGranularity = kPageSize;
# define GET_CALLER_PC() (uintptr_t)__builtin_return_address(0)
# define GET_CURRENT_FRAME() (uintptr_t)__builtin_frame_address(0)
# define THREAD_CALLING_CONV
typedef void* thread_return_t;
#else
const size_t kMmapGranularity = 1UL << 16;
# define GET_CALLER_PC() (uintptr_t)_ReturnAddress()
// CaptureStackBackTrace doesn't need to know BP on Windows.
// FIXME: This macro is still used when printing error reports though it's not
// clear if the BP value is needed in the ASan reports on Windows.
# define GET_CURRENT_FRAME() (uintptr_t)0xDEADBEEF
# define THREAD_CALLING_CONV __stdcall
typedef DWORD thread_return_t;

# ifndef ASAN_USE_EXTERNAL_SYMBOLIZER
#  define ASAN_USE_EXTERNAL_SYMBOLIZER __asan::WinSymbolize
bool WinSymbolize(const void *addr, char *out_buffer, int buffer_size);
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

static const uintptr_t kCurrentStackFrameMagic = 0x41B58AB3;
static const uintptr_t kRetiredStackFrameMagic = 0x45E0360E;

// --------------------------- Bit twiddling ------- {{{1
inline bool IsPowerOfTwo(size_t x) {
  return (x & (x - 1)) == 0;
}

inline size_t RoundUpTo(size_t size, size_t boundary) {
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
  void *Allocate(size_t size);
 private:
  char *allocated_end_;
  char *allocated_current_;
};

}  // namespace __asan

#endif  // ASAN_INTERNAL_H
