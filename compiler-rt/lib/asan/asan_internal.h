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

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_libc.h"

#if !defined(__linux__) && !defined(__APPLE__) && !defined(_WIN32)
# error "This operating system is not supported by AddressSanitizer"
#endif

#if defined(_WIN32)
extern "C" void* _ReturnAddress(void);
# pragma intrinsic(_ReturnAddress)
#endif  // defined(_WIN32)

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
const char *AsanGetEnv(const char *name);
void AsanDumpProcessMap();

void *AsanMmapFixedNoReserve(uptr fixed_addr, uptr size);
void *AsanMmapFixedReserve(uptr fixed_addr, uptr size);
void *AsanMprotect(uptr fixed_addr, uptr size);
void *AsanMmapSomewhereOrDie(uptr size, const char *where);
void AsanUnmapOrDie(void *ptr, uptr size);

void AsanDisableCoreDumper();
void GetPcSpBp(void *context, uptr *pc, uptr *sp, uptr *bp);

bool AsanInterceptsSignal(int signum);
void SetAlternateSignalStack();
void UnsetAlternateSignalStack();
void InstallSignalHandlers();
uptr GetThreadSelf();
int AtomicInc(int *a);
u16 AtomicExchange(u16 *a, u16 new_val);

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
void Report(const char *format, ...);

// Don't use std::min and std::max, to minimize dependency on libstdc++.
template<class T> T Min(T a, T b) { return a < b ? a : b; }
template<class T> T Max(T a, T b) { return a > b ? a : b; }

void SortArray(uptr *array, uptr size);

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

extern uptr  FLAG_quarantine_size;
extern s64 FLAG_demangle;
extern bool    FLAG_symbolize;
extern s64 FLAG_v;
extern uptr  FLAG_redzone;
extern s64 FLAG_debug;
extern bool    FLAG_poison_shadow;
extern s64 FLAG_report_globals;
extern uptr  FLAG_malloc_context_size;
extern bool    FLAG_replace_str;
extern bool    FLAG_replace_intrin;
extern bool    FLAG_replace_cfallocator;
extern bool    FLAG_fast_unwind;
extern bool    FLAG_use_fake_stack;
extern uptr  FLAG_max_malloc_fill_size;
extern s64 FLAG_exitcode;
extern bool    FLAG_allow_user_poisoning;
extern s64 FLAG_sleep_before_dying;
extern bool    FLAG_handle_segv;
extern bool    FLAG_use_sigaltstack;
extern bool    FLAG_check_malloc_usable_size;
extern bool    FLAG_unmap_shadow_on_exit;
extern bool    FLAG_abort_on_error;

extern int asan_inited;
// Used to avoid infinite recursion in __asan_init().
extern bool asan_init_is_running;
extern void (*death_callback)(void);

enum LinkerInitialized { LINKER_INITIALIZED = 0 };

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
    Die(); \
  } \
} while (0)

#define RAW_CHECK(expr) RAW_CHECK_MSG(expr, #expr)

#define UNIMPLEMENTED() CHECK("unimplemented" && 0)

#define ASAN_ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))

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
