//===-- asan_rtl.cc -------------------------------------------------------===//
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
// Main file of the ASan run-time library.
//===----------------------------------------------------------------------===//
#include "asan_allocator.h"
#include "asan_interceptors.h"
#include "asan_interface.h"
#include "asan_internal.h"
#include "asan_lock.h"
#include "asan_mapping.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"

namespace __sanitizer {
using namespace __asan;

void Die() {
  static atomic_uint32_t num_calls;
  if (atomic_fetch_add(&num_calls, 1, memory_order_relaxed) != 0) {
    // Don't die twice - run a busy loop.
    while (1) { }
  }
  if (flags()->sleep_before_dying) {
    Report("Sleeping for %zd second(s)\n", flags()->sleep_before_dying);
    SleepForSeconds(flags()->sleep_before_dying);
  }
  if (flags()->unmap_shadow_on_exit)
    UnmapOrDie((void*)kLowShadowBeg, kHighShadowEnd - kLowShadowBeg);
  if (death_callback)
    death_callback();
  if (flags()->abort_on_error)
    Abort();
  Exit(flags()->exitcode);
}

void CheckFailed(const char *file, int line, const char *cond, u64 v1, u64 v2) {
  AsanReport("AddressSanitizer CHECK failed: %s:%d \"%s\" (0x%zx, 0x%zx)\n",
             file, line, cond, (uptr)v1, (uptr)v2);
  PRINT_CURRENT_STACK();
  ShowStatsAndAbort();
}

}  // namespace __sanitizer

namespace __asan {

// -------------------------- Flags ------------------------- {{{1
static const int kMallocContextSize = 30;

static Flags asan_flags;

Flags *flags() {
  return &asan_flags;
}

static void ParseFlagsFromString(Flags *f, const char *str) {
  ParseFlag(str, &f->quarantine_size, "quarantine_size");
  ParseFlag(str, &f->symbolize, "symbolize");
  ParseFlag(str, &f->verbosity, "verbosity");
  ParseFlag(str, &f->redzone, "redzone");
  CHECK(f->redzone >= 16);
  CHECK(IsPowerOfTwo(f->redzone));

  ParseFlag(str, &f->debug, "debug");
  ParseFlag(str, &f->report_globals, "report_globals");
  ParseFlag(str, &f->malloc_context_size, "malloc_context_size");
  CHECK(f->malloc_context_size <= kMallocContextSize);

  ParseFlag(str, &f->replace_str, "replace_str");
  ParseFlag(str, &f->replace_intrin, "replace_intrin");
  ParseFlag(str, &f->replace_cfallocator, "replace_cfallocator");
  ParseFlag(str, &f->mac_ignore_invalid_free, "mac_ignore_invalid_free");
  ParseFlag(str, &f->use_fake_stack, "use_fake_stack");
  ParseFlag(str, &f->max_malloc_fill_size, "max_malloc_fill_size");
  ParseFlag(str, &f->exitcode, "exitcode");
  ParseFlag(str, &f->allow_user_poisoning, "allow_user_poisoning");
  ParseFlag(str, &f->sleep_before_dying, "sleep_before_dying");
  ParseFlag(str, &f->handle_segv, "handle_segv");
  ParseFlag(str, &f->use_sigaltstack, "use_sigaltstack");
  ParseFlag(str, &f->check_malloc_usable_size, "check_malloc_usable_size");
  ParseFlag(str, &f->unmap_shadow_on_exit, "unmap_shadow_on_exit");
  ParseFlag(str, &f->abort_on_error, "abort_on_error");
  ParseFlag(str, &f->atexit, "atexit");
  ParseFlag(str, &f->disable_core, "disable_core");
}

void InitializeFlags(Flags *f, const char *env) {
  internal_memset(f, 0, sizeof(*f));

  f->quarantine_size = (ASAN_LOW_MEMORY) ? 1UL << 24 : 1UL << 28;
  f->symbolize = false;
  f->verbosity = 0;
  f->redzone = (ASAN_LOW_MEMORY) ? 64 : 128;
  f->debug = false;
  f->report_globals = 1;
  f->malloc_context_size = kMallocContextSize;
  f->replace_str = true;
  f->replace_intrin = true;
  f->replace_cfallocator = true;
  f->mac_ignore_invalid_free = false;
  f->use_fake_stack = true;
  f->max_malloc_fill_size = 0;
  f->exitcode = ASAN_DEFAULT_FAILURE_EXITCODE;
  f->allow_user_poisoning = true;
  f->sleep_before_dying = 0;
  f->handle_segv = ASAN_NEEDS_SEGV;
  f->use_sigaltstack = false;
  f->check_malloc_usable_size = true;
  f->unmap_shadow_on_exit = false;
  f->abort_on_error = false;
  f->atexit = false;
  f->disable_core = (__WORDSIZE == 64);

  // Override from user-specified string.
#if !defined(_WIN32)
  if (__asan_default_options) {
    ParseFlagsFromString(f, __asan_default_options);
    if (flags()->verbosity) {
      Report("Using the defaults from __asan_default_options: %s\n",
             __asan_default_options);
    }
  }
#endif

  // Override from command line.
  ParseFlagsFromString(f, env);
}

// -------------------------- Globals --------------------- {{{1
int asan_inited;
bool asan_init_is_running;
void (*death_callback)(void);
static void (*error_report_callback)(const char*);
char *error_message_buffer = 0;
uptr error_message_buffer_pos = 0;
uptr error_message_buffer_size = 0;

// -------------------------- Misc ---------------- {{{1
void ShowStatsAndAbort() {
  __asan_print_accumulated_stats();
  Die();
}

static void PrintBytes(const char *before, uptr *a) {
  u8 *bytes = (u8*)a;
  uptr byte_num = (__WORDSIZE) / 8;
  AsanPrintf("%s%p:", before, (void*)a);
  for (uptr i = 0; i < byte_num; i++) {
    AsanPrintf(" %x%x", bytes[i] >> 4, bytes[i] & 15);
  }
  AsanPrintf("\n");
}

void AppendToErrorMessageBuffer(const char *buffer) {
  if (error_message_buffer) {
    uptr length = internal_strlen(buffer);
    CHECK_GE(error_message_buffer_size, error_message_buffer_pos);
    uptr remaining = error_message_buffer_size - error_message_buffer_pos;
    internal_strncpy(error_message_buffer + error_message_buffer_pos,
                     buffer, remaining);
    error_message_buffer[error_message_buffer_size - 1] = '\0';
    // FIXME: reallocate the buffer instead of truncating the message.
    error_message_buffer_pos += remaining > length ? length : remaining;
  }
}

// ---------------------- mmap -------------------- {{{1
// Reserve memory range [beg, end].
static void ReserveShadowMemoryRange(uptr beg, uptr end) {
  CHECK((beg % kPageSize) == 0);
  CHECK(((end + 1) % kPageSize) == 0);
  uptr size = end - beg + 1;
  void *res = MmapFixedNoReserve(beg, size);
  CHECK(res == (void*)beg && "ReserveShadowMemoryRange failed");
}

// ---------------------- LowLevelAllocator ------------- {{{1
void *LowLevelAllocator::Allocate(uptr size) {
  CHECK((size & (size - 1)) == 0 && "size must be a power of two");
  if (allocated_end_ - allocated_current_ < (sptr)size) {
    uptr size_to_allocate = Max(size, kPageSize);
    allocated_current_ =
        (char*)MmapOrDie(size_to_allocate, __FUNCTION__);
    allocated_end_ = allocated_current_ + size_to_allocate;
    PoisonShadow((uptr)allocated_current_, size_to_allocate,
                 kAsanInternalHeapMagic);
  }
  CHECK(allocated_end_ - allocated_current_ >= (sptr)size);
  void *res = allocated_current_;
  allocated_current_ += size;
  return res;
}

// ---------------------- DescribeAddress -------------------- {{{1
static bool DescribeStackAddress(uptr addr, uptr access_size) {
  AsanThread *t = asanThreadRegistry().FindThreadByStackAddress(addr);
  if (!t) return false;
  const sptr kBufSize = 4095;
  char buf[kBufSize];
  uptr offset = 0;
  const char *frame_descr = t->GetFrameNameByAddr(addr, &offset);
  // This string is created by the compiler and has the following form:
  // "FunctioName n alloc_1 alloc_2 ... alloc_n"
  // where alloc_i looks like "offset size len ObjectName ".
  CHECK(frame_descr);
  // Report the function name and the offset.
  const char *name_end = internal_strchr(frame_descr, ' ');
  CHECK(name_end);
  buf[0] = 0;
  internal_strncat(buf, frame_descr,
                   Min(kBufSize,
                       static_cast<sptr>(name_end - frame_descr)));
  AsanPrintf("Address %p is located at offset %zu "
             "in frame <%s> of T%d's stack:\n",
             (void*)addr, offset, buf, t->tid());
  // Report the number of stack objects.
  char *p;
  uptr n_objects = internal_simple_strtoll(name_end, &p, 10);
  CHECK(n_objects > 0);
  AsanPrintf("  This frame has %zu object(s):\n", n_objects);
  // Report all objects in this frame.
  for (uptr i = 0; i < n_objects; i++) {
    uptr beg, size;
    sptr len;
    beg  = internal_simple_strtoll(p, &p, 10);
    size = internal_simple_strtoll(p, &p, 10);
    len  = internal_simple_strtoll(p, &p, 10);
    if (beg <= 0 || size <= 0 || len < 0 || *p != ' ') {
      AsanPrintf("AddressSanitizer can't parse the stack frame "
                 "descriptor: |%s|\n", frame_descr);
      break;
    }
    p++;
    buf[0] = 0;
    internal_strncat(buf, p, Min(kBufSize, len));
    p += len;
    AsanPrintf("    [%zu, %zu) '%s'\n", beg, beg + size, buf);
  }
  AsanPrintf("HINT: this may be a false positive if your program uses "
             "some custom stack unwind mechanism\n"
             "      (longjmp and C++ exceptions *are* supported)\n");
  t->summary()->Announce();
  return true;
}

static bool DescribeAddrIfShadow(uptr addr) {
  if (AddrIsInMem(addr))
    return false;
  static const char kAddrInShadowReport[] =
      "Address %p is located in the %s.\n";
  if (AddrIsInShadowGap(addr)) {
    AsanPrintf(kAddrInShadowReport, addr, "shadow gap area");
    return true;
  }
  if (AddrIsInHighShadow(addr)) {
    AsanPrintf(kAddrInShadowReport, addr, "high shadow area");
    return true;
  }
  if (AddrIsInLowShadow(addr)) {
    AsanPrintf(kAddrInShadowReport, addr, "low shadow area");
    return true;
  }

  CHECK(0);  // Unreachable.
  return false;
}

static NOINLINE void DescribeAddress(uptr addr, uptr access_size) {
  // Check if this is shadow or shadow gap.
  if (DescribeAddrIfShadow(addr))
    return;

  CHECK(AddrIsInMem(addr));

  // Check if this is a global.
  if (DescribeAddrIfGlobal(addr))
    return;

  if (DescribeStackAddress(addr, access_size))
    return;

  // finally, check if this is a heap.
  DescribeHeapAddress(addr, access_size);
}

// -------------------------- Run-time entry ------------------- {{{1
// exported functions
#define ASAN_REPORT_ERROR(type, is_write, size)                     \
extern "C" NOINLINE INTERFACE_ATTRIBUTE                        \
void __asan_report_ ## type ## size(uptr addr);                \
void __asan_report_ ## type ## size(uptr addr) {               \
  GET_CALLER_PC_BP_SP;                                              \
  __asan_report_error(pc, bp, sp, addr, is_write, size);            \
}

ASAN_REPORT_ERROR(load, false, 1)
ASAN_REPORT_ERROR(load, false, 2)
ASAN_REPORT_ERROR(load, false, 4)
ASAN_REPORT_ERROR(load, false, 8)
ASAN_REPORT_ERROR(load, false, 16)
ASAN_REPORT_ERROR(store, true, 1)
ASAN_REPORT_ERROR(store, true, 2)
ASAN_REPORT_ERROR(store, true, 4)
ASAN_REPORT_ERROR(store, true, 8)
ASAN_REPORT_ERROR(store, true, 16)

// Force the linker to keep the symbols for various ASan interface functions.
// We want to keep those in the executable in order to let the instrumented
// dynamic libraries access the symbol even if it is not used by the executable
// itself. This should help if the build system is removing dead code at link
// time.
static NOINLINE void force_interface_symbols() {
  volatile int fake_condition = 0;  // prevent dead condition elimination.
  if (fake_condition) {
    __asan_report_load1(0);
    __asan_report_load2(0);
    __asan_report_load4(0);
    __asan_report_load8(0);
    __asan_report_load16(0);
    __asan_report_store1(0);
    __asan_report_store2(0);
    __asan_report_store4(0);
    __asan_report_store8(0);
    __asan_report_store16(0);
    __asan_register_global(0, 0, 0);
    __asan_register_globals(0, 0);
    __asan_unregister_globals(0, 0);
    __asan_set_death_callback(0);
    __asan_set_error_report_callback(0);
    __asan_handle_no_return();
  }
}

// -------------------------- Init ------------------- {{{1
static void asan_atexit() {
  AsanPrintf("AddressSanitizer exit stats:\n");
  __asan_print_accumulated_stats();
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

int __asan_set_error_exit_code(int exit_code) {
  int old = flags()->exitcode;
  flags()->exitcode = exit_code;
  return old;
}

void NOINLINE __asan_handle_no_return() {
  int local_stack;
  AsanThread *curr_thread = asanThreadRegistry().GetCurrent();
  CHECK(curr_thread);
  uptr top = curr_thread->stack_top();
  uptr bottom = ((uptr)&local_stack - kPageSize) & ~(kPageSize-1);
  PoisonShadow(bottom, top - bottom, 0);
}

void NOINLINE __asan_set_death_callback(void (*callback)(void)) {
  death_callback = callback;
}

void NOINLINE __asan_set_error_report_callback(void (*callback)(const char*)) {
  error_report_callback = callback;
  if (callback) {
    error_message_buffer_size = 1 << 16;
    error_message_buffer =
        (char*)MmapOrDie(error_message_buffer_size, __FUNCTION__);
    error_message_buffer_pos = 0;
  }
}

void __asan_report_error(uptr pc, uptr bp, uptr sp,
                         uptr addr, bool is_write, uptr access_size) {
  // Do not print more than one report, otherwise they will mix up.
  static atomic_uint32_t num_calls;
  if (atomic_fetch_add(&num_calls, 1, memory_order_relaxed) != 0) return;

  AsanPrintf("===================================================="
             "=============\n");
  const char *bug_descr = "unknown-crash";
  if (AddrIsInMem(addr)) {
    u8 *shadow_addr = (u8*)MemToShadow(addr);
    // If we are accessing 16 bytes, look at the second shadow byte.
    if (*shadow_addr == 0 && access_size > SHADOW_GRANULARITY)
      shadow_addr++;
    // If we are in the partial right redzone, look at the next shadow byte.
    if (*shadow_addr > 0 && *shadow_addr < 128)
      shadow_addr++;
    switch (*shadow_addr) {
      case kAsanHeapLeftRedzoneMagic:
      case kAsanHeapRightRedzoneMagic:
        bug_descr = "heap-buffer-overflow";
        break;
      case kAsanHeapFreeMagic:
        bug_descr = "heap-use-after-free";
        break;
      case kAsanStackLeftRedzoneMagic:
        bug_descr = "stack-buffer-underflow";
        break;
      case kAsanStackMidRedzoneMagic:
      case kAsanStackRightRedzoneMagic:
      case kAsanStackPartialRedzoneMagic:
        bug_descr = "stack-buffer-overflow";
        break;
      case kAsanStackAfterReturnMagic:
        bug_descr = "stack-use-after-return";
        break;
      case kAsanUserPoisonedMemoryMagic:
        bug_descr = "use-after-poison";
        break;
      case kAsanGlobalRedzoneMagic:
        bug_descr = "global-buffer-overflow";
        break;
    }
  }

  AsanThread *curr_thread = asanThreadRegistry().GetCurrent();
  u32 curr_tid = asanThreadRegistry().GetCurrentTidOrInvalid();

  if (curr_thread) {
    // We started reporting an error message. Stop using the fake stack
    // in case we will call an instrumented function from a symbolizer.
    curr_thread->fake_stack().StopUsingFakeStack();
  }

  AsanReport("ERROR: AddressSanitizer %s on address "
             "%p at pc 0x%zx bp 0x%zx sp 0x%zx\n",
             bug_descr, (void*)addr, pc, bp, sp);

  AsanPrintf("%s of size %zu at %p thread T%d\n",
             access_size ? (is_write ? "WRITE" : "READ") : "ACCESS",
             access_size, (void*)addr, curr_tid);

  if (flags()->debug) {
    PrintBytes("PC: ", (uptr*)pc);
  }

  GET_STACK_TRACE_WITH_PC_AND_BP(kStackTraceMax, pc, bp);
  stack.PrintStack();

  DescribeAddress(addr, access_size);

  if (AddrIsInMem(addr)) {
    uptr shadow_addr = MemToShadow(addr);
    AsanReport("ABORTING\n");
    __asan_print_accumulated_stats();
    AsanPrintf("Shadow byte and word:\n");
    AsanPrintf("  %p: %x\n", (void*)shadow_addr, *(unsigned char*)shadow_addr);
    uptr aligned_shadow = shadow_addr & ~(kWordSize - 1);
    PrintBytes("  ", (uptr*)(aligned_shadow));
    AsanPrintf("More shadow bytes:\n");
    PrintBytes("  ", (uptr*)(aligned_shadow-4*kWordSize));
    PrintBytes("  ", (uptr*)(aligned_shadow-3*kWordSize));
    PrintBytes("  ", (uptr*)(aligned_shadow-2*kWordSize));
    PrintBytes("  ", (uptr*)(aligned_shadow-1*kWordSize));
    PrintBytes("=>", (uptr*)(aligned_shadow+0*kWordSize));
    PrintBytes("  ", (uptr*)(aligned_shadow+1*kWordSize));
    PrintBytes("  ", (uptr*)(aligned_shadow+2*kWordSize));
    PrintBytes("  ", (uptr*)(aligned_shadow+3*kWordSize));
    PrintBytes("  ", (uptr*)(aligned_shadow+4*kWordSize));
  }
  if (error_report_callback) {
    error_report_callback(error_message_buffer);
  }
  Die();
}


void __asan_init() {
  if (asan_inited) return;
  asan_init_is_running = true;

  // Make sure we are not statically linked.
  AsanDoesNotSupportStaticLinkage();

  // Initialize flags.
  const char *options = GetEnv("ASAN_OPTIONS");
  InitializeFlags(flags(), options);

  if (flags()->verbosity && options) {
    Report("Parsed ASAN_OPTIONS: %s\n", options);
  }

  if (flags()->atexit) {
    Atexit(asan_atexit);
  }

  // interceptors
  InitializeAsanInterceptors();

  ReplaceSystemMalloc();
  ReplaceOperatorsNewAndDelete();

  if (flags()->verbosity) {
    Printf("|| `[%p, %p]` || HighMem    ||\n",
           (void*)kHighMemBeg, (void*)kHighMemEnd);
    Printf("|| `[%p, %p]` || HighShadow ||\n",
           (void*)kHighShadowBeg, (void*)kHighShadowEnd);
    Printf("|| `[%p, %p]` || ShadowGap  ||\n",
           (void*)kShadowGapBeg, (void*)kShadowGapEnd);
    Printf("|| `[%p, %p]` || LowShadow  ||\n",
           (void*)kLowShadowBeg, (void*)kLowShadowEnd);
    Printf("|| `[%p, %p]` || LowMem     ||\n",
           (void*)kLowMemBeg, (void*)kLowMemEnd);
    Printf("MemToShadow(shadow): %p %p %p %p\n",
           (void*)MEM_TO_SHADOW(kLowShadowBeg),
           (void*)MEM_TO_SHADOW(kLowShadowEnd),
           (void*)MEM_TO_SHADOW(kHighShadowBeg),
           (void*)MEM_TO_SHADOW(kHighShadowEnd));
    Printf("red_zone=%zu\n", (uptr)flags()->redzone);
    Printf("malloc_context_size=%zu\n", (uptr)flags()->malloc_context_size);

    Printf("SHADOW_SCALE: %zx\n", (uptr)SHADOW_SCALE);
    Printf("SHADOW_GRANULARITY: %zx\n", (uptr)SHADOW_GRANULARITY);
    Printf("SHADOW_OFFSET: %zx\n", (uptr)SHADOW_OFFSET);
    CHECK(SHADOW_SCALE >= 3 && SHADOW_SCALE <= 7);
  }

  if (flags()->disable_core) {
    DisableCoreDumper();
  }

  uptr shadow_start = kLowShadowBeg;
  if (kLowShadowBeg > 0) shadow_start -= kMmapGranularity;
  uptr shadow_end = kHighShadowEnd;
  if (MemoryRangeIsAvailable(shadow_start, shadow_end)) {
    if (kLowShadowBeg != kLowShadowEnd) {
      // mmap the low shadow plus at least one page.
      ReserveShadowMemoryRange(kLowShadowBeg - kMmapGranularity, kLowShadowEnd);
    }
    // mmap the high shadow.
    ReserveShadowMemoryRange(kHighShadowBeg, kHighShadowEnd);
    // protect the gap
    void *prot = Mprotect(kShadowGapBeg, kShadowGapEnd - kShadowGapBeg + 1);
    CHECK(prot == (void*)kShadowGapBeg);
  } else {
    Report("Shadow memory range interleaves with an existing memory mapping. "
           "ASan cannot proceed correctly. ABORTING.\n");
    DumpProcessMap();
    Die();
  }

  InstallSignalHandlers();

  // On Linux AsanThread::ThreadStart() calls malloc() that's why asan_inited
  // should be set to 1 prior to initializing the threads.
  asan_inited = 1;
  asan_init_is_running = false;

  asanThreadRegistry().Init();
  asanThreadRegistry().GetMain()->ThreadStart();
  force_interface_symbols();  // no-op.

  if (flags()->verbosity) {
    Report("AddressSanitizer Init done\n");
  }
}

#if defined(ASAN_USE_PREINIT_ARRAY)
  // On Linux, we force __asan_init to be called before anyone else
  // by placing it into .preinit_array section.
  // FIXME: do we have anything like this on Mac?
  __attribute__((section(".preinit_array")))
    typeof(__asan_init) *__asan_preinit =__asan_init;
#elif defined(_WIN32) && defined(_DLL)
  // On Windows, when using dynamic CRT (/MD), we can put a pointer
  // to __asan_init into the global list of C initializers.
  // See crt0dat.c in the CRT sources for the details.
  #pragma section(".CRT$XIB", long, read)  // NOLINT
  __declspec(allocate(".CRT$XIB")) void (*__asan_preinit)() = __asan_init;
#endif
