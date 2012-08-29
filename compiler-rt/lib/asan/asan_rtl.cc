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
#include "asan_internal.h"
#include "asan_lock.h"
#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"
#include "sanitizer/asan_interface.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

namespace __sanitizer {
using namespace __asan;

void Die() {
  static atomic_uint32_t num_calls;
  if (atomic_fetch_add(&num_calls, 1, memory_order_relaxed) != 0) {
    // Don't die twice - run a busy loop.
    while (1) { }
  }
  if (flags()->sleep_before_dying) {
    Report("Sleeping for %d second(s)\n", flags()->sleep_before_dying);
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

SANITIZER_INTERFACE_ATTRIBUTE
void CheckFailed(const char *file, int line, const char *cond, u64 v1, u64 v2) {
  Report("AddressSanitizer CHECK failed: %s:%d \"%s\" (0x%zx, 0x%zx)\n",
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
  ParseFlag(str, &f->check_initialization_order, "initialization_order");
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
  ParseFlag(str, &f->strip_path_prefix, "strip_path_prefix");
  ParseFlag(str, &f->allow_reexec, "allow_reexec");
}

extern "C" {
SANITIZER_WEAK_ATTRIBUTE
SANITIZER_INTERFACE_ATTRIBUTE
const char* __asan_default_options() { return ""; }
}  // extern "C"

void InitializeFlags(Flags *f, const char *env) {
  internal_memset(f, 0, sizeof(*f));

  f->quarantine_size = (ASAN_LOW_MEMORY) ? 1UL << 24 : 1UL << 28;
  f->symbolize = false;
  f->verbosity = 0;
  f->redzone = (ASAN_LOW_MEMORY) ? 64 : 128;
  f->debug = false;
  f->report_globals = 1;
  f->check_initialization_order = true;
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
  f->strip_path_prefix = "";
  f->allow_reexec = true;

  // Override from user-specified string.
  ParseFlagsFromString(f, __asan_default_options());
  if (flags()->verbosity) {
    Report("Using the defaults from __asan_default_options: %s\n",
           __asan_default_options());
  }

  // Override from command line.
  ParseFlagsFromString(f, env);
}

// -------------------------- Globals --------------------- {{{1
int asan_inited;
bool asan_init_is_running;
void (*death_callback)(void);

// -------------------------- Misc ---------------- {{{1
void ShowStatsAndAbort() {
  __asan_print_accumulated_stats();
  Die();
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

// --------------- LowLevelAllocateCallbac ---------- {{{1
static void OnLowLevelAllocate(uptr ptr, uptr size) {
  PoisonShadow(ptr, size, kAsanInternalHeapMagic);
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
  // __asan_report_* functions are noreturn, so we need a switch to prevent
  // the compiler from removing any of them.
  switch (fake_condition) {
    case 1: __asan_report_load1(0); break;
    case 2: __asan_report_load2(0); break;
    case 3: __asan_report_load4(0); break;
    case 4: __asan_report_load8(0); break;
    case 5: __asan_report_load16(0); break;
    case 6: __asan_report_store1(0); break;
    case 7: __asan_report_store2(0); break;
    case 8: __asan_report_store4(0); break;
    case 9: __asan_report_store8(0); break;
    case 10: __asan_report_store16(0); break;
    case 11: __asan_register_global(0, 0, 0); break;
    case 12: __asan_register_globals(0, 0); break;
    case 13: __asan_unregister_globals(0, 0); break;
    case 14: __asan_set_death_callback(0); break;
    case 15: __asan_set_error_report_callback(0); break;
    case 16: __asan_handle_no_return(); break;
    case 17: __asan_address_is_poisoned(0); break;
    case 18: __asan_get_allocated_size(0); break;
    case 19: __asan_get_current_allocated_bytes(); break;
    case 20: __asan_get_estimated_allocated_size(0); break;
    case 21: __asan_get_free_bytes(); break;
    case 22: __asan_get_heap_size(); break;
    case 23: __asan_get_ownership(0); break;
    case 24: __asan_get_unmapped_bytes(); break;
    case 25: __asan_poison_memory_region(0, 0); break;
    case 26: __asan_unpoison_memory_region(0, 0); break;
    case 27: __asan_set_error_exit_code(0); break;
    case 28: __asan_stack_free(0, 0, 0); break;
    case 29: __asan_stack_malloc(0, 0); break;
    case 30: __asan_set_on_error_callback(0); break;
    case 31: __asan_default_options(); break;
    case 32: __asan_before_dynamic_init(0, 0); break;
    case 33: __asan_after_dynamic_init(); break;
    case 34: __asan_malloc_hook(0, 0); break;
    case 35: __asan_free_hook(0); break;
    case 36: __asan_set_symbolize_callback(0); break;
  }
}

static void asan_atexit() {
  Printf("AddressSanitizer exit stats:\n");
  __asan_print_accumulated_stats();
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

int NOINLINE __asan_set_error_exit_code(int exit_code) {
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

void __asan_init() {
  if (asan_inited) return;
  CHECK(!asan_init_is_running && "ASan init calls itself!");
  asan_init_is_running = true;

  // Make sure we are not statically linked.
  AsanDoesNotSupportStaticLinkage();

  SetPrintfAndReportCallback(AppendToErrorMessageBuffer);

  // Initialize flags. This must be done early, because most of the
  // initialization steps look at flags().
  const char *options = GetEnv("ASAN_OPTIONS");
  InitializeFlags(flags(), options);

  if (flags()->verbosity && options) {
    Report("Parsed ASAN_OPTIONS: %s\n", options);
  }

  // Re-exec ourselves if we need to set additional env or command line args.
  MaybeReexec();

  // Setup internal allocator callback.
  SetLowLevelAllocateCallback(OnLowLevelAllocate);

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
  // Start symbolizer process if necessary.
  if (flags()->symbolize) {
    const char *external_symbolizer = GetEnv("ASAN_SYMBOLIZER_PATH");
    if (external_symbolizer) {
      InitializeExternalSymbolizer(external_symbolizer);
    }
  }
#ifdef _WIN32
  __asan_set_symbolize_callback(WinSymbolize);
#endif  // _WIN32

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
