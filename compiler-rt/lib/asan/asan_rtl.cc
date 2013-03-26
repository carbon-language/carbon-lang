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
#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

namespace __asan {

uptr AsanMappingProfile[kAsanMappingProfileSize];

static void AsanDie() {
  static atomic_uint32_t num_calls;
  if (atomic_fetch_add(&num_calls, 1, memory_order_relaxed) != 0) {
    // Don't die twice - run a busy loop.
    while (1) { }
  }
  if (flags()->sleep_before_dying) {
    Report("Sleeping for %d second(s)\n", flags()->sleep_before_dying);
    SleepForSeconds(flags()->sleep_before_dying);
  }
  if (flags()->unmap_shadow_on_exit) {
    if (kMidMemBeg) {
      UnmapOrDie((void*)kLowShadowBeg, kMidMemBeg - kLowShadowBeg);
      UnmapOrDie((void*)kMidMemEnd, kHighShadowEnd - kMidMemEnd);
    } else {
      UnmapOrDie((void*)kLowShadowBeg, kHighShadowEnd - kLowShadowBeg);
    }
  }
  if (death_callback)
    death_callback();
  if (flags()->abort_on_error)
    Abort();
  internal__exit(flags()->exitcode);
}

static void AsanCheckFailed(const char *file, int line, const char *cond,
                            u64 v1, u64 v2) {
  Report("AddressSanitizer CHECK failed: %s:%d \"%s\" (0x%zx, 0x%zx)\n",
             file, line, cond, (uptr)v1, (uptr)v2);
  // FIXME: check for infinite recursion without a thread-local counter here.
  PRINT_CURRENT_STACK();
  Die();
}

// -------------------------- Flags ------------------------- {{{1
static const int kDeafultMallocContextSize = 30;

static Flags asan_flags;

Flags *flags() {
  return &asan_flags;
}

static const char *MaybeCallAsanDefaultOptions() {
  return (&__asan_default_options) ? __asan_default_options() : "";
}

static const char *MaybeUseAsanDefaultOptionsCompileDefiniton() {
#ifdef ASAN_DEFAULT_OPTIONS
// Stringize the macro value.
# define ASAN_STRINGIZE(x) #x
# define ASAN_STRINGIZE_OPTIONS(options) ASAN_STRINGIZE(options)
  return ASAN_STRINGIZE_OPTIONS(ASAN_DEFAULT_OPTIONS);
#else
  return "";
#endif
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
  CHECK((uptr)f->malloc_context_size <= kStackTraceMax);

  ParseFlag(str, &f->replace_str, "replace_str");
  ParseFlag(str, &f->replace_intrin, "replace_intrin");
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
  ParseFlag(str, &f->print_stats, "print_stats");
  ParseFlag(str, &f->print_legend, "print_legend");
  ParseFlag(str, &f->atexit, "atexit");
  ParseFlag(str, &f->disable_core, "disable_core");
  ParseFlag(str, &f->strip_path_prefix, "strip_path_prefix");
  ParseFlag(str, &f->allow_reexec, "allow_reexec");
  ParseFlag(str, &f->print_full_thread_history, "print_full_thread_history");
  ParseFlag(str, &f->log_path, "log_path");
  ParseFlag(str, &f->fast_unwind_on_fatal, "fast_unwind_on_fatal");
  ParseFlag(str, &f->fast_unwind_on_malloc, "fast_unwind_on_malloc");
  ParseFlag(str, &f->poison_heap, "poison_heap");
  ParseFlag(str, &f->alloc_dealloc_mismatch, "alloc_dealloc_mismatch");
  ParseFlag(str, &f->use_stack_depot, "use_stack_depot");
  ParseFlag(str, &f->strict_memcmp, "strict_memcmp");
}

static const char *asan_external_symbolizer;

void InitializeFlags(Flags *f, const char *env) {
  internal_memset(f, 0, sizeof(*f));

  f->quarantine_size = (ASAN_LOW_MEMORY) ? 1UL << 26 : 1UL << 28;
  f->symbolize = (asan_external_symbolizer != 0);
  f->verbosity = 0;
  f->redzone = ASAN_ALLOCATOR_VERSION == 2 ? 16 : (ASAN_LOW_MEMORY) ? 64 : 128;
  f->debug = false;
  f->report_globals = 1;
  f->check_initialization_order = false;
  f->malloc_context_size = kDeafultMallocContextSize;
  f->replace_str = true;
  f->replace_intrin = true;
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
  f->print_stats = false;
  f->print_legend = true;
  f->atexit = false;
  f->disable_core = (SANITIZER_WORDSIZE == 64);
  f->strip_path_prefix = "";
  f->allow_reexec = true;
  f->print_full_thread_history = true;
  f->log_path = 0;
  f->fast_unwind_on_fatal = false;
  f->fast_unwind_on_malloc = true;
  f->poison_heap = true;
  // Turn off alloc/dealloc mismatch checker on Mac for now.
  // TODO(glider): Fix known issues and enable this back.
  f->alloc_dealloc_mismatch = (SANITIZER_MAC == 0);;
  f->use_stack_depot = true;  // Only affects allocator2.
  f->strict_memcmp = true;

  // Override from compile definition.
  ParseFlagsFromString(f, MaybeUseAsanDefaultOptionsCompileDefiniton());

  // Override from user-specified string.
  ParseFlagsFromString(f, MaybeCallAsanDefaultOptions());
  if (flags()->verbosity) {
    Report("Using the defaults from __asan_default_options: %s\n",
           MaybeCallAsanDefaultOptions());
  }

  // Override from command line.
  ParseFlagsFromString(f, env);
}

// -------------------------- Globals --------------------- {{{1
int asan_inited;
bool asan_init_is_running;
void (*death_callback)(void);

#if !ASAN_FIXED_MAPPING
uptr kHighMemEnd, kMidMemBeg, kMidMemEnd;
#endif

// -------------------------- Misc ---------------- {{{1
void ShowStatsAndAbort() {
  __asan_print_accumulated_stats();
  Die();
}

// ---------------------- mmap -------------------- {{{1
// Reserve memory range [beg, end].
static void ReserveShadowMemoryRange(uptr beg, uptr end) {
  CHECK((beg % GetPageSizeCached()) == 0);
  CHECK(((end + 1) % GetPageSizeCached()) == 0);
  uptr size = end - beg + 1;
  void *res = MmapFixedNoReserve(beg, size);
  if (res != (void*)beg) {
    Report("ReserveShadowMemoryRange failed while trying to map 0x%zx bytes. "
           "Perhaps you're using ulimit -v\n", size);
    Abort();
  }
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

#define ASAN_REPORT_ERROR_N(type, is_write)                    \
extern "C" NOINLINE INTERFACE_ATTRIBUTE                        \
void __asan_report_ ## type ## _n(uptr addr, uptr size);       \
void __asan_report_ ## type ## _n(uptr addr, uptr size) {      \
  GET_CALLER_PC_BP_SP;                                         \
  __asan_report_error(pc, bp, sp, addr, is_write, size);       \
}

ASAN_REPORT_ERROR_N(load, false)
ASAN_REPORT_ERROR_N(store, true)

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
    case 30: __asan_before_dynamic_init(0); break;
    case 31: __asan_after_dynamic_init(); break;
    case 32: __asan_poison_stack_memory(0, 0); break;
    case 33: __asan_unpoison_stack_memory(0, 0); break;
    case 34: __asan_region_is_poisoned(0, 0); break;
    case 35: __asan_describe_address(0); break;
  }
}

static void asan_atexit() {
  Printf("AddressSanitizer exit stats:\n");
  __asan_print_accumulated_stats();
  // Print AsanMappingProfile.
  for (uptr i = 0; i < kAsanMappingProfileSize; i++) {
    if (AsanMappingProfile[i] == 0) continue;
    Printf("asan_mapping.h:%zd -- %zd\n", i, AsanMappingProfile[i]);
  }
}

static void InitializeHighMemEnd() {
#if !ASAN_FIXED_MAPPING
#if SANITIZER_WORDSIZE == 64
# if defined(__powerpc64__)
  // FIXME:
  // On PowerPC64 we have two different address space layouts: 44- and 46-bit.
  // We somehow need to figure our which one we are using now and choose
  // one of 0x00000fffffffffffUL and 0x00003fffffffffffUL.
  // Note that with 'ulimit -s unlimited' the stack is moved away from the top
  // of the address space, so simply checking the stack address is not enough.
  kHighMemEnd = (1ULL << 44) - 1;  // 0x00000fffffffffffUL
# else
  kHighMemEnd = (1ULL << 47) - 1;  // 0x00007fffffffffffUL;
# endif
#else  // SANITIZER_WORDSIZE == 32
  kHighMemEnd = (1ULL << 32) - 1;  // 0xffffffff;
#endif  // SANITIZER_WORDSIZE
#endif  // !ASAN_FIXED_MAPPING
}

static void ProtectGap(uptr a, uptr size) {
  CHECK_EQ(a, (uptr)Mprotect(a, size));
}

static void PrintAddressSpaceLayout() {
  Printf("|| `[%p, %p]` || HighMem    ||\n",
         (void*)kHighMemBeg, (void*)kHighMemEnd);
  Printf("|| `[%p, %p]` || HighShadow ||\n",
         (void*)kHighShadowBeg, (void*)kHighShadowEnd);
  if (kMidMemBeg) {
    Printf("|| `[%p, %p]` || ShadowGap3 ||\n",
           (void*)kShadowGap3Beg, (void*)kShadowGap3End);
    Printf("|| `[%p, %p]` || MidMem     ||\n",
           (void*)kMidMemBeg, (void*)kMidMemEnd);
    Printf("|| `[%p, %p]` || ShadowGap2 ||\n",
           (void*)kShadowGap2Beg, (void*)kShadowGap2End);
    Printf("|| `[%p, %p]` || MidShadow  ||\n",
           (void*)kMidShadowBeg, (void*)kMidShadowEnd);
  }
  Printf("|| `[%p, %p]` || ShadowGap  ||\n",
         (void*)kShadowGapBeg, (void*)kShadowGapEnd);
  if (kLowShadowBeg) {
    Printf("|| `[%p, %p]` || LowShadow  ||\n",
           (void*)kLowShadowBeg, (void*)kLowShadowEnd);
    Printf("|| `[%p, %p]` || LowMem     ||\n",
           (void*)kLowMemBeg, (void*)kLowMemEnd);
  }
  Printf("MemToShadow(shadow): %p %p %p %p",
         (void*)MEM_TO_SHADOW(kLowShadowBeg),
         (void*)MEM_TO_SHADOW(kLowShadowEnd),
         (void*)MEM_TO_SHADOW(kHighShadowBeg),
         (void*)MEM_TO_SHADOW(kHighShadowEnd));
  if (kMidMemBeg) {
    Printf(" %p %p",
           (void*)MEM_TO_SHADOW(kMidShadowBeg),
           (void*)MEM_TO_SHADOW(kMidShadowEnd));
  }
  Printf("\n");
  Printf("red_zone=%zu\n", (uptr)flags()->redzone);
  Printf("malloc_context_size=%zu\n", (uptr)flags()->malloc_context_size);

  Printf("SHADOW_SCALE: %zx\n", (uptr)SHADOW_SCALE);
  Printf("SHADOW_GRANULARITY: %zx\n", (uptr)SHADOW_GRANULARITY);
  Printf("SHADOW_OFFSET: %zx\n", (uptr)SHADOW_OFFSET);
  CHECK(SHADOW_SCALE >= 3 && SHADOW_SCALE <= 7);
  if (kMidMemBeg)
    CHECK(kMidShadowBeg > kLowShadowEnd &&
          kMidMemBeg > kMidShadowEnd &&
          kHighShadowBeg > kMidMemEnd);
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
extern "C" {
SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE
const char* __asan_default_options() { return ""; }
}  // extern "C"
#endif

int NOINLINE __asan_set_error_exit_code(int exit_code) {
  int old = flags()->exitcode;
  flags()->exitcode = exit_code;
  return old;
}

void NOINLINE __asan_handle_no_return() {
  int local_stack;
  AsanThread *curr_thread = GetCurrentThread();
  CHECK(curr_thread);
  uptr PageSize = GetPageSizeCached();
  uptr top = curr_thread->stack_top();
  uptr bottom = ((uptr)&local_stack - PageSize) & ~(PageSize-1);
  PoisonShadow(bottom, top - bottom, 0);
}

void NOINLINE __asan_set_death_callback(void (*callback)(void)) {
  death_callback = callback;
}

void __asan_init() {
  if (asan_inited) return;
  SanitizerToolName = "AddressSanitizer";
  CHECK(!asan_init_is_running && "ASan init calls itself!");
  asan_init_is_running = true;
  InitializeHighMemEnd();

  // Make sure we are not statically linked.
  AsanDoesNotSupportStaticLinkage();

  // Install tool-specific callbacks in sanitizer_common.
  SetDieCallback(AsanDie);
  SetCheckFailedCallback(AsanCheckFailed);
  SetPrintfAndReportCallback(AppendToErrorMessageBuffer);

  // Check if external symbolizer is defined before parsing the flags.
  asan_external_symbolizer = GetEnv("ASAN_SYMBOLIZER_PATH");
  // Initialize flags. This must be done early, because most of the
  // initialization steps look at flags().
  const char *options = GetEnv("ASAN_OPTIONS");
  InitializeFlags(flags(), options);
  __sanitizer_set_report_path(flags()->log_path);

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

  uptr shadow_start = kLowShadowBeg;
  if (kLowShadowBeg) shadow_start -= GetMmapGranularity();
  uptr shadow_end = kHighShadowEnd;
  bool full_shadow_is_available =
      MemoryRangeIsAvailable(shadow_start, shadow_end);

#if SANITIZER_LINUX && defined(__x86_64__) && !ASAN_FIXED_MAPPING
  if (!full_shadow_is_available) {
    kMidMemBeg = kLowMemEnd < 0x3000000000ULL ? 0x3000000000ULL : 0;
    kMidMemEnd = kLowMemEnd < 0x3000000000ULL ? 0x4fffffffffULL : 0;
  }
#endif

  if (flags()->verbosity)
    PrintAddressSpaceLayout();

  if (flags()->disable_core) {
    DisableCoreDumper();
  }

  if (full_shadow_is_available) {
    // mmap the low shadow plus at least one page at the left.
    if (kLowShadowBeg)
      ReserveShadowMemoryRange(shadow_start, kLowShadowEnd);
    // mmap the high shadow.
    ReserveShadowMemoryRange(kHighShadowBeg, kHighShadowEnd);
    // protect the gap.
    ProtectGap(kShadowGapBeg, kShadowGapEnd - kShadowGapBeg + 1);
  } else if (kMidMemBeg &&
      MemoryRangeIsAvailable(shadow_start, kMidMemBeg - 1) &&
      MemoryRangeIsAvailable(kMidMemEnd + 1, shadow_end)) {
    CHECK(kLowShadowBeg != kLowShadowEnd);
    // mmap the low shadow plus at least one page at the left.
    ReserveShadowMemoryRange(shadow_start, kLowShadowEnd);
    // mmap the mid shadow.
    ReserveShadowMemoryRange(kMidShadowBeg, kMidShadowEnd);
    // mmap the high shadow.
    ReserveShadowMemoryRange(kHighShadowBeg, kHighShadowEnd);
    // protect the gaps.
    ProtectGap(kShadowGapBeg, kShadowGapEnd - kShadowGapBeg + 1);
    ProtectGap(kShadowGap2Beg, kShadowGap2End - kShadowGap2Beg + 1);
    ProtectGap(kShadowGap3Beg, kShadowGap3End - kShadowGap3Beg + 1);
  } else {
    Report("Shadow memory range interleaves with an existing memory mapping. "
           "ASan cannot proceed correctly. ABORTING.\n");
    DumpProcessMap();
    Die();
  }

  InstallSignalHandlers();
  // Start symbolizer process if necessary.
  if (flags()->symbolize && asan_external_symbolizer &&
      asan_external_symbolizer[0]) {
    InitializeExternalSymbolizer(asan_external_symbolizer);
  }

  // On Linux AsanThread::ThreadStart() calls malloc() that's why asan_inited
  // should be set to 1 prior to initializing the threads.
  asan_inited = 1;
  asan_init_is_running = false;

  // Create main thread.
  AsanTSDInit(AsanThread::TSDDtor);
  AsanThread *main_thread = AsanThread::Create(0, 0);
  CreateThreadContextArgs create_main_args = { main_thread, 0 };
  u32 main_tid = asanThreadRegistry().CreateThread(
      0, true, 0, &create_main_args);
  CHECK_EQ(0, main_tid);
  SetCurrentThread(main_thread);
  main_thread->ThreadStart(GetPid());
  force_interface_symbols();  // no-op.

  InitializeAllocator();

  if (flags()->verbosity) {
    Report("AddressSanitizer Init done\n");
  }
}
