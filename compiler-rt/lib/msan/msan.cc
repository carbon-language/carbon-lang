//===-- msan.cc -----------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// MemorySanitizer runtime.
//===----------------------------------------------------------------------===//

#include "msan.h"
#include "msan_chained_origin_depot.h"
#include "msan_origin.h"
#include "msan_thread.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "sanitizer_common/sanitizer_stackdepot.h"


// ACHTUNG! No system header includes in this file.

using namespace __sanitizer;

// Globals.
static THREADLOCAL int msan_expect_umr = 0;
static THREADLOCAL int msan_expected_umr_found = 0;

static bool msan_running_under_dr;

// Function argument shadow. Each argument starts at the next available 8-byte
// aligned address.
SANITIZER_INTERFACE_ATTRIBUTE
THREADLOCAL u64 __msan_param_tls[kMsanParamTlsSizeInWords];

// Function argument origin. Each argument starts at the same offset as the
// corresponding shadow in (__msan_param_tls). Slightly weird, but changing this
// would break compatibility with older prebuilt binaries.
SANITIZER_INTERFACE_ATTRIBUTE
THREADLOCAL u32 __msan_param_origin_tls[kMsanParamTlsSizeInWords];

SANITIZER_INTERFACE_ATTRIBUTE
THREADLOCAL u64 __msan_retval_tls[kMsanRetvalTlsSizeInWords];

SANITIZER_INTERFACE_ATTRIBUTE
THREADLOCAL u32 __msan_retval_origin_tls;

SANITIZER_INTERFACE_ATTRIBUTE
THREADLOCAL u64 __msan_va_arg_tls[kMsanParamTlsSizeInWords];

SANITIZER_INTERFACE_ATTRIBUTE
THREADLOCAL u64 __msan_va_arg_overflow_size_tls;

SANITIZER_INTERFACE_ATTRIBUTE
THREADLOCAL u32 __msan_origin_tls;

static THREADLOCAL int is_in_symbolizer;
static THREADLOCAL int is_in_loader;

extern "C" SANITIZER_WEAK_ATTRIBUTE const int __msan_track_origins;

int __msan_get_track_origins() {
  return &__msan_track_origins ? __msan_track_origins : 0;
}

extern "C" SANITIZER_WEAK_ATTRIBUTE const int __msan_keep_going;

namespace __msan {

void EnterSymbolizer() { ++is_in_symbolizer; }
void ExitSymbolizer()  { --is_in_symbolizer; }
bool IsInSymbolizer() { return is_in_symbolizer; }

void EnterLoader() { ++is_in_loader; }
void ExitLoader()  { --is_in_loader; }

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
bool __msan_is_in_loader() { return is_in_loader; }
}

static Flags msan_flags;

Flags *flags() {
  return &msan_flags;
}

int msan_inited = 0;
bool msan_init_is_running;

int msan_report_count = 0;

void (*death_callback)(void);

// Array of stack origins.
// FIXME: make it resizable.
static const uptr kNumStackOriginDescrs = 1024 * 1024;
static const char *StackOriginDescr[kNumStackOriginDescrs];
static uptr StackOriginPC[kNumStackOriginDescrs];
static atomic_uint32_t NumStackOriginDescrs;

static void ParseFlagsFromString(Flags *f, const char *str) {
  CommonFlags *cf = common_flags();
  ParseCommonFlagsFromString(cf, str);
  ParseFlag(str, &f->poison_heap_with_zeroes, "poison_heap_with_zeroes", "");
  ParseFlag(str, &f->poison_stack_with_zeroes, "poison_stack_with_zeroes", "");
  ParseFlag(str, &f->poison_in_malloc, "poison_in_malloc", "");
  ParseFlag(str, &f->poison_in_free, "poison_in_free", "");
  ParseFlag(str, &f->exit_code, "exit_code", "");
  if (f->exit_code < 0 || f->exit_code > 127) {
    Printf("Exit code not in [0, 128) range: %d\n", f->exit_code);
    Die();
  }
  ParseFlag(str, &f->origin_history_size, "origin_history_size", "");
  if (f->origin_history_size < 0 ||
      f->origin_history_size > Origin::kMaxDepth) {
    Printf(
        "Origin history size invalid: %d. Must be 0 (unlimited) or in [1, %d] "
        "range.\n",
        f->origin_history_size, Origin::kMaxDepth);
    Die();
  }
  ParseFlag(str, &f->origin_history_per_stack_limit,
            "origin_history_per_stack_limit", "");
  // Limiting to kStackDepotMaxUseCount / 2 to avoid overflow in
  // StackDepotHandle::inc_use_count_unsafe.
  if (f->origin_history_per_stack_limit < 0 ||
      f->origin_history_per_stack_limit > kStackDepotMaxUseCount / 2) {
    Printf(
        "Origin per-stack limit invalid: %d. Must be 0 (unlimited) or in [1, "
        "%d] range.\n",
        f->origin_history_per_stack_limit, kStackDepotMaxUseCount / 2);
    Die();
  }

  ParseFlag(str, &f->report_umrs, "report_umrs", "");
  ParseFlag(str, &f->wrap_signals, "wrap_signals", "");
  ParseFlag(str, &f->print_stats, "print_stats", "");
  ParseFlag(str, &f->atexit, "atexit", "");
  ParseFlag(str, &f->store_context_size, "store_context_size", "");
  if (f->store_context_size < 1) f->store_context_size = 1;

  // keep_going is an old name for halt_on_error,
  // and it has inverse meaning.
  f->halt_on_error = !f->halt_on_error;
  ParseFlag(str, &f->halt_on_error, "keep_going", "");
  f->halt_on_error = !f->halt_on_error;
  ParseFlag(str, &f->halt_on_error, "halt_on_error", "");
}

static void InitializeFlags(Flags *f, const char *options) {
  CommonFlags *cf = common_flags();
  SetCommonFlagsDefaults(cf);
  cf->external_symbolizer_path = GetEnv("MSAN_SYMBOLIZER_PATH");
  cf->malloc_context_size = 20;
  cf->handle_ioctl = true;
  // FIXME: test and enable.
  cf->check_printf = false;
  cf->intercept_tls_get_addr = true;

  internal_memset(f, 0, sizeof(*f));
  f->poison_heap_with_zeroes = false;
  f->poison_stack_with_zeroes = false;
  f->poison_in_malloc = true;
  f->poison_in_free = true;
  f->exit_code = 77;
  f->origin_history_size = Origin::kMaxDepth;
  f->origin_history_per_stack_limit = 20000;
  f->report_umrs = true;
  f->wrap_signals = true;
  f->print_stats = false;
  f->atexit = false;
  f->halt_on_error = !&__msan_keep_going;
  f->store_context_size = 20;

  // Override from user-specified string.
  if (__msan_default_options)
    ParseFlagsFromString(f, __msan_default_options());
  ParseFlagsFromString(f, options);
}

void GetStackTrace(StackTrace *stack, uptr max_s, uptr pc, uptr bp,
                   bool request_fast_unwind) {
  MsanThread *t = GetCurrentThread();
  if (!t || !StackTrace::WillUseFastUnwind(request_fast_unwind)) {
    // Block reports from our interceptors during _Unwind_Backtrace.
    SymbolizerScope sym_scope;
    return stack->Unwind(max_s, pc, bp, 0, 0, 0, request_fast_unwind);
  }
  stack->Unwind(max_s, pc, bp, 0, t->stack_top(), t->stack_bottom(),
                request_fast_unwind);
}

void PrintWarning(uptr pc, uptr bp) {
  PrintWarningWithOrigin(pc, bp, __msan_origin_tls);
}

void PrintWarningWithOrigin(uptr pc, uptr bp, u32 origin) {
  if (msan_expect_umr) {
    // Printf("Expected UMR\n");
    __msan_origin_tls = origin;
    msan_expected_umr_found = 1;
    return;
  }

  ++msan_report_count;

  GET_FATAL_STACK_TRACE_PC_BP(pc, bp);

  u32 report_origin =
    (__msan_get_track_origins() && Origin(origin).isValid()) ? origin : 0;
  ReportUMR(&stack, report_origin);

  if (__msan_get_track_origins() && !Origin(origin).isValid()) {
    Printf(
        "  ORIGIN: invalid (%x). Might be a bug in MemorySanitizer origin "
        "tracking.\n    This could still be a bug in your code, too!\n",
        origin);
  }
}

void UnpoisonParam(uptr n) {
  internal_memset(__msan_param_tls, 0, n * sizeof(*__msan_param_tls));
}

// Backup MSan runtime TLS state.
// Implementation must be async-signal-safe.
// Instances of this class may live on the signal handler stack, and data size
// may be an issue.
void ScopedThreadLocalStateBackup::Backup() {
  va_arg_overflow_size_tls = __msan_va_arg_overflow_size_tls;
}

void ScopedThreadLocalStateBackup::Restore() {
  // A lame implementation that only keeps essential state and resets the rest.
  __msan_va_arg_overflow_size_tls = va_arg_overflow_size_tls;

  internal_memset(__msan_param_tls, 0, sizeof(__msan_param_tls));
  internal_memset(__msan_retval_tls, 0, sizeof(__msan_retval_tls));
  internal_memset(__msan_va_arg_tls, 0, sizeof(__msan_va_arg_tls));

  if (__msan_get_track_origins()) {
    internal_memset(&__msan_retval_origin_tls, 0,
                    sizeof(__msan_retval_origin_tls));
    internal_memset(__msan_param_origin_tls, 0,
                    sizeof(__msan_param_origin_tls));
  }
}

void UnpoisonThreadLocalState() {
}

const char *GetStackOriginDescr(u32 id, uptr *pc) {
  CHECK_LT(id, kNumStackOriginDescrs);
  if (pc) *pc = StackOriginPC[id];
  return StackOriginDescr[id];
}

u32 ChainOrigin(u32 id, StackTrace *stack) {
  MsanThread *t = GetCurrentThread();
  if (t && t->InSignalHandler())
    return id;

  Origin o(id);
  int depth = o.depth();
  // 0 means unlimited depth.
  if (flags()->origin_history_size > 0 && depth > 0) {
    if (depth >= flags()->origin_history_size) {
      return id;
    } else {
      ++depth;
    }
  }

  StackDepotHandle h = StackDepotPut_WithHandle(stack->trace, stack->size);
  if (!h.valid()) return id;
  int use_count = h.use_count();
  if (use_count > flags()->origin_history_per_stack_limit)
    return id;

  u32 chained_id;
  bool inserted = ChainedOriginDepotPut(h.id(), o.id(), &chained_id);

  if (inserted) h.inc_use_count_unsafe();

  return Origin(chained_id, depth).raw_id();
}

}  // namespace __msan

// Interface.

using namespace __msan;

#define MSAN_MAYBE_WARNING(type, size)              \
  void __msan_maybe_warning_##size(type s, u32 o) { \
    GET_CALLER_PC_BP_SP;                            \
    (void) sp;                                      \
    if (UNLIKELY(s)) {                              \
      PrintWarningWithOrigin(pc, bp, o);            \
      if (__msan::flags()->halt_on_error) {         \
        Printf("Exiting\n");                        \
        Die();                                      \
      }                                             \
    }                                               \
  }

MSAN_MAYBE_WARNING(u8, 1)
MSAN_MAYBE_WARNING(u16, 2)
MSAN_MAYBE_WARNING(u32, 4)
MSAN_MAYBE_WARNING(u64, 8)

#define MSAN_MAYBE_STORE_ORIGIN(type, size)                       \
  void __msan_maybe_store_origin_##size(type s, void *p, u32 o) { \
    if (UNLIKELY(s)) {                                            \
      if (__msan_get_track_origins() > 1) {                       \
        GET_CALLER_PC_BP_SP;                                      \
        (void) sp;                                                \
        GET_STORE_STACK_TRACE_PC_BP(pc, bp);                      \
        o = ChainOrigin(o, &stack);                               \
      }                                                           \
      *(u32 *)MEM_TO_ORIGIN((uptr)p & ~3UL) = o;                  \
    }                                                             \
  }

MSAN_MAYBE_STORE_ORIGIN(u8, 1)
MSAN_MAYBE_STORE_ORIGIN(u16, 2)
MSAN_MAYBE_STORE_ORIGIN(u32, 4)
MSAN_MAYBE_STORE_ORIGIN(u64, 8)

void __msan_warning() {
  GET_CALLER_PC_BP_SP;
  (void)sp;
  PrintWarning(pc, bp);
  if (__msan::flags()->halt_on_error) {
    if (__msan::flags()->print_stats)
      ReportStats();
    Printf("Exiting\n");
    Die();
  }
}

void __msan_warning_noreturn() {
  GET_CALLER_PC_BP_SP;
  (void)sp;
  PrintWarning(pc, bp);
  if (__msan::flags()->print_stats)
    ReportStats();
  Printf("Exiting\n");
  Die();
}

void __msan_init() {
  CHECK(!msan_init_is_running);
  if (msan_inited) return;
  msan_init_is_running = 1;
  SanitizerToolName = "MemorySanitizer";

  SetDieCallback(MsanDie);
  InitTlsSize();

  const char *msan_options = GetEnv("MSAN_OPTIONS");
  InitializeFlags(&msan_flags, msan_options);
  if (common_flags()->help) PrintFlagDescriptions();
  __sanitizer_set_report_path(common_flags()->log_path);

  InitializeInterceptors();
  InstallAtExitHandler(); // Needs __cxa_atexit interceptor.

  if (MSAN_REPLACE_OPERATORS_NEW_AND_DELETE)
    ReplaceOperatorsNewAndDelete();
  if (StackSizeIsUnlimited()) {
    VPrintf(1, "Unlimited stack, doing reexec\n");
    // A reasonably large stack size. It is bigger than the usual 8Mb, because,
    // well, the program could have been run with unlimited stack for a reason.
    SetStackSizeLimitInBytes(32 * 1024 * 1024);
    ReExec();
  }

  VPrintf(1, "MSAN_OPTIONS: %s\n", msan_options ? msan_options : "<empty>");

  __msan_clear_on_return();
  if (__msan_get_track_origins())
    VPrintf(1, "msan_track_origins\n");
  if (!InitShadow(/* prot1 */ !msan_running_under_dr, /* prot2 */ true,
                  /* map_shadow */ true, __msan_get_track_origins())) {
    Printf("FATAL: MemorySanitizer can not mmap the shadow memory.\n");
    Printf("FATAL: Make sure to compile with -fPIE and to link with -pie.\n");
    Printf("FATAL: Disabling ASLR is known to cause this error.\n");
    Printf("FATAL: If running under GDB, try "
           "'set disable-randomization off'.\n");
    DumpProcessMap();
    Die();
  }

  Symbolizer::Init(common_flags()->external_symbolizer_path);
  Symbolizer::Get()->AddHooks(EnterSymbolizer, ExitSymbolizer);

  MsanTSDInit(MsanTSDDtor);

  MsanThread *main_thread = MsanThread::Create(0, 0);
  SetCurrentThread(main_thread);
  main_thread->ThreadStart();

  VPrintf(1, "MemorySanitizer init done\n");

  msan_init_is_running = 0;
  msan_inited = 1;
}

void __msan_set_exit_code(int exit_code) {
  flags()->exit_code = exit_code;
}

void __msan_set_keep_going(int keep_going) {
  flags()->halt_on_error = !keep_going;
}

void __msan_set_expect_umr(int expect_umr) {
  if (expect_umr) {
    msan_expected_umr_found = 0;
  } else if (!msan_expected_umr_found) {
    GET_CALLER_PC_BP_SP;
    (void)sp;
    GET_FATAL_STACK_TRACE_PC_BP(pc, bp);
    ReportExpectedUMRNotFound(&stack);
    Die();
  }
  msan_expect_umr = expect_umr;
}

void __msan_print_shadow(const void *x, uptr size) {
  if (!MEM_IS_APP(x)) {
    Printf("Not a valid application address: %p\n", x);
    return;
  }

  DescribeMemoryRange(x, size);
}

void __msan_dump_shadow(const void *x, uptr size) {
  if (!MEM_IS_APP(x)) {
    Printf("Not a valid application address: %p\n", x);
    return;
  }

  unsigned char *s = (unsigned char*)MEM_TO_SHADOW(x);
  for (uptr i = 0; i < size; i++) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    Printf("%x%x ", s[i] & 0xf, s[i] >> 4);
#else
    Printf("%x%x ", s[i] >> 4, s[i] & 0xf);
#endif
  }
  Printf("\n");
}

sptr __msan_test_shadow(const void *x, uptr size) {
  if (!MEM_IS_APP(x)) return -1;
  unsigned char *s = (unsigned char *)MEM_TO_SHADOW((uptr)x);
  for (uptr i = 0; i < size; ++i)
    if (s[i])
      return i;
  return -1;
}

void __msan_check_mem_is_initialized(const void *x, uptr size) {
  if (!__msan::flags()->report_umrs) return;
  sptr offset = __msan_test_shadow(x, size);
  if (offset < 0)
    return;

  GET_CALLER_PC_BP_SP;
  (void)sp;
  ReportUMRInsideAddressRange(__func__, x, size, offset);
  __msan::PrintWarningWithOrigin(pc, bp,
                                 __msan_get_origin(((char *)x) + offset));
  if (__msan::flags()->halt_on_error) {
    Printf("Exiting\n");
    Die();
  }
}

int __msan_set_poison_in_malloc(int do_poison) {
  int old = flags()->poison_in_malloc;
  flags()->poison_in_malloc = do_poison;
  return old;
}

int  __msan_has_dynamic_component() {
  return msan_running_under_dr;
}

NOINLINE
void __msan_clear_on_return() {
  __msan_param_tls[0] = 0;
}

static void* get_tls_base() {
  u64 p;
  asm("mov %%fs:0, %0"
      : "=r"(p) ::);
  return (void*)p;
}

int __msan_get_retval_tls_offset() {
  // volatile here is needed to avoid UB, because the compiler thinks that we
  // are doing address arithmetics on unrelated pointers, and takes some
  // shortcuts
  volatile sptr retval_tls_p = (sptr)&__msan_retval_tls;
  volatile sptr tls_base_p = (sptr)get_tls_base();
  return retval_tls_p - tls_base_p;
}

int __msan_get_param_tls_offset() {
  // volatile here is needed to avoid UB, because the compiler thinks that we
  // are doing address arithmetics on unrelated pointers, and takes some
  // shortcuts
  volatile sptr param_tls_p = (sptr)&__msan_param_tls;
  volatile sptr tls_base_p = (sptr)get_tls_base();
  return param_tls_p - tls_base_p;
}

void __msan_partial_poison(const void* data, void* shadow, uptr size) {
  internal_memcpy((void*)MEM_TO_SHADOW((uptr)data), shadow, size);
}

void __msan_load_unpoisoned(void *src, uptr size, void *dst) {
  internal_memcpy(dst, src, size);
  __msan_unpoison(dst, size);
}

void __msan_set_origin(const void *a, uptr size, u32 origin) {
  // Origin mapping is 4 bytes per 4 bytes of application memory.
  // Here we extend the range such that its left and right bounds are both
  // 4 byte aligned.
  if (!__msan_get_track_origins()) return;
  uptr x = MEM_TO_ORIGIN((uptr)a);
  uptr beg = x & ~3UL;  // align down.
  uptr end = (x + size + 3) & ~3UL;  // align up.
  u64 origin64 = ((u64)origin << 32) | origin;
  // This is like memset, but the value is 32-bit. We unroll by 2 to write
  // 64 bits at once. May want to unroll further to get 128-bit stores.
  if (beg & 7ULL) {
    *(u32*)beg = origin;
    beg += 4;
  }
  for (uptr addr = beg; addr < (end & ~7UL); addr += 8)
    *(u64*)addr = origin64;
  if (end & 7ULL)
    *(u32*)(end - 4) = origin;
}

// 'descr' is created at compile time and contains '----' in the beginning.
// When we see descr for the first time we replace '----' with a uniq id
// and set the origin to (id | (31-th bit)).
void __msan_set_alloca_origin(void *a, uptr size, const char *descr) {
  __msan_set_alloca_origin4(a, size, descr, 0);
}

void __msan_set_alloca_origin4(void *a, uptr size, const char *descr, uptr pc) {
  static const u32 dash = '-';
  static const u32 first_timer =
      dash + (dash << 8) + (dash << 16) + (dash << 24);
  u32 *id_ptr = (u32*)descr;
  bool print = false;  // internal_strstr(descr + 4, "AllocaTOTest") != 0;
  u32 id = *id_ptr;
  if (id == first_timer) {
    u32 idx = atomic_fetch_add(&NumStackOriginDescrs, 1, memory_order_relaxed);
    CHECK_LT(idx, kNumStackOriginDescrs);
    StackOriginDescr[idx] = descr + 4;
    StackOriginPC[idx] = pc;
    ChainedOriginDepotPut(idx, Origin::kStackRoot, &id);
    *id_ptr = id;
    if (print)
      Printf("First time: idx=%d id=%d %s %p \n", idx, id, descr + 4, pc);
  }
  if (print)
    Printf("__msan_set_alloca_origin: descr=%s id=%x\n", descr + 4, id);
  __msan_set_origin(a, size, Origin(id, 1).raw_id());
}

u32 __msan_chain_origin(u32 id) {
  GET_CALLER_PC_BP_SP;
  (void)sp;
  GET_STORE_STACK_TRACE_PC_BP(pc, bp);
  return ChainOrigin(id, &stack);
}

u32 __msan_get_origin(const void *a) {
  if (!__msan_get_track_origins()) return 0;
  uptr x = (uptr)a;
  uptr aligned = x & ~3ULL;
  uptr origin_ptr = MEM_TO_ORIGIN(aligned);
  return *(u32*)origin_ptr;
}

u32 __msan_get_umr_origin() {
  return __msan_origin_tls;
}

u16 __sanitizer_unaligned_load16(const uu16 *p) {
  __msan_retval_tls[0] = *(uu16 *)MEM_TO_SHADOW((uptr)p);
  if (__msan_get_track_origins())
    __msan_retval_origin_tls = GetOriginIfPoisoned((uptr)p, sizeof(*p));
  return *p;
}
u32 __sanitizer_unaligned_load32(const uu32 *p) {
  __msan_retval_tls[0] = *(uu32 *)MEM_TO_SHADOW((uptr)p);
  if (__msan_get_track_origins())
    __msan_retval_origin_tls = GetOriginIfPoisoned((uptr)p, sizeof(*p));
  return *p;
}
u64 __sanitizer_unaligned_load64(const uu64 *p) {
  __msan_retval_tls[0] = *(uu64 *)MEM_TO_SHADOW((uptr)p);
  if (__msan_get_track_origins())
    __msan_retval_origin_tls = GetOriginIfPoisoned((uptr)p, sizeof(*p));
  return *p;
}
void __sanitizer_unaligned_store16(uu16 *p, u16 x) {
  u16 s = __msan_param_tls[1];
  *(uu16 *)MEM_TO_SHADOW((uptr)p) = s;
  if (s && __msan_get_track_origins())
    if (uu32 o = __msan_param_origin_tls[2])
      SetOriginIfPoisoned((uptr)p, (uptr)&s, sizeof(s), o);
  *p = x;
}
void __sanitizer_unaligned_store32(uu32 *p, u32 x) {
  u32 s = __msan_param_tls[1];
  *(uu32 *)MEM_TO_SHADOW((uptr)p) = s;
  if (s && __msan_get_track_origins())
    if (uu32 o = __msan_param_origin_tls[2])
      SetOriginIfPoisoned((uptr)p, (uptr)&s, sizeof(s), o);
  *p = x;
}
void __sanitizer_unaligned_store64(uu64 *p, u64 x) {
  u64 s = __msan_param_tls[1];
  *(uu64 *)MEM_TO_SHADOW((uptr)p) = s;
  if (s && __msan_get_track_origins())
    if (uu32 o = __msan_param_origin_tls[2])
      SetOriginIfPoisoned((uptr)p, (uptr)&s, sizeof(s), o);
  *p = x;
}

void __msan_set_death_callback(void (*callback)(void)) {
  death_callback = callback;
}

void *__msan_wrap_indirect_call(void *target) {
  return IndirectExternCall(target);
}

void __msan_dr_is_initialized() {
  msan_running_under_dr = true;
}

void __msan_set_indirect_call_wrapper(uptr wrapper) {
  SetIndirectCallWrapper(wrapper);
}

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
const char* __msan_default_options() { return ""; }
}  // extern "C"
#endif

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_print_stack_trace() {
  GET_FATAL_STACK_TRACE_PC_BP(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME());
  stack.Print();
}
}  // extern "C"
