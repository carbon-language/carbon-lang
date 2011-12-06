//===-- asan_rtl.cc ---------------------------------------------*- C++ -*-===//
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
#ifdef __APPLE__
#include "asan_mac.h"
#endif
#include "asan_mapping.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"

#include <new>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ucontext.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
// must not include <setjmp.h> on Linux

#ifndef ASAN_NEEDS_SEGV
# define ASAN_NEEDS_SEGV 1
#endif

namespace __asan {

// -------------------------- Flags ------------------------- {{{1
static const size_t kMallocContextSize = 30;
static int    FLAG_atexit;
bool   FLAG_fast_unwind = true;

size_t FLAG_redzone;  // power of two, >= 32
size_t FLAG_quarantine_size;
int    FLAG_demangle;
bool   FLAG_symbolize;
int    FLAG_v;
int    FLAG_debug;
bool   FLAG_poison_shadow;
int    FLAG_report_globals;
size_t FLAG_malloc_context_size = kMallocContextSize;
uintptr_t FLAG_large_malloc;
bool   FLAG_lazy_shadow;
bool   FLAG_handle_segv;
bool   FLAG_handle_sigill;
bool   FLAG_replace_str;
bool   FLAG_replace_intrin;
bool   FLAG_replace_cfallocator;  // Used on Mac only.
size_t FLAG_max_malloc_fill_size = 0;
bool   FLAG_use_fake_stack;
int    FLAG_exitcode = EXIT_FAILURE;
bool   FLAG_allow_user_poisoning;

// -------------------------- Globals --------------------- {{{1
int asan_inited;
bool asan_init_is_running;

// -------------------------- Interceptors ---------------- {{{1
typedef int (*sigaction_f)(int signum, const struct sigaction *act,
                           struct sigaction *oldact);
typedef sig_t (*signal_f)(int signum, sig_t handler);
typedef void (*longjmp_f)(void *env, int val);
typedef longjmp_f _longjmp_f;
typedef longjmp_f siglongjmp_f;
typedef void (*__cxa_throw_f)(void *, void *, void *);
typedef int (*pthread_create_f)(pthread_t *thread, const pthread_attr_t *attr,
                                void *(*start_routine) (void *), void *arg);
#ifdef __APPLE__
dispatch_async_f_f real_dispatch_async_f;
dispatch_sync_f_f real_dispatch_sync_f;
dispatch_after_f_f real_dispatch_after_f;
dispatch_barrier_async_f_f real_dispatch_barrier_async_f;
dispatch_group_async_f_f real_dispatch_group_async_f;
pthread_workqueue_additem_np_f real_pthread_workqueue_additem_np;
#endif

sigaction_f             real_sigaction;
signal_f                real_signal;
longjmp_f               real_longjmp;
_longjmp_f              real__longjmp;
siglongjmp_f            real_siglongjmp;
__cxa_throw_f           real___cxa_throw;
pthread_create_f        real_pthread_create;

// -------------------------- Misc ---------------- {{{1
void ShowStatsAndAbort() {
  __asan_print_accumulated_stats();
  ASAN_DIE;
}

static void PrintBytes(const char *before, uintptr_t *a) {
  uint8_t *bytes = (uint8_t*)a;
  size_t byte_num = (__WORDSIZE) / 8;
  Printf("%s%p:", before, (uintptr_t)a);
  for (size_t i = 0; i < byte_num; i++) {
    Printf(" %lx%lx", bytes[i] >> 4, bytes[i] & 15);
  }
  Printf("\n");
}

// ---------------------- Thread ------------------------- {{{1
static void *asan_thread_start(void *arg) {
  AsanThread *t= (AsanThread*)arg;
  asanThreadRegistry().SetCurrent(t);
  return t->ThreadStart();
}

// ---------------------- mmap -------------------- {{{1
static void OutOfMemoryMessage(const char *mem_type, size_t size) {
  Report("ERROR: AddressSanitizer failed to allocate "
         "0x%lx (%ld) bytes of %s\n",
         size, size, mem_type);
}

static char *mmap_pages(size_t start_page, size_t n_pages, const char *mem_type,
                        bool abort_on_failure = true) {
  void *res = asan_mmap((void*)start_page, kPageSize * n_pages,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE, 0, 0);
  // Printf("%p => %p\n", (void*)start_page, res);
  char *ch = (char*)res;
  if (res == (void*)-1L && abort_on_failure) {
    OutOfMemoryMessage(mem_type, n_pages * kPageSize);
    ShowStatsAndAbort();
  }
  CHECK(res == (void*)start_page || res == (void*)-1L);
  return ch;
}

// mmap range [beg, end]
static char *mmap_range(uintptr_t beg, uintptr_t end, const char *mem_type) {
  CHECK((beg % kPageSize) == 0);
  CHECK(((end + 1) % kPageSize) == 0);
  // Printf("mmap_range %p %p %ld\n", beg, end, (end - beg) / kPageSize);
  return mmap_pages(beg, (end - beg + 1) / kPageSize, mem_type);
}

// protect range [beg, end]
static void protect_range(uintptr_t beg, uintptr_t end) {
  CHECK((beg % kPageSize) == 0);
  CHECK(((end+1) % kPageSize) == 0);
  // Printf("protect_range %p %p %ld\n", beg, end, (end - beg) / kPageSize);
  void *res = asan_mmap((void*)beg, end - beg + 1,
                   PROT_NONE,
                   MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE, 0, 0);
  CHECK(res == (void*)beg);
}

// ---------------------- LowLevelAllocator ------------- {{{1
void *LowLevelAllocator::Allocate(size_t size) {
  CHECK((size & (size - 1)) == 0 && "size must be a power of two");
  if (allocated_end_ - allocated_current_ < size) {
    size_t size_to_allocate = Max(size, kPageSize);
    allocated_current_ = (char*)asan_mmap(0, size_to_allocate,
                                          PROT_READ | PROT_WRITE,
                                          MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK((allocated_current_ != (char*)-1) && "Can't mmap");
    allocated_end_ = allocated_current_ + size_to_allocate;
  }
  CHECK(allocated_end_ - allocated_current_ >= size);
  void *res = allocated_current_;
  allocated_current_ += size;
  return res;
}

// ---------------------- DescribeAddress -------------------- {{{1
static bool DescribeStackAddress(uintptr_t addr, uintptr_t access_size) {
  AsanThread *t = asanThreadRegistry().FindThreadByStackAddress(addr);
  if (!t) return false;
  const intptr_t kBufSize = 4095;
  char buf[kBufSize];
  uintptr_t offset = 0;
  const char *frame_descr = t->GetFrameNameByAddr(addr, &offset);
  // This string is created by the compiler and has the following form:
  // "FunctioName n alloc_1 alloc_2 ... alloc_n"
  // where alloc_i looks like "offset size len ObjectName ".
  CHECK(frame_descr);
  // Report the function name and the offset.
  const char *name_end = real_strchr(frame_descr, ' ');
  CHECK(name_end);
  buf[0] = 0;
  strncat(buf, frame_descr,
          Min(kBufSize, static_cast<intptr_t>(name_end - frame_descr)));
  Printf("Address %p is located at offset %ld "
         "in frame <%s> of T%d's stack:\n",
         addr, offset, buf, t->tid());
  // Report the number of stack objects.
  char *p;
  size_t n_objects = strtol(name_end, &p, 10);
  CHECK(n_objects > 0);
  Printf("  This frame has %ld object(s):\n", n_objects);
  // Report all objects in this frame.
  for (size_t i = 0; i < n_objects; i++) {
    size_t beg, size;
    intptr_t len;
    beg  = strtol(p, &p, 10);
    size = strtol(p, &p, 10);
    len  = strtol(p, &p, 10);
    if (beg <= 0 || size <= 0 || len < 0 || *p != ' ') {
      Printf("AddressSanitizer can't parse the stack frame descriptor: |%s|\n",
             frame_descr);
      break;
    }
    p++;
    buf[0] = 0;
    strncat(buf, p, Min(kBufSize, len));
    p += len;
    Printf("    [%ld, %ld) '%s'\n", beg, beg + size, buf);
  }
  Printf("HINT: this may be a false positive if your program uses "
         "some custom stack unwind mechanism\n"
         "      (longjmp and C++ exceptions *are* supported)\n");
  t->summary()->Announce();
  return true;
}

__attribute__((noinline))
static void DescribeAddress(uintptr_t addr, uintptr_t access_size) {
  // Check if this is a global.
  if (DescribeAddrIfGlobal(addr))
    return;

  if (DescribeStackAddress(addr, access_size))
    return;

  // finally, check if this is a heap.
  DescribeHeapAddress(addr, access_size);
}

// -------------------------- Run-time entry ------------------- {{{1
void GetPcSpBpAx(void *context,
                 uintptr_t *pc, uintptr_t *sp, uintptr_t *bp, uintptr_t *ax) {
  ucontext_t *ucontext = (ucontext_t*)context;
#ifdef __APPLE__
# if __WORDSIZE == 64
  *pc = ucontext->uc_mcontext->__ss.__rip;
  *bp = ucontext->uc_mcontext->__ss.__rbp;
  *sp = ucontext->uc_mcontext->__ss.__rsp;
  *ax = ucontext->uc_mcontext->__ss.__rax;
# else
  *pc = ucontext->uc_mcontext->__ss.__eip;
  *bp = ucontext->uc_mcontext->__ss.__ebp;
  *sp = ucontext->uc_mcontext->__ss.__esp;
  *ax = ucontext->uc_mcontext->__ss.__eax;
# endif  // __WORDSIZE
#else  // assume linux
# if defined(__arm__)
  *pc = ucontext->uc_mcontext.arm_pc;
  *bp = ucontext->uc_mcontext.arm_fp;
  *sp = ucontext->uc_mcontext.arm_sp;
  *ax = ucontext->uc_mcontext.arm_r0;
# elif __WORDSIZE == 64
  *pc = ucontext->uc_mcontext.gregs[REG_RIP];
  *bp = ucontext->uc_mcontext.gregs[REG_RBP];
  *sp = ucontext->uc_mcontext.gregs[REG_RSP];
  *ax = ucontext->uc_mcontext.gregs[REG_RAX];
# else
  *pc = ucontext->uc_mcontext.gregs[REG_EIP];
  *bp = ucontext->uc_mcontext.gregs[REG_EBP];
  *sp = ucontext->uc_mcontext.gregs[REG_ESP];
  *ax = ucontext->uc_mcontext.gregs[REG_EAX];
# endif  // __WORDSIZE
#endif
}

static void     ASAN_OnSIGSEGV(int, siginfo_t *siginfo, void *context) {
  uintptr_t addr = (uintptr_t)siginfo->si_addr;
  if (AddrIsInShadow(addr) && FLAG_lazy_shadow) {
    // We traped on access to a shadow address. Just map a large chunk around
    // this address.
    const uintptr_t chunk_size = kPageSize << 10;  // 4M
    uintptr_t chunk = addr & ~(chunk_size - 1);
    asan_mmap((void*)chunk, chunk_size,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON | MAP_FIXED, 0, 0);
    return;
  }
  // Write the first message using the bullet-proof write.
  if (13 != asan_write(2, "ASAN:SIGSEGV\n", 13)) ASAN_DIE;
  uintptr_t pc, sp, bp, ax;
  GetPcSpBpAx(context, &pc, &sp, &bp, &ax);
  Report("ERROR: AddressSanitizer crashed on unknown address %p"
         " (pc %p sp %p bp %p ax %p T%d)\n",
         addr, pc, sp, bp, ax,
         asanThreadRegistry().GetCurrentTidOrMinusOne());
  Printf("AddressSanitizer can not provide additional info. ABORTING\n");
  GET_STACK_TRACE_WITH_PC_AND_BP(kStackTraceMax, false, pc, bp);
  stack.PrintStack();
  ShowStatsAndAbort();
}

static void     ASAN_OnSIGILL(int, siginfo_t *siginfo, void *context) {
  // Write the first message using the bullet-proof write.
  if (12 != asan_write(2, "ASAN:SIGILL\n", 12)) ASAN_DIE;
  uintptr_t pc, sp, bp, ax;
  GetPcSpBpAx(context, &pc, &sp, &bp, &ax);

  uintptr_t addr = ax;

  uint8_t *insn = (uint8_t*)pc;
  CHECK(insn[0] == 0x0f && insn[1] == 0x0b);  // ud2
  unsigned access_size_and_type = insn[2] - 0x50;
  CHECK(access_size_and_type < 16);
  bool is_write = access_size_and_type & 8;
  int access_size = 1 << (access_size_and_type & 7);
  __asan_report_error(pc, bp, sp, addr, is_write, access_size);
}

// exported functions
#define ASAN_REPORT_ERROR(type, is_write, size) \
extern "C" void __asan_report_ ## type ## size(uintptr_t addr)   \
  __attribute__((visibility("default")));                        \
extern "C" void __asan_report_ ## type ## size(uintptr_t addr) { \
  GET_BP_PC_SP;                                                  \
  __asan_report_error(pc, bp, sp, addr, is_write, size);  \
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
extern "C"
void __asan_force_interface_symbols() {
  volatile int fake_condition = 0;  // prevent dead condition elimination.
  if (fake_condition) {
    __asan_report_load1(NULL);
    __asan_report_load2(NULL);
    __asan_report_load4(NULL);
    __asan_report_load8(NULL);
    __asan_report_load16(NULL);
    __asan_report_store1(NULL);
    __asan_report_store2(NULL);
    __asan_report_store4(NULL);
    __asan_report_store8(NULL);
    __asan_report_store16(NULL);
    __asan_register_global(0, 0, NULL);
    __asan_register_globals(NULL, 0);
  }
}

// -------------------------- Init ------------------- {{{1
static int64_t IntFlagValue(const char *flags, const char *flag,
                            int64_t default_val) {
  if (!flags) return default_val;
  const char *str = strstr(flags, flag);
  if (!str) return default_val;
  return atoll(str + internal_strlen(flag));
}

static void asan_atexit() {
  Printf("AddressSanitizer exit stats:\n");
  __asan_print_accumulated_stats();
}

void CheckFailed(const char *cond, const char *file, int line) {
  Report("CHECK failed: %s at %s:%d, pthread_self=%p\n",
         cond, file, line, pthread_self());
  PRINT_CURRENT_STACK();
  ShowStatsAndAbort();
}

}  // namespace __asan

// -------------------------- Interceptors ------------------- {{{1
using namespace __asan;  // NOLINT

#define OPERATOR_NEW_BODY \
  GET_STACK_TRACE_HERE_FOR_MALLOC;\
  return asan_memalign(0, size, &stack);

void *operator new(size_t size) throw(std::bad_alloc) { OPERATOR_NEW_BODY; }
void *operator new[](size_t size) throw(std::bad_alloc) { OPERATOR_NEW_BODY; }
void *operator new(size_t size, std::nothrow_t const&) throw()
{ OPERATOR_NEW_BODY; }
void *operator new[](size_t size, std::nothrow_t const&) throw()
{ OPERATOR_NEW_BODY; }

#define OPERATOR_DELETE_BODY \
  GET_STACK_TRACE_HERE_FOR_FREE(ptr);\
  asan_free(ptr, &stack);

void operator delete(void *ptr) throw() { OPERATOR_DELETE_BODY; }
void operator delete[](void *ptr) throw() { OPERATOR_DELETE_BODY; }
void operator delete(void *ptr, std::nothrow_t const&) throw()
{ OPERATOR_DELETE_BODY; }
void operator delete[](void *ptr, std::nothrow_t const&) throw()
{ OPERATOR_DELETE_BODY;}

extern "C"
#ifndef __APPLE__
__attribute__((visibility("default")))
#endif
int WRAP(pthread_create)(pthread_t *thread, const pthread_attr_t *attr,
                         void *(*start_routine) (void *), void *arg) {
  GET_STACK_TRACE_HERE(kStackTraceMax, /*fast_unwind*/false);
  AsanThread *t = (AsanThread*)asan_malloc(sizeof(AsanThread), &stack);
  AsanThread *curr_thread = asanThreadRegistry().GetCurrent();
  CHECK(curr_thread || asanThreadRegistry().IsCurrentThreadDying());
  new(t) AsanThread(asanThreadRegistry().GetCurrentTidOrMinusOne(),
                    start_routine, arg, &stack);
  return real_pthread_create(thread, attr, asan_thread_start, t);
}

static bool MySignal(int signum) {
  if (FLAG_handle_sigill && signum == SIGILL) return true;
  if (FLAG_handle_segv && signum == SIGSEGV) return true;
#ifdef __APPLE__
  if (FLAG_handle_segv && signum == SIGBUS) return true;
#endif
  return false;
}

static void MaybeInstallSigaction(int signum,
                                  void (*handler)(int, siginfo_t *, void *)) {
  if (!MySignal(signum))
    return;
  struct sigaction sigact;
  real_memset(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = handler;
  sigact.sa_flags = SA_SIGINFO;
  CHECK(0 == real_sigaction(signum, &sigact, 0));
}

extern "C"
sig_t WRAP(signal)(int signum, sig_t handler) {
  if (!MySignal(signum)) {
    return real_signal(signum, handler);
  }
  return NULL;
}

extern "C"
int WRAP(sigaction)(int signum, const struct sigaction *act,
                    struct sigaction *oldact) {
  if (!MySignal(signum)) {
    return real_sigaction(signum, act, oldact);
  }
  return 0;
}


static void UnpoisonStackFromHereToTop() {
  int local_stack;
  AsanThread *curr_thread = asanThreadRegistry().GetCurrent();
  CHECK(curr_thread);
  uintptr_t top = curr_thread->stack_top();
  uintptr_t bottom = ((uintptr_t)&local_stack - kPageSize) & ~(kPageSize-1);
  PoisonShadow(bottom, top - bottom, 0);
}

extern "C" void WRAP(longjmp)(void *env, int val) {
  UnpoisonStackFromHereToTop();
  real_longjmp(env, val);
}

extern "C" void WRAP(_longjmp)(void *env, int val) {
  UnpoisonStackFromHereToTop();
  real__longjmp(env, val);
}

extern "C" void WRAP(siglongjmp)(void *env, int val) {
  UnpoisonStackFromHereToTop();
  real_siglongjmp(env, val);
}

extern "C" void __cxa_throw(void *a, void *b, void *c);

#if ASAN_HAS_EXCEPTIONS
extern "C" void WRAP(__cxa_throw)(void *a, void *b, void *c) {
  CHECK(&real___cxa_throw);
  UnpoisonStackFromHereToTop();
  real___cxa_throw(a, b, c);
}
#endif

extern "C" {
// intercept mlock and friends.
// Since asan maps 16T of RAM, mlock is completely unfriendly to asan.
// All functions return 0 (success).
static void MlockIsUnsupported() {
  static bool printed = 0;
  if (printed) return;
  printed = true;
  Printf("INFO: AddressSanitizer ignores mlock/mlockall/munlock/munlockall\n");
}
int mlock(const void *addr, size_t len) {
  MlockIsUnsupported();
  return 0;
}
int munlock(const void *addr, size_t len) {
  MlockIsUnsupported();
  return 0;
}
int mlockall(int flags) {
  MlockIsUnsupported();
  return 0;
}
int munlockall(void) {
  MlockIsUnsupported();
  return 0;
}
}  // extern "C"

// ---------------------- Interface ---------------- {{{1
int __asan_set_error_exit_code(int exit_code) {
  int old = FLAG_exitcode;
  FLAG_exitcode = exit_code;
  return old;
}

void __asan_report_error(uintptr_t pc, uintptr_t bp, uintptr_t sp,
                         uintptr_t addr, bool is_write, size_t access_size) {
  // Do not print more than one report, otherwise they will mix up.
  static int num_calls = 0;
  if (AtomicInc(&num_calls) > 1) return;

  Printf("=================================================================\n");
  const char *bug_descr = "unknown-crash";
  if (AddrIsInMem(addr)) {
    uint8_t *shadow_addr = (uint8_t*)MemToShadow(addr);
    uint8_t shadow_byte = shadow_addr[0];
    if (shadow_byte > 0 && shadow_byte < 128) {
      // we are in the partial right redzone, look at the next shadow byte.
      shadow_byte = shadow_addr[1];
    }
    switch (shadow_byte) {
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

  Report("ERROR: AddressSanitizer %s on address "
         "%p at pc 0x%lx bp 0x%lx sp 0x%lx\n",
         bug_descr, addr, pc, bp, sp);

  Printf("%s of size %d at %p thread T%d\n",
         access_size ? (is_write ? "WRITE" : "READ") : "ACCESS",
         access_size, addr, asanThreadRegistry().GetCurrentTidOrMinusOne());

  if (FLAG_debug) {
    PrintBytes("PC: ", (uintptr_t*)pc);
  }

  GET_STACK_TRACE_WITH_PC_AND_BP(kStackTraceMax,
                                 false,  // FLAG_fast_unwind,
                                 pc, bp);
  stack.PrintStack();

  CHECK(AddrIsInMem(addr));

  DescribeAddress(addr, access_size);

  uintptr_t shadow_addr = MemToShadow(addr);
  Report("ABORTING\n");
  __asan_print_accumulated_stats();
  Printf("Shadow byte and word:\n");
  Printf("  %p: %x\n", shadow_addr, *(unsigned char*)shadow_addr);
  uintptr_t aligned_shadow = shadow_addr & ~(kWordSize - 1);
  PrintBytes("  ", (uintptr_t*)(aligned_shadow));
  Printf("More shadow bytes:\n");
  PrintBytes("  ", (uintptr_t*)(aligned_shadow-4*kWordSize));
  PrintBytes("  ", (uintptr_t*)(aligned_shadow-3*kWordSize));
  PrintBytes("  ", (uintptr_t*)(aligned_shadow-2*kWordSize));
  PrintBytes("  ", (uintptr_t*)(aligned_shadow-1*kWordSize));
  PrintBytes("=>", (uintptr_t*)(aligned_shadow+0*kWordSize));
  PrintBytes("  ", (uintptr_t*)(aligned_shadow+1*kWordSize));
  PrintBytes("  ", (uintptr_t*)(aligned_shadow+2*kWordSize));
  PrintBytes("  ", (uintptr_t*)(aligned_shadow+3*kWordSize));
  PrintBytes("  ", (uintptr_t*)(aligned_shadow+4*kWordSize));
  ASAN_DIE;
}

void __asan_init() {
  if (asan_inited) return;
  asan_init_is_running = true;

  // Make sure we are not statically linked.
  AsanDoesNotSupportStaticLinkage();

  // flags
  const char *options = getenv("ASAN_OPTIONS");
  FLAG_malloc_context_size =
      IntFlagValue(options, "malloc_context_size=", kMallocContextSize);
  CHECK(FLAG_malloc_context_size <= kMallocContextSize);

  FLAG_max_malloc_fill_size =
      IntFlagValue(options, "max_malloc_fill_size=", 0);

  FLAG_v = IntFlagValue(options, "verbosity=", 0);

  FLAG_redzone = IntFlagValue(options, "redzone=", 128);
  CHECK(FLAG_redzone >= 32);
  CHECK((FLAG_redzone & (FLAG_redzone - 1)) == 0);

  FLAG_atexit = IntFlagValue(options, "atexit=", 0);
  FLAG_poison_shadow = IntFlagValue(options, "poison_shadow=", 1);
  FLAG_report_globals = IntFlagValue(options, "report_globals=", 1);
  FLAG_lazy_shadow = IntFlagValue(options, "lazy_shadow=", 0);
  FLAG_handle_segv = IntFlagValue(options, "handle_segv=",
                                         ASAN_NEEDS_SEGV);
  FLAG_handle_sigill = IntFlagValue(options, "handle_sigill=", 0);
  FLAG_symbolize = IntFlagValue(options, "symbolize=", 1);
  FLAG_demangle = IntFlagValue(options, "demangle=", 1);
  FLAG_debug = IntFlagValue(options, "debug=", 0);
  FLAG_replace_cfallocator = IntFlagValue(options, "replace_cfallocator=", 1);
  FLAG_fast_unwind = IntFlagValue(options, "fast_unwind=", 1);
  FLAG_replace_str = IntFlagValue(options, "replace_str=", 1);
  FLAG_replace_intrin = IntFlagValue(options, "replace_intrin=", 0);
  FLAG_use_fake_stack = IntFlagValue(options, "use_fake_stack=", 1);
  FLAG_exitcode = IntFlagValue(options, "exitcode=", EXIT_FAILURE);
  FLAG_allow_user_poisoning = IntFlagValue(options,
                                           "allow_user_poisoning=", 1);

  if (FLAG_atexit) {
    atexit(asan_atexit);
  }

  FLAG_quarantine_size =
      IntFlagValue(options, "quarantine_size=", 1UL << 28);

  // interceptors
  InitializeAsanInterceptors();

  ReplaceSystemMalloc();

  INTERCEPT_FUNCTION(sigaction);
  INTERCEPT_FUNCTION(signal);
  INTERCEPT_FUNCTION(longjmp);
  INTERCEPT_FUNCTION(_longjmp);
  INTERCEPT_FUNCTION_IF_EXISTS(__cxa_throw);
  INTERCEPT_FUNCTION(pthread_create);
#ifdef __APPLE__
  INTERCEPT_FUNCTION(dispatch_async_f);
  INTERCEPT_FUNCTION(dispatch_sync_f);
  INTERCEPT_FUNCTION(dispatch_after_f);
  INTERCEPT_FUNCTION(dispatch_barrier_async_f);
  INTERCEPT_FUNCTION(dispatch_group_async_f);
  // We don't need to intercept pthread_workqueue_additem_np() to support the
  // libdispatch API, but it helps us to debug the unsupported functions. Let's
  // intercept it only during verbose runs.
  if (FLAG_v >= 2) {
    INTERCEPT_FUNCTION(pthread_workqueue_additem_np);
  }
#else
  // On Darwin siglongjmp tailcalls longjmp, so we don't want to intercept it
  // there.
  INTERCEPT_FUNCTION(siglongjmp);
#endif

  MaybeInstallSigaction(SIGSEGV, ASAN_OnSIGSEGV);
  MaybeInstallSigaction(SIGBUS, ASAN_OnSIGSEGV);
  MaybeInstallSigaction(SIGILL, ASAN_OnSIGILL);

  if (FLAG_v) {
    Printf("|| `[%p, %p]` || HighMem    ||\n", kHighMemBeg, kHighMemEnd);
    Printf("|| `[%p, %p]` || HighShadow ||\n",
           kHighShadowBeg, kHighShadowEnd);
    Printf("|| `[%p, %p]` || ShadowGap  ||\n",
           kShadowGapBeg, kShadowGapEnd);
    Printf("|| `[%p, %p]` || LowShadow  ||\n",
           kLowShadowBeg, kLowShadowEnd);
    Printf("|| `[%p, %p]` || LowMem     ||\n", kLowMemBeg, kLowMemEnd);
    Printf("MemToShadow(shadow): %p %p %p %p\n",
           MEM_TO_SHADOW(kLowShadowBeg),
           MEM_TO_SHADOW(kLowShadowEnd),
           MEM_TO_SHADOW(kHighShadowBeg),
           MEM_TO_SHADOW(kHighShadowEnd));
    Printf("red_zone=%ld\n", FLAG_redzone);
    Printf("malloc_context_size=%ld\n", (int)FLAG_malloc_context_size);
    Printf("fast_unwind=%d\n", (int)FLAG_fast_unwind);

    Printf("SHADOW_SCALE: %lx\n", SHADOW_SCALE);
    Printf("SHADOW_GRANULARITY: %lx\n", SHADOW_GRANULARITY);
    Printf("SHADOW_OFFSET: %lx\n", SHADOW_OFFSET);
    CHECK(SHADOW_SCALE >= 3 && SHADOW_SCALE <= 7);
  }

  if (__WORDSIZE == 64) {
    // Disable core dumper -- it makes little sense to dump 16T+ core.
    struct rlimit nocore;
    nocore.rlim_cur = 0;
    nocore.rlim_max = 0;
    setrlimit(RLIMIT_CORE, &nocore);
  }

  {
    if (!FLAG_lazy_shadow) {
      if (kLowShadowBeg != kLowShadowEnd) {
        // mmap the low shadow plus one page.
        mmap_range(kLowShadowBeg - kPageSize, kLowShadowEnd, "LowShadow");
      }
      // mmap the high shadow.
      mmap_range(kHighShadowBeg, kHighShadowEnd, "HighShadow");
    }
    // protect the gap
    protect_range(kShadowGapBeg, kShadowGapEnd);
  }

  // On Linux AsanThread::ThreadStart() calls malloc() that's why asan_inited
  // should be set to 1 prior to initializing the threads.
  asan_inited = 1;
  asan_init_is_running = false;

  asanThreadRegistry().Init();
  asanThreadRegistry().GetMain()->ThreadStart();
  __asan_force_interface_symbols();  // no-op.

  if (FLAG_v) {
    Report("AddressSanitizer Init done\n");
  }
}
