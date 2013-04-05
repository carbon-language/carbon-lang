//===-- asan_report.cc ----------------------------------------------------===//
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
// This file contains error reporting code.
//===----------------------------------------------------------------------===//
#include "asan_flags.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

namespace __asan {

// -------------------- User-specified callbacks ----------------- {{{1
static void (*error_report_callback)(const char*);
static char *error_message_buffer = 0;
static uptr error_message_buffer_pos = 0;
static uptr error_message_buffer_size = 0;

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

// ---------------------- Decorator ------------------------------ {{{1
bool PrintsToTtyCached() {
  static int cached = 0;
  static bool prints_to_tty;
  if (!cached) {  // Ok wrt threads since we are printing only from one thread.
    prints_to_tty = PrintsToTty();
    cached = 1;
  }
  return prints_to_tty;
}
class Decorator: private __sanitizer::AnsiColorDecorator {
 public:
  Decorator() : __sanitizer::AnsiColorDecorator(PrintsToTtyCached()) { }
  const char *Warning()    { return Red(); }
  const char *EndWarning() { return Default(); }
  const char *Access()     { return Blue(); }
  const char *EndAccess()  { return Default(); }
  const char *Location()   { return Green(); }
  const char *EndLocation() { return Default(); }
  const char *Allocation()  { return Magenta(); }
  const char *EndAllocation()  { return Default(); }

  const char *ShadowByte(u8 byte) {
    switch (byte) {
      case kAsanHeapLeftRedzoneMagic:
      case kAsanHeapRightRedzoneMagic:
        return Red();
      case kAsanHeapFreeMagic:
        return Magenta();
      case kAsanStackLeftRedzoneMagic:
      case kAsanStackMidRedzoneMagic:
      case kAsanStackRightRedzoneMagic:
      case kAsanStackPartialRedzoneMagic:
        return Red();
      case kAsanStackAfterReturnMagic:
        return Magenta();
      case kAsanInitializationOrderMagic:
        return Cyan();
      case kAsanUserPoisonedMemoryMagic:
        return Blue();
      case kAsanStackUseAfterScopeMagic:
        return Magenta();
      case kAsanGlobalRedzoneMagic:
        return Red();
      case kAsanInternalHeapMagic:
        return Yellow();
      default:
        return Default();
    }
  }
  const char *EndShadowByte() { return Default(); }
};

// ---------------------- Helper functions ----------------------- {{{1

static void PrintShadowByte(const char *before, u8 byte,
                            const char *after = "\n") {
  Decorator d;
  Printf("%s%s%x%x%s%s", before,
         d.ShadowByte(byte), byte >> 4, byte & 15, d.EndShadowByte(), after);
}

static void PrintShadowBytes(const char *before, u8 *bytes,
                             u8 *guilty, uptr n) {
  Decorator d;
  if (before)
    Printf("%s%p:", before, bytes);
  for (uptr i = 0; i < n; i++) {
    u8 *p = bytes + i;
    const char *before = p == guilty ? "[" :
        p - 1 == guilty ? "" : " ";
    const char *after = p == guilty ? "]" : "";
    PrintShadowByte(before, *p, after);
  }
  Printf("\n");
}

static void PrintLegend() {
  Printf("Shadow byte legend (one shadow byte represents %d "
         "application bytes):\n", (int)SHADOW_GRANULARITY);
  PrintShadowByte("  Addressable:           ", 0);
  Printf("  Partially addressable: ");
  for (uptr i = 1; i < SHADOW_GRANULARITY; i++)
    PrintShadowByte("", i, " ");
  Printf("\n");
  PrintShadowByte("  Heap left redzone:     ", kAsanHeapLeftRedzoneMagic);
  PrintShadowByte("  Heap righ redzone:     ", kAsanHeapRightRedzoneMagic);
  PrintShadowByte("  Freed Heap region:     ", kAsanHeapFreeMagic);
  PrintShadowByte("  Stack left redzone:    ", kAsanStackLeftRedzoneMagic);
  PrintShadowByte("  Stack mid redzone:     ", kAsanStackMidRedzoneMagic);
  PrintShadowByte("  Stack right redzone:   ", kAsanStackRightRedzoneMagic);
  PrintShadowByte("  Stack partial redzone: ", kAsanStackPartialRedzoneMagic);
  PrintShadowByte("  Stack after return:    ", kAsanStackAfterReturnMagic);
  PrintShadowByte("  Stack use after scope: ", kAsanStackUseAfterScopeMagic);
  PrintShadowByte("  Global redzone:        ", kAsanGlobalRedzoneMagic);
  PrintShadowByte("  Global init order:     ", kAsanInitializationOrderMagic);
  PrintShadowByte("  Poisoned by user:      ", kAsanUserPoisonedMemoryMagic);
  PrintShadowByte("  ASan internal:         ", kAsanInternalHeapMagic);
}

static void PrintShadowMemoryForAddress(uptr addr) {
  if (!AddrIsInMem(addr))
    return;
  uptr shadow_addr = MemToShadow(addr);
  const uptr n_bytes_per_row = 16;
  uptr aligned_shadow = shadow_addr & ~(n_bytes_per_row - 1);
  Printf("Shadow bytes around the buggy address:\n");
  for (int i = -5; i <= 5; i++) {
    const char *prefix = (i == 0) ? "=>" : "  ";
    PrintShadowBytes(prefix,
                     (u8*)(aligned_shadow + i * n_bytes_per_row),
                     (u8*)shadow_addr, n_bytes_per_row);
  }
  if (flags()->print_legend)
    PrintLegend();
}

static void PrintZoneForPointer(uptr ptr, uptr zone_ptr,
                                const char *zone_name) {
  if (zone_ptr) {
    if (zone_name) {
      Printf("malloc_zone_from_ptr(%p) = %p, which is %s\n",
                 ptr, zone_ptr, zone_name);
    } else {
      Printf("malloc_zone_from_ptr(%p) = %p, which doesn't have a name\n",
                 ptr, zone_ptr);
    }
  } else {
    Printf("malloc_zone_from_ptr(%p) = 0\n", ptr);
  }
}

// ---------------------- Address Descriptions ------------------- {{{1

static bool IsASCII(unsigned char c) {
  return /*0x00 <= c &&*/ c <= 0x7F;
}

static const char *MaybeDemangleGlobalName(const char *name) {
  // We can spoil names of globals with C linkage, so use an heuristic
  // approach to check if the name should be demangled.
  return (name[0] == '_' && name[1] == 'Z') ? Demangle(name) : name;
}

// Check if the global is a zero-terminated ASCII string. If so, print it.
static void PrintGlobalNameIfASCII(const __asan_global &g) {
  for (uptr p = g.beg; p < g.beg + g.size - 1; p++) {
    unsigned char c = *(unsigned char*)p;
    if (c == '\0' || !IsASCII(c)) return;
  }
  if (*(char*)(g.beg + g.size - 1) != '\0') return;
  Printf("  '%s' is ascii string '%s'\n",
         MaybeDemangleGlobalName(g.name), (char*)g.beg);
}

bool DescribeAddressRelativeToGlobal(uptr addr, uptr size,
                                     const __asan_global &g) {
  static const uptr kMinimalDistanceFromAnotherGlobal = 64;
  if (addr <= g.beg - kMinimalDistanceFromAnotherGlobal) return false;
  if (addr >= g.beg + g.size_with_redzone) return false;
  Decorator d;
  Printf("%s", d.Location());
  if (addr < g.beg) {
    Printf("%p is located %zd bytes to the left", (void*)addr, g.beg - addr);
  } else if (addr + size > g.beg + g.size) {
    if (addr < g.beg + g.size)
      addr = g.beg + g.size;
    Printf("%p is located %zd bytes to the right", (void*)addr,
           addr - (g.beg + g.size));
  } else {
    // Can it happen?
    Printf("%p is located %zd bytes inside", (void*)addr, addr - g.beg);
  }
  Printf(" of global variable '%s' from '%s' (0x%zx) of size %zu\n",
             MaybeDemangleGlobalName(g.name), g.module_name, g.beg, g.size);
  Printf("%s", d.EndLocation());
  PrintGlobalNameIfASCII(g);
  return true;
}

bool DescribeAddressIfShadow(uptr addr) {
  if (AddrIsInMem(addr))
    return false;
  static const char kAddrInShadowReport[] =
      "Address %p is located in the %s.\n";
  if (AddrIsInShadowGap(addr)) {
    Printf(kAddrInShadowReport, addr, "shadow gap area");
    return true;
  }
  if (AddrIsInHighShadow(addr)) {
    Printf(kAddrInShadowReport, addr, "high shadow area");
    return true;
  }
  if (AddrIsInLowShadow(addr)) {
    Printf(kAddrInShadowReport, addr, "low shadow area");
    return true;
  }
  CHECK(0 && "Address is not in memory and not in shadow?");
  return false;
}

// Return " (thread_name) " or an empty string if the name is empty.
const char *ThreadNameWithParenthesis(AsanThreadContext *t, char buff[],
                                      uptr buff_len) {
  const char *name = t->name;
  if (name[0] == '\0') return "";
  buff[0] = 0;
  internal_strncat(buff, " (", 3);
  internal_strncat(buff, name, buff_len - 4);
  internal_strncat(buff, ")", 2);
  return buff;
}

const char *ThreadNameWithParenthesis(u32 tid, char buff[],
                                      uptr buff_len) {
  if (tid == kInvalidTid) return "";
  asanThreadRegistry().CheckLocked();
  AsanThreadContext *t = GetThreadContextByTidLocked(tid);
  return ThreadNameWithParenthesis(t, buff, buff_len);
}

bool DescribeAddressIfStack(uptr addr, uptr access_size) {
  AsanThread *t = FindThreadByStackAddress(addr);
  if (!t) return false;
  const sptr kBufSize = 4095;
  char buf[kBufSize];
  uptr offset = 0;
  uptr frame_pc = 0;
  char tname[128];
  const char *frame_descr = t->GetFrameNameByAddr(addr, &offset, &frame_pc);
  // This string is created by the compiler and has the following form:
  // "n alloc_1 alloc_2 ... alloc_n"
  // where alloc_i looks like "offset size len ObjectName ".
  CHECK(frame_descr);
  Decorator d;
  Printf("%s", d.Location());
  Printf("Address %p is located in stack of thread T%d%s "
         "at offset %zu in frame\n",
         addr, t->tid(),
         ThreadNameWithParenthesis(t->tid(), tname, sizeof(tname)),
         offset);
  // Now we print the frame where the alloca has happened.
  // We print this frame as a stack trace with one element.
  // The symbolizer may print more than one frame if inlining was involved.
  // The frame numbers may be different than those in the stack trace printed
  // previously. That's unfortunate, but I have no better solution,
  // especially given that the alloca may be from entirely different place
  // (e.g. use-after-scope, or different thread's stack).
  StackTrace alloca_stack;
  alloca_stack.trace[0] = frame_pc + 16;
  alloca_stack.size = 1;
  Printf("%s", d.EndLocation());
  PrintStack(&alloca_stack);
  // Report the number of stack objects.
  char *p;
  uptr n_objects = internal_simple_strtoll(frame_descr, &p, 10);
  CHECK(n_objects > 0);
  Printf("  This frame has %zu object(s):\n", n_objects);
  // Report all objects in this frame.
  for (uptr i = 0; i < n_objects; i++) {
    uptr beg, size;
    sptr len;
    beg  = internal_simple_strtoll(p, &p, 10);
    size = internal_simple_strtoll(p, &p, 10);
    len  = internal_simple_strtoll(p, &p, 10);
    if (beg <= 0 || size <= 0 || len < 0 || *p != ' ') {
      Printf("AddressSanitizer can't parse the stack frame "
                 "descriptor: |%s|\n", frame_descr);
      break;
    }
    p++;
    buf[0] = 0;
    internal_strncat(buf, p, Min(kBufSize, len));
    p += len;
    Printf("    [%zu, %zu) '%s'\n", beg, beg + size, buf);
  }
  Printf("HINT: this may be a false positive if your program uses "
             "some custom stack unwind mechanism or swapcontext\n"
             "      (longjmp and C++ exceptions *are* supported)\n");
  DescribeThread(t->context());
  return true;
}

static void DescribeAccessToHeapChunk(AsanChunkView chunk, uptr addr,
                                      uptr access_size) {
  sptr offset;
  Decorator d;
  Printf("%s", d.Location());
  if (chunk.AddrIsAtLeft(addr, access_size, &offset)) {
    Printf("%p is located %zd bytes to the left of", (void*)addr, offset);
  } else if (chunk.AddrIsAtRight(addr, access_size, &offset)) {
    if (offset < 0) {
      addr -= offset;
      offset = 0;
    }
    Printf("%p is located %zd bytes to the right of", (void*)addr, offset);
  } else if (chunk.AddrIsInside(addr, access_size, &offset)) {
    Printf("%p is located %zd bytes inside of", (void*)addr, offset);
  } else {
    Printf("%p is located somewhere around (this is AddressSanitizer bug!)",
           (void*)addr);
  }
  Printf(" %zu-byte region [%p,%p)\n", chunk.UsedSize(),
         (void*)(chunk.Beg()), (void*)(chunk.End()));
  Printf("%s", d.EndLocation());
}

void DescribeHeapAddress(uptr addr, uptr access_size) {
  AsanChunkView chunk = FindHeapChunkByAddress(addr);
  if (!chunk.IsValid()) return;
  DescribeAccessToHeapChunk(chunk, addr, access_size);
  CHECK(chunk.AllocTid() != kInvalidTid);
  asanThreadRegistry().CheckLocked();
  AsanThreadContext *alloc_thread =
      GetThreadContextByTidLocked(chunk.AllocTid());
  StackTrace alloc_stack;
  chunk.GetAllocStack(&alloc_stack);
  AsanThread *t = GetCurrentThread();
  CHECK(t);
  char tname[128];
  Decorator d;
  if (chunk.FreeTid() != kInvalidTid) {
    AsanThreadContext *free_thread =
        GetThreadContextByTidLocked(chunk.FreeTid());
    Printf("%sfreed by thread T%d%s here:%s\n", d.Allocation(),
           free_thread->tid,
           ThreadNameWithParenthesis(free_thread, tname, sizeof(tname)),
           d.EndAllocation());
    StackTrace free_stack;
    chunk.GetFreeStack(&free_stack);
    PrintStack(&free_stack);
    Printf("%spreviously allocated by thread T%d%s here:%s\n",
           d.Allocation(), alloc_thread->tid,
           ThreadNameWithParenthesis(alloc_thread, tname, sizeof(tname)),
           d.EndAllocation());
    PrintStack(&alloc_stack);
    DescribeThread(t->context());
    DescribeThread(free_thread);
    DescribeThread(alloc_thread);
  } else {
    Printf("%sallocated by thread T%d%s here:%s\n", d.Allocation(),
           alloc_thread->tid,
           ThreadNameWithParenthesis(alloc_thread, tname, sizeof(tname)),
           d.EndAllocation());
    PrintStack(&alloc_stack);
    DescribeThread(t->context());
    DescribeThread(alloc_thread);
  }
}

void DescribeAddress(uptr addr, uptr access_size) {
  // Check if this is shadow or shadow gap.
  if (DescribeAddressIfShadow(addr))
    return;
  CHECK(AddrIsInMem(addr));
  if (DescribeAddressIfGlobal(addr, access_size))
    return;
  if (DescribeAddressIfStack(addr, access_size))
    return;
  // Assume it is a heap address.
  DescribeHeapAddress(addr, access_size);
}

// ------------------- Thread description -------------------- {{{1

void DescribeThread(AsanThreadContext *context) {
  CHECK(context);
  asanThreadRegistry().CheckLocked();
  // No need to announce the main thread.
  if (context->tid == 0 || context->announced) {
    return;
  }
  context->announced = true;
  char tname[128];
  Printf("Thread T%d%s", context->tid,
         ThreadNameWithParenthesis(context->tid, tname, sizeof(tname)));
  Printf(" created by T%d%s here:\n",
         context->parent_tid,
         ThreadNameWithParenthesis(context->parent_tid,
                                   tname, sizeof(tname)));
  PrintStack(&context->stack);
  // Recursively described parent thread if needed.
  if (flags()->print_full_thread_history) {
    AsanThreadContext *parent_context =
        GetThreadContextByTidLocked(context->parent_tid);
    DescribeThread(parent_context);
  }
}

// -------------------- Different kinds of reports ----------------- {{{1

// Use ScopedInErrorReport to run common actions just before and
// immediately after printing error report.
class ScopedInErrorReport {
 public:
  ScopedInErrorReport() {
    static atomic_uint32_t num_calls;
    static u32 reporting_thread_tid;
    if (atomic_fetch_add(&num_calls, 1, memory_order_relaxed) != 0) {
      // Do not print more than one report, otherwise they will mix up.
      // Error reporting functions shouldn't return at this situation, as
      // they are defined as no-return.
      Report("AddressSanitizer: while reporting a bug found another one."
                 "Ignoring.\n");
      u32 current_tid = GetCurrentTidOrInvalid();
      if (current_tid != reporting_thread_tid) {
        // ASan found two bugs in different threads simultaneously. Sleep
        // long enough to make sure that the thread which started to print
        // an error report will finish doing it.
        SleepForSeconds(Max(100, flags()->sleep_before_dying + 1));
      }
      // If we're still not dead for some reason, use raw _exit() instead of
      // Die() to bypass any additional checks.
      internal__exit(flags()->exitcode);
    }
    ASAN_ON_ERROR();
    // Make sure the registry and sanitizer report mutexes are locked while
    // we're printing an error report.
    // We can lock them only here to avoid self-deadlock in case of
    // recursive reports.
    asanThreadRegistry().Lock();
    CommonSanitizerReportMutex.Lock();
    reporting_thread_tid = GetCurrentTidOrInvalid();
    Printf("===================================================="
           "=============\n");
    if (reporting_thread_tid != kInvalidTid) {
      // We started reporting an error message. Stop using the fake stack
      // in case we call an instrumented function from a symbolizer.
      AsanThread *curr_thread = GetCurrentThread();
      CHECK(curr_thread);
      curr_thread->fake_stack().StopUsingFakeStack();
    }
  }
  // Destructor is NORETURN, as functions that report errors are.
  NORETURN ~ScopedInErrorReport() {
    // Make sure the current thread is announced.
    AsanThread *curr_thread = GetCurrentThread();
    if (curr_thread) {
      DescribeThread(curr_thread->context());
    }
    // Print memory stats.
    if (flags()->print_stats)
      __asan_print_accumulated_stats();
    if (error_report_callback) {
      error_report_callback(error_message_buffer);
    }
    Report("ABORTING\n");
    Die();
  }
};

static void ReportSummary(const char *error_type, StackTrace *stack) {
  if (!stack->size) return;
  if (IsSymbolizerAvailable()) {
    AddressInfo ai;
    // Currently, we include the first stack frame into the report summary.
    // Maybe sometimes we need to choose another frame (e.g. skip memcpy/etc).
    SymbolizeCode(stack->trace[0], &ai, 1);
    ReportErrorSummary(error_type,
                       StripPathPrefix(ai.file, flags()->strip_path_prefix),
                       ai.line, ai.function);
  }
  // FIXME: do we need to print anything at all if there is no symbolizer?
}

void ReportSIGSEGV(uptr pc, uptr sp, uptr bp, uptr addr) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: SEGV on unknown address %p"
             " (pc %p sp %p bp %p T%d)\n",
             (void*)addr, (void*)pc, (void*)sp, (void*)bp,
             GetCurrentTidOrInvalid());
  Printf("%s", d.EndWarning());
  Printf("AddressSanitizer can not provide additional info.\n");
  GET_STACK_TRACE_FATAL(pc, bp);
  PrintStack(&stack);
  ReportSummary("SEGV", &stack);
}

void ReportDoubleFree(uptr addr, StackTrace *stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  char tname[128];
  u32 curr_tid = GetCurrentTidOrInvalid();
  Report("ERROR: AddressSanitizer: attempting double-free on %p in "
         "thread T%d%s:\n",
         addr, curr_tid,
         ThreadNameWithParenthesis(curr_tid, tname, sizeof(tname)));

  Printf("%s", d.EndWarning());
  PrintStack(stack);
  DescribeHeapAddress(addr, 1);
  ReportSummary("double-free", stack);
}

void ReportFreeNotMalloced(uptr addr, StackTrace *stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  char tname[128];
  u32 curr_tid = GetCurrentTidOrInvalid();
  Report("ERROR: AddressSanitizer: attempting free on address "
             "which was not malloc()-ed: %p in thread T%d%s\n", addr,
         curr_tid, ThreadNameWithParenthesis(curr_tid, tname, sizeof(tname)));
  Printf("%s", d.EndWarning());
  PrintStack(stack);
  DescribeHeapAddress(addr, 1);
  ReportSummary("bad-free", stack);
}

void ReportAllocTypeMismatch(uptr addr, StackTrace *stack,
                             AllocType alloc_type,
                             AllocType dealloc_type) {
  static const char *alloc_names[] =
    {"INVALID", "malloc", "operator new", "operator new []"};
  static const char *dealloc_names[] =
    {"INVALID", "free", "operator delete", "operator delete []"};
  CHECK_NE(alloc_type, dealloc_type);
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: alloc-dealloc-mismatch (%s vs %s) on %p\n",
        alloc_names[alloc_type], dealloc_names[dealloc_type], addr);
  Printf("%s", d.EndWarning());
  PrintStack(stack);
  DescribeHeapAddress(addr, 1);
  ReportSummary("alloc-dealloc-mismatch", stack);
  Report("HINT: if you don't care about these warnings you may set "
         "ASAN_OPTIONS=alloc_dealloc_mismatch=0\n");
}

void ReportMallocUsableSizeNotOwned(uptr addr, StackTrace *stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: attempting to call "
             "malloc_usable_size() for pointer which is "
             "not owned: %p\n", addr);
  Printf("%s", d.EndWarning());
  PrintStack(stack);
  DescribeHeapAddress(addr, 1);
  ReportSummary("bad-malloc_usable_size", stack);
}

void ReportAsanGetAllocatedSizeNotOwned(uptr addr, StackTrace *stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: attempting to call "
             "__asan_get_allocated_size() for pointer which is "
             "not owned: %p\n", addr);
  Printf("%s", d.EndWarning());
  PrintStack(stack);
  DescribeHeapAddress(addr, 1);
  ReportSummary("bad-__asan_get_allocated_size", stack);
}

void ReportStringFunctionMemoryRangesOverlap(
    const char *function, const char *offset1, uptr length1,
    const char *offset2, uptr length2, StackTrace *stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  char bug_type[100];
  internal_snprintf(bug_type, sizeof(bug_type), "%s-param-overlap", function);
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: %s: "
             "memory ranges [%p,%p) and [%p, %p) overlap\n", \
             bug_type, offset1, offset1 + length1, offset2, offset2 + length2);
  Printf("%s", d.EndWarning());
  PrintStack(stack);
  DescribeAddress((uptr)offset1, length1);
  DescribeAddress((uptr)offset2, length2);
  ReportSummary(bug_type, stack);
}

// ----------------------- Mac-specific reports ----------------- {{{1

void WarnMacFreeUnallocated(
    uptr addr, uptr zone_ptr, const char *zone_name, StackTrace *stack) {
  // Just print a warning here.
  Printf("free_common(%p) -- attempting to free unallocated memory.\n"
             "AddressSanitizer is ignoring this error on Mac OS now.\n",
             addr);
  PrintZoneForPointer(addr, zone_ptr, zone_name);
  PrintStack(stack);
  DescribeHeapAddress(addr, 1);
}

void ReportMacMzReallocUnknown(
    uptr addr, uptr zone_ptr, const char *zone_name, StackTrace *stack) {
  ScopedInErrorReport in_report;
  Printf("mz_realloc(%p) -- attempting to realloc unallocated memory.\n"
             "This is an unrecoverable problem, exiting now.\n",
             addr);
  PrintZoneForPointer(addr, zone_ptr, zone_name);
  PrintStack(stack);
  DescribeHeapAddress(addr, 1);
}

void ReportMacCfReallocUnknown(
    uptr addr, uptr zone_ptr, const char *zone_name, StackTrace *stack) {
  ScopedInErrorReport in_report;
  Printf("cf_realloc(%p) -- attempting to realloc unallocated memory.\n"
             "This is an unrecoverable problem, exiting now.\n",
             addr);
  PrintZoneForPointer(addr, zone_ptr, zone_name);
  PrintStack(stack);
  DescribeHeapAddress(addr, 1);
}

}  // namespace __asan

// --------------------------- Interface --------------------- {{{1
using namespace __asan;  // NOLINT

void __asan_report_error(uptr pc, uptr bp, uptr sp,
                         uptr addr, bool is_write, uptr access_size) {
  ScopedInErrorReport in_report;

  // Determine the error type.
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
      case kAsanInitializationOrderMagic:
        bug_descr = "initialization-order-fiasco";
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
      case kAsanStackUseAfterScopeMagic:
        bug_descr = "stack-use-after-scope";
        break;
      case kAsanGlobalRedzoneMagic:
        bug_descr = "global-buffer-overflow";
        break;
    }
  }
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: %s on address "
             "%p at pc 0x%zx bp 0x%zx sp 0x%zx\n",
             bug_descr, (void*)addr, pc, bp, sp);
  Printf("%s", d.EndWarning());

  u32 curr_tid = GetCurrentTidOrInvalid();
  char tname[128];
  Printf("%s%s of size %zu at %p thread T%d%s%s\n",
         d.Access(),
         access_size ? (is_write ? "WRITE" : "READ") : "ACCESS",
         access_size, (void*)addr, curr_tid,
         ThreadNameWithParenthesis(curr_tid, tname, sizeof(tname)),
         d.EndAccess());

  GET_STACK_TRACE_FATAL(pc, bp);
  PrintStack(&stack);

  DescribeAddress(addr, access_size);
  ReportSummary(bug_descr, &stack);
  PrintShadowMemoryForAddress(addr);
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

void __asan_describe_address(uptr addr) {
  DescribeAddress(addr, 1);
}

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
// Provide default implementation of __asan_on_error that does nothing
// and may be overriden by user.
SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE NOINLINE
void __asan_on_error() {}
#endif
