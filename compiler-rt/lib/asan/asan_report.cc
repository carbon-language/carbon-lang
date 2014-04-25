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
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
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
class Decorator: public __sanitizer::SanitizerCommonDecorator {
 public:
  Decorator() : SanitizerCommonDecorator() { }
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
      case kAsanContiguousContainerOOBMagic:
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

static void PrintShadowByte(InternalScopedString *str, const char *before,
                            u8 byte, const char *after = "\n") {
  Decorator d;
  str->append("%s%s%x%x%s%s", before, d.ShadowByte(byte), byte >> 4, byte & 15,
              d.EndShadowByte(), after);
}

static void PrintShadowBytes(InternalScopedString *str, const char *before,
                             u8 *bytes, u8 *guilty, uptr n) {
  Decorator d;
  if (before) str->append("%s%p:", before, bytes);
  for (uptr i = 0; i < n; i++) {
    u8 *p = bytes + i;
    const char *before =
        p == guilty ? "[" : (p - 1 == guilty && i != 0) ? "" : " ";
    const char *after = p == guilty ? "]" : "";
    PrintShadowByte(str, before, *p, after);
  }
  str->append("\n");
}

static void PrintLegend(InternalScopedString *str) {
  str->append(
      "Shadow byte legend (one shadow byte represents %d "
      "application bytes):\n",
      (int)SHADOW_GRANULARITY);
  PrintShadowByte(str, "  Addressable:           ", 0);
  str->append("  Partially addressable: ");
  for (u8 i = 1; i < SHADOW_GRANULARITY; i++) PrintShadowByte(str, "", i, " ");
  str->append("\n");
  PrintShadowByte(str, "  Heap left redzone:       ",
                  kAsanHeapLeftRedzoneMagic);
  PrintShadowByte(str, "  Heap right redzone:      ",
                  kAsanHeapRightRedzoneMagic);
  PrintShadowByte(str, "  Freed heap region:       ", kAsanHeapFreeMagic);
  PrintShadowByte(str, "  Stack left redzone:      ",
                  kAsanStackLeftRedzoneMagic);
  PrintShadowByte(str, "  Stack mid redzone:       ",
                  kAsanStackMidRedzoneMagic);
  PrintShadowByte(str, "  Stack right redzone:     ",
                  kAsanStackRightRedzoneMagic);
  PrintShadowByte(str, "  Stack partial redzone:   ",
                  kAsanStackPartialRedzoneMagic);
  PrintShadowByte(str, "  Stack after return:      ",
                  kAsanStackAfterReturnMagic);
  PrintShadowByte(str, "  Stack use after scope:   ",
                  kAsanStackUseAfterScopeMagic);
  PrintShadowByte(str, "  Global redzone:          ", kAsanGlobalRedzoneMagic);
  PrintShadowByte(str, "  Global init order:       ",
                  kAsanInitializationOrderMagic);
  PrintShadowByte(str, "  Poisoned by user:        ",
                  kAsanUserPoisonedMemoryMagic);
  PrintShadowByte(str, "  Container overflow:      ",
                  kAsanContiguousContainerOOBMagic);
  PrintShadowByte(str, "  ASan internal:           ", kAsanInternalHeapMagic);
}

static void PrintShadowMemoryForAddress(uptr addr) {
  if (!AddrIsInMem(addr)) return;
  uptr shadow_addr = MemToShadow(addr);
  const uptr n_bytes_per_row = 16;
  uptr aligned_shadow = shadow_addr & ~(n_bytes_per_row - 1);
  InternalScopedString str(4096 * 8);
  str.append("Shadow bytes around the buggy address:\n");
  for (int i = -5; i <= 5; i++) {
    const char *prefix = (i == 0) ? "=>" : "  ";
    PrintShadowBytes(&str, prefix, (u8 *)(aligned_shadow + i * n_bytes_per_row),
                     (u8 *)shadow_addr, n_bytes_per_row);
  }
  if (flags()->print_legend) PrintLegend(&str);
  Printf("%s", str.data());
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

static void DescribeThread(AsanThread *t) {
  if (t)
    DescribeThread(t->context());
}

// ---------------------- Address Descriptions ------------------- {{{1

static bool IsASCII(unsigned char c) {
  return /*0x00 <= c &&*/ c <= 0x7F;
}

static const char *MaybeDemangleGlobalName(const char *name) {
  // We can spoil names of globals with C linkage, so use an heuristic
  // approach to check if the name should be demangled.
  return (name[0] == '_' && name[1] == 'Z')
             ? Symbolizer::Get()->Demangle(name)
             : name;
}

// Check if the global is a zero-terminated ASCII string. If so, print it.
static void PrintGlobalNameIfASCII(InternalScopedString *str,
                                   const __asan_global &g) {
  for (uptr p = g.beg; p < g.beg + g.size - 1; p++) {
    unsigned char c = *(unsigned char*)p;
    if (c == '\0' || !IsASCII(c)) return;
  }
  if (*(char*)(g.beg + g.size - 1) != '\0') return;
  str->append("  '%s' is ascii string '%s'\n", MaybeDemangleGlobalName(g.name),
              (char *)g.beg);
}

bool DescribeAddressRelativeToGlobal(uptr addr, uptr size,
                                     const __asan_global &g) {
  static const uptr kMinimalDistanceFromAnotherGlobal = 64;
  if (addr <= g.beg - kMinimalDistanceFromAnotherGlobal) return false;
  if (addr >= g.beg + g.size_with_redzone) return false;
  InternalScopedString str(4096);
  Decorator d;
  str.append("%s", d.Location());
  if (addr < g.beg) {
    str.append("%p is located %zd bytes to the left", (void *)addr,
               g.beg - addr);
  } else if (addr + size > g.beg + g.size) {
    if (addr < g.beg + g.size)
      addr = g.beg + g.size;
    str.append("%p is located %zd bytes to the right", (void *)addr,
               addr - (g.beg + g.size));
  } else {
    // Can it happen?
    str.append("%p is located %zd bytes inside", (void *)addr, addr - g.beg);
  }
  str.append(" of global variable '%s' from '%s' (0x%zx) of size %zu\n",
             MaybeDemangleGlobalName(g.name), g.module_name, g.beg, g.size);
  str.append("%s", d.EndLocation());
  PrintGlobalNameIfASCII(&str, g);
  Printf("%s", str.data());
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

void PrintAccessAndVarIntersection(const char *var_name,
                                   uptr var_beg, uptr var_size,
                                   uptr addr, uptr access_size,
                                   uptr prev_var_end, uptr next_var_beg) {
  uptr var_end = var_beg + var_size;
  uptr addr_end = addr + access_size;
  const char *pos_descr = 0;
  // If the variable [var_beg, var_end) is the nearest variable to the
  // current memory access, indicate it in the log.
  if (addr >= var_beg) {
    if (addr_end <= var_end)
      pos_descr = "is inside";  // May happen if this is a use-after-return.
    else if (addr < var_end)
      pos_descr = "partially overflows";
    else if (addr_end <= next_var_beg &&
             next_var_beg - addr_end >= addr - var_end)
      pos_descr = "overflows";
  } else {
    if (addr_end > var_beg)
      pos_descr = "partially underflows";
    else if (addr >= prev_var_end &&
             addr - prev_var_end >= var_beg - addr_end)
      pos_descr = "underflows";
  }
  InternalScopedString str(1024);
  str.append("    [%zd, %zd) '%s'", var_beg, var_beg + var_size, var_name);
  if (pos_descr) {
    Decorator d;
    // FIXME: we may want to also print the size of the access here,
    // but in case of accesses generated by memset it may be confusing.
    str.append("%s <== Memory access at offset %zd %s this variable%s\n",
               d.Location(), addr, pos_descr, d.EndLocation());
  } else {
    str.append("\n");
  }
  Printf("%s", str.data());
}

struct StackVarDescr {
  uptr beg;
  uptr size;
  const char *name_pos;
  uptr name_len;
};

bool DescribeAddressIfStack(uptr addr, uptr access_size) {
  AsanThread *t = FindThreadByStackAddress(addr);
  if (!t) return false;
  const uptr kBufSize = 4095;
  char buf[kBufSize];
  uptr offset = 0;
  uptr frame_pc = 0;
  char tname[128];
  const char *frame_descr = t->GetFrameNameByAddr(addr, &offset, &frame_pc);

#ifdef __powerpc64__
  // On PowerPC64, the address of a function actually points to a
  // three-doubleword data structure with the first field containing
  // the address of the function's code.
  frame_pc = *reinterpret_cast<uptr *>(frame_pc);
#endif

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
  alloca_stack.Print();
  // Report the number of stack objects.
  char *p;
  uptr n_objects = (uptr)internal_simple_strtoll(frame_descr, &p, 10);
  CHECK_GT(n_objects, 0);
  Printf("  This frame has %zu object(s):\n", n_objects);

  // Report all objects in this frame.
  InternalScopedBuffer<StackVarDescr> vars(n_objects);
  for (uptr i = 0; i < n_objects; i++) {
    uptr beg, size;
    uptr len;
    beg  = (uptr)internal_simple_strtoll(p, &p, 10);
    size = (uptr)internal_simple_strtoll(p, &p, 10);
    len  = (uptr)internal_simple_strtoll(p, &p, 10);
    if (beg == 0 || size == 0 || *p != ' ') {
      Printf("AddressSanitizer can't parse the stack frame "
                 "descriptor: |%s|\n", frame_descr);
      break;
    }
    p++;
    vars[i].beg = beg;
    vars[i].size = size;
    vars[i].name_pos = p;
    vars[i].name_len = len;
    p += len;
  }
  for (uptr i = 0; i < n_objects; i++) {
    buf[0] = 0;
    internal_strncat(buf, vars[i].name_pos,
                     static_cast<uptr>(Min(kBufSize, vars[i].name_len)));
    uptr prev_var_end = i ? vars[i - 1].beg + vars[i - 1].size : 0;
    uptr next_var_beg = i + 1 < n_objects ? vars[i + 1].beg : ~(0UL);
    PrintAccessAndVarIntersection(buf, vars[i].beg, vars[i].size,
                                  offset, access_size,
                                  prev_var_end, next_var_beg);
  }
  Printf("HINT: this may be a false positive if your program uses "
             "some custom stack unwind mechanism or swapcontext\n"
             "      (longjmp and C++ exceptions *are* supported)\n");
  DescribeThread(t);
  return true;
}

static void DescribeAccessToHeapChunk(AsanChunkView chunk, uptr addr,
                                      uptr access_size) {
  sptr offset;
  Decorator d;
  InternalScopedString str(4096);
  str.append("%s", d.Location());
  if (chunk.AddrIsAtLeft(addr, access_size, &offset)) {
    str.append("%p is located %zd bytes to the left of", (void *)addr, offset);
  } else if (chunk.AddrIsAtRight(addr, access_size, &offset)) {
    if (offset < 0) {
      addr -= offset;
      offset = 0;
    }
    str.append("%p is located %zd bytes to the right of", (void *)addr, offset);
  } else if (chunk.AddrIsInside(addr, access_size, &offset)) {
    str.append("%p is located %zd bytes inside of", (void*)addr, offset);
  } else {
    str.append("%p is located somewhere around (this is AddressSanitizer bug!)",
               (void *)addr);
  }
  str.append(" %zu-byte region [%p,%p)\n", chunk.UsedSize(),
             (void *)(chunk.Beg()), (void *)(chunk.End()));
  str.append("%s", d.EndLocation());
  Printf("%s", str.data());
}

void DescribeHeapAddress(uptr addr, uptr access_size) {
  AsanChunkView chunk = FindHeapChunkByAddress(addr);
  if (!chunk.IsValid()) {
    Printf("AddressSanitizer can not describe address in more detail "
           "(wild memory access suspected).\n");
    return;
  }
  DescribeAccessToHeapChunk(chunk, addr, access_size);
  CHECK(chunk.AllocTid() != kInvalidTid);
  asanThreadRegistry().CheckLocked();
  AsanThreadContext *alloc_thread =
      GetThreadContextByTidLocked(chunk.AllocTid());
  StackTrace alloc_stack;
  chunk.GetAllocStack(&alloc_stack);
  char tname[128];
  Decorator d;
  AsanThreadContext *free_thread = 0;
  if (chunk.FreeTid() != kInvalidTid) {
    free_thread = GetThreadContextByTidLocked(chunk.FreeTid());
    Printf("%sfreed by thread T%d%s here:%s\n", d.Allocation(),
           free_thread->tid,
           ThreadNameWithParenthesis(free_thread, tname, sizeof(tname)),
           d.EndAllocation());
    StackTrace free_stack;
    chunk.GetFreeStack(&free_stack);
    free_stack.Print();
    Printf("%spreviously allocated by thread T%d%s here:%s\n",
           d.Allocation(), alloc_thread->tid,
           ThreadNameWithParenthesis(alloc_thread, tname, sizeof(tname)),
           d.EndAllocation());
  } else {
    Printf("%sallocated by thread T%d%s here:%s\n", d.Allocation(),
           alloc_thread->tid,
           ThreadNameWithParenthesis(alloc_thread, tname, sizeof(tname)),
           d.EndAllocation());
  }
  alloc_stack.Print();
  DescribeThread(GetCurrentThread());
  if (free_thread)
    DescribeThread(free_thread);
  DescribeThread(alloc_thread);
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
  InternalScopedString str(1024);
  str.append("Thread T%d%s", context->tid,
             ThreadNameWithParenthesis(context->tid, tname, sizeof(tname)));
  str.append(
      " created by T%d%s here:\n", context->parent_tid,
      ThreadNameWithParenthesis(context->parent_tid, tname, sizeof(tname)));
  Printf("%s", str.data());
  uptr stack_size;
  const uptr *stack_trace = StackDepotGet(context->stack_id, &stack_size);
  StackTrace::PrintStack(stack_trace, stack_size);
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
  }
  // Destructor is NORETURN, as functions that report errors are.
  NORETURN ~ScopedInErrorReport() {
    // Make sure the current thread is announced.
    DescribeThread(GetCurrentThread());
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

void ReportStackOverflow(uptr pc, uptr sp, uptr bp, void *context, uptr addr) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report(
      "ERROR: AddressSanitizer: stack-overflow on address %p"
      " (pc %p sp %p bp %p T%d)\n",
      (void *)addr, (void *)pc, (void *)sp, (void *)bp,
      GetCurrentTidOrInvalid());
  Printf("%s", d.EndWarning());
  GET_STACK_TRACE_SIGNAL(pc, bp, context);
  stack.Print();
  ReportErrorSummary("stack-overflow", &stack);
}

void ReportSIGSEGV(uptr pc, uptr sp, uptr bp, void *context, uptr addr) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report(
      "ERROR: AddressSanitizer: SEGV on unknown address %p"
      " (pc %p sp %p bp %p T%d)\n",
      (void *)addr, (void *)pc, (void *)sp, (void *)bp,
      GetCurrentTidOrInvalid());
  Printf("%s", d.EndWarning());
  GET_STACK_TRACE_SIGNAL(pc, bp, context);
  stack.Print();
  Printf("AddressSanitizer can not provide additional info.\n");
  ReportErrorSummary("SEGV", &stack);
}

void ReportDoubleFree(uptr addr, StackTrace *free_stack) {
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
  CHECK_GT(free_stack->size, 0);
  GET_STACK_TRACE_FATAL(free_stack->trace[0], free_stack->top_frame_bp);
  stack.Print();
  DescribeHeapAddress(addr, 1);
  ReportErrorSummary("double-free", &stack);
}

void ReportFreeNotMalloced(uptr addr, StackTrace *free_stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  char tname[128];
  u32 curr_tid = GetCurrentTidOrInvalid();
  Report("ERROR: AddressSanitizer: attempting free on address "
             "which was not malloc()-ed: %p in thread T%d%s\n", addr,
         curr_tid, ThreadNameWithParenthesis(curr_tid, tname, sizeof(tname)));
  Printf("%s", d.EndWarning());
  CHECK_GT(free_stack->size, 0);
  GET_STACK_TRACE_FATAL(free_stack->trace[0], free_stack->top_frame_bp);
  stack.Print();
  DescribeHeapAddress(addr, 1);
  ReportErrorSummary("bad-free", &stack);
}

void ReportAllocTypeMismatch(uptr addr, StackTrace *free_stack,
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
  CHECK_GT(free_stack->size, 0);
  GET_STACK_TRACE_FATAL(free_stack->trace[0], free_stack->top_frame_bp);
  stack.Print();
  DescribeHeapAddress(addr, 1);
  ReportErrorSummary("alloc-dealloc-mismatch", &stack);
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
  stack->Print();
  DescribeHeapAddress(addr, 1);
  ReportErrorSummary("bad-malloc_usable_size", stack);
}

void ReportAsanGetAllocatedSizeNotOwned(uptr addr, StackTrace *stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: attempting to call "
             "__asan_get_allocated_size() for pointer which is "
             "not owned: %p\n", addr);
  Printf("%s", d.EndWarning());
  stack->Print();
  DescribeHeapAddress(addr, 1);
  ReportErrorSummary("bad-__asan_get_allocated_size", stack);
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
  stack->Print();
  DescribeAddress((uptr)offset1, length1);
  DescribeAddress((uptr)offset2, length2);
  ReportErrorSummary(bug_type, stack);
}

void ReportStringFunctionSizeOverflow(uptr offset, uptr size,
                                      StackTrace *stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  const char *bug_type = "negative-size-param";
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: %s: (size=%zd)\n", bug_type, size);
  Printf("%s", d.EndWarning());
  stack->Print();
  DescribeAddress(offset, size);
  ReportErrorSummary(bug_type, stack);
}

void ReportBadParamsToAnnotateContiguousContainer(uptr beg, uptr end,
                                                  uptr old_mid, uptr new_mid,
                                                  StackTrace *stack) {
  ScopedInErrorReport in_report;
  Report("ERROR: AddressSanitizer: bad parameters to "
         "__sanitizer_annotate_contiguous_container:\n"
         "      beg     : %p\n"
         "      end     : %p\n"
         "      old_mid : %p\n"
         "      new_mid : %p\n",
         beg, end, old_mid, new_mid);
  stack->Print();
  ReportErrorSummary("bad-__sanitizer_annotate_contiguous_container", stack);
}

void ReportODRViolation(const __asan_global *g1, const __asan_global *g2) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: odr-violation (%p):\n", g1->beg);
  Printf("%s", d.EndWarning());
  Printf("  [1] size=%zd %s %s\n", g1->size, g1->name, g1->module_name);
  Printf("  [2] size=%zd %s %s\n", g2->size, g2->name, g2->module_name);
  Report("HINT: if you don't care about these warnings you may set "
         "ASAN_OPTIONS=detect_odr_violation=0\n");
  ReportErrorSummary("odr-violation", g1->module_name, 0, g1->name);
}

// ----------------------- CheckForInvalidPointerPair ----------- {{{1
static NOINLINE void
ReportInvalidPointerPair(uptr pc, uptr bp, uptr sp, uptr a1, uptr a2) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: invalid-pointer-pair: %p %p\n", a1, a2);
  Printf("%s", d.EndWarning());
  GET_STACK_TRACE_FATAL(pc, bp);
  stack.Print();
  DescribeAddress(a1, 1);
  DescribeAddress(a2, 1);
  ReportErrorSummary("invalid-pointer-pair", &stack);
}

static INLINE void CheckForInvalidPointerPair(void *p1, void *p2) {
  if (!flags()->detect_invalid_pointer_pairs) return;
  uptr a1 = reinterpret_cast<uptr>(p1);
  uptr a2 = reinterpret_cast<uptr>(p2);
  AsanChunkView chunk1 = FindHeapChunkByAddress(a1);
  AsanChunkView chunk2 = FindHeapChunkByAddress(a2);
  bool valid1 = chunk1.IsValid();
  bool valid2 = chunk2.IsValid();
  if ((valid1 != valid2) || (valid1 && valid2 && !chunk1.Eq(chunk2))) {
    GET_CALLER_PC_BP_SP;                                              \
    return ReportInvalidPointerPair(pc, bp, sp, a1, a2);
  }
}
// ----------------------- Mac-specific reports ----------------- {{{1

void WarnMacFreeUnallocated(
    uptr addr, uptr zone_ptr, const char *zone_name, StackTrace *stack) {
  // Just print a warning here.
  Printf("free_common(%p) -- attempting to free unallocated memory.\n"
             "AddressSanitizer is ignoring this error on Mac OS now.\n",
             addr);
  PrintZoneForPointer(addr, zone_ptr, zone_name);
  stack->Print();
  DescribeHeapAddress(addr, 1);
}

void ReportMacMzReallocUnknown(
    uptr addr, uptr zone_ptr, const char *zone_name, StackTrace *stack) {
  ScopedInErrorReport in_report;
  Printf("mz_realloc(%p) -- attempting to realloc unallocated memory.\n"
             "This is an unrecoverable problem, exiting now.\n",
             addr);
  PrintZoneForPointer(addr, zone_ptr, zone_name);
  stack->Print();
  DescribeHeapAddress(addr, 1);
}

void ReportMacCfReallocUnknown(
    uptr addr, uptr zone_ptr, const char *zone_name, StackTrace *stack) {
  ScopedInErrorReport in_report;
  Printf("cf_realloc(%p) -- attempting to realloc unallocated memory.\n"
             "This is an unrecoverable problem, exiting now.\n",
             addr);
  PrintZoneForPointer(addr, zone_ptr, zone_name);
  stack->Print();
  DescribeHeapAddress(addr, 1);
}

}  // namespace __asan

// --------------------------- Interface --------------------- {{{1
using namespace __asan;  // NOLINT

void __asan_report_error(uptr pc, uptr bp, uptr sp, uptr addr, int is_write,
                         uptr access_size) {
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
      case kAsanContiguousContainerOOBMagic:
        bug_descr = "container-overflow";
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
  stack.Print();

  DescribeAddress(addr, access_size);
  ReportErrorSummary(bug_descr, &stack);
  PrintShadowMemoryForAddress(addr);
}

void NOINLINE __asan_set_error_report_callback(void (*callback)(const char*)) {
  error_report_callback = callback;
  if (callback) {
    error_message_buffer_size = 1 << 16;
    error_message_buffer =
        (char*)MmapOrDie(error_message_buffer_size, __func__);
    error_message_buffer_pos = 0;
  }
}

void __asan_describe_address(uptr addr) {
  DescribeAddress(addr, 1);
}

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_ptr_sub(void *a, void *b) {
  CheckForInvalidPointerPair(a, b);
}
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_ptr_cmp(void *a, void *b) {
  CheckForInvalidPointerPair(a, b);
}
}  // extern "C"

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
// Provide default implementation of __asan_on_error that does nothing
// and may be overriden by user.
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE NOINLINE
void __asan_on_error() {}
#endif
