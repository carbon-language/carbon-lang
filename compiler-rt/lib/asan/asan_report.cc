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
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"

namespace __asan {

// ---------------------- Error report callback ------------------- {{{1
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

// ---------------------- Address Descriptions ------------------- {{{1

static bool IsASCII(unsigned char c) {
  return 0x00 <= c && c <= 0x7F;
}

// Check if the global is a zero-terminated ASCII string. If so, print it.
static void PrintGlobalNameIfASCII(const __asan_global &g) {
  for (uptr p = g.beg; p < g.beg + g.size - 1; p++) {
    if (!IsASCII(*(unsigned char*)p)) return;
  }
  if (*(char*)(g.beg + g.size - 1) != 0) return;
  AsanPrintf("  '%s' is ascii string '%s'\n", g.name, (char*)g.beg);
}

bool DescribeAddressRelativeToGlobal(uptr addr, const __asan_global &g) {
  if (addr < g.beg - kGlobalAndStackRedzone) return false;
  if (addr >= g.beg + g.size_with_redzone) return false;
  AsanPrintf("%p is located ", (void*)addr);
  if (addr < g.beg) {
    AsanPrintf("%zd bytes to the left", g.beg - addr);
  } else if (addr >= g.beg + g.size) {
    AsanPrintf("%zd bytes to the right", addr - (g.beg + g.size));
  } else {
    AsanPrintf("%zd bytes inside", addr - g.beg);  // Can it happen?
  }
  AsanPrintf(" of global variable '%s' (0x%zx) of size %zu\n",
             g.name, g.beg, g.size);
  PrintGlobalNameIfASCII(g);
  return true;
}

bool DescribeAddressIfShadow(uptr addr) {
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
  CHECK(0 && "Address is not in memory and not in shadow?");
  return false;
}

bool DescribeAddressIfStack(uptr addr, uptr access_size) {
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

void DescribeAddress(uptr addr, uptr access_size) {
  // Check if this is shadow or shadow gap.
  if (DescribeAddressIfShadow(addr))
    return;
  CHECK(AddrIsInMem(addr));
  if (DescribeAddressIfGlobal(addr))
    return;
  if (DescribeAddressIfStack(addr, access_size))
    return;
  // Assume it is a heap address.
  DescribeHeapAddress(addr, access_size);
}

// -------------------- Different kinds of reports ----------------- {{{1

void ReportSIGSEGV(uptr pc, uptr sp, uptr bp, uptr addr) {
  AsanReport("ERROR: AddressSanitizer crashed on unknown address %p"
             " (pc %p sp %p bp %p T%d)\n",
             (void*)addr, (void*)pc, (void*)sp, (void*)bp,
             asanThreadRegistry().GetCurrentTidOrInvalid());
  AsanPrintf("AddressSanitizer can not provide additional info. ABORTING\n");
  GET_STACK_TRACE_WITH_PC_AND_BP(kStackTraceMax, pc, bp);
  stack.PrintStack();
  ShowStatsAndAbort();
}

void ReportDoubleFree(uptr addr, AsanStackTrace *stack) {
  AsanReport("ERROR: AddressSanitizer attempting double-free on %p:\n", addr);
  stack->PrintStack();
  DescribeHeapAddress(addr, 1);
  ShowStatsAndAbort();
}

void ReportFreeNotMalloced(uptr addr, AsanStackTrace *stack) {
  AsanReport("ERROR: AddressSanitizer attempting free on address "
             "which was not malloc()-ed: %p\n", addr);
  stack->PrintStack();
  ShowStatsAndAbort();
}

void ReportMallocUsableSizeNotOwned(uptr addr, AsanStackTrace *stack) {
  AsanReport("ERROR: AddressSanitizer attempting to call "
             "malloc_usable_size() for pointer which is "
             "not owned: %p\n", addr);
  stack->PrintStack();
  DescribeHeapAddress(addr, 1);
  ShowStatsAndAbort();
}

void ReportAsanGetAllocatedSizeNotOwned(uptr addr, AsanStackTrace *stack) {
  AsanReport("ERROR: AddressSanitizer attempting to call "
             "__asan_get_allocated_size() for pointer which is "
             "not owned: %p\n", addr);
  stack->PrintStack();
  DescribeHeapAddress(addr, 1);
  ShowStatsAndAbort();
}

void ReportStringFunctionMemoryRangesOverlap(
    const char *function, const char *offset1, uptr length1,
    const char *offset2, uptr length2, AsanStackTrace *stack) {
  AsanReport("ERROR: AddressSanitizer %s-param-overlap: "
             "memory ranges [%p,%p) and [%p, %p) overlap\n", \
             function, offset1, offset1 + length1, offset2, offset2 + length2);
  stack->PrintStack();
  ShowStatsAndAbort();
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

void ReportInvalidMemoryAccess(uptr pc, uptr bp, uptr sp, uptr addr,
                               bool is_write, uptr access_size) {
  static atomic_uint32_t num_calls;
  if (atomic_fetch_add(&num_calls, 1, memory_order_relaxed) != 0) {
    // Do not print more than one report, otherwise they will mix up.
    // We can not return here because the function is marked as never-return.
    AsanPrintf("AddressSanitizer: while reporting a bug found another one."
               "Ignoring.\n");
    SleepForSeconds(5);
    Die();
  }

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

}  // namespace __asan

// --------------------------- Interface --------------------- {{{1
using namespace __asan;  // NOLINT

void __asan_report_error(uptr pc, uptr bp, uptr sp,
                         uptr addr, bool is_write, uptr access_size) {
  ReportInvalidMemoryAccess(pc, bp, sp, addr, is_write, access_size);
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
