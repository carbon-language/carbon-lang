//===-- msan_report.cc -----------------------------------------------------===//
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
// Error reporting.
//===----------------------------------------------------------------------===//

#include "msan.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

using namespace __sanitizer;

static StaticSpinMutex report_mu;

namespace __msan {

static void DescribeOrigin(u32 origin) {
  if (flags()->verbosity)
    Printf("  raw origin id: %d\n", origin);
  if (const char *so = __msan_get_origin_descr_if_stack(origin)) {
    char* s = internal_strdup(so);
    char* sep = internal_strchr(s, '@');
    CHECK(sep);
    *sep = '\0';
    Printf("  Uninitialised value was created by an allocation of '%s'"
           " in the stack frame of function '%s'\n", s, sep + 1);
    InternalFree(s);
  } else {
    uptr size = 0;
    const uptr *trace = StackDepotGet(origin, &size);
    Printf("  Uninitialised value was created by a heap allocation\n");
    StackTrace::PrintStack(trace, size, true, "", 0);
  }
}

void ReportUMR(StackTrace *stack, u32 origin) {
  if (!__msan::flags()->report_umrs) return;

  GenericScopedLock<StaticSpinMutex> lock(&report_mu);

  Report(" WARNING: Use of uninitialized value\n");
  StackTrace::PrintStack(stack->trace, stack->size, true, "", 0);
  if (origin) {
    DescribeOrigin(origin);
  }
}

void ReportExpectedUMRNotFound(StackTrace *stack) {
  GenericScopedLock<StaticSpinMutex> lock(&report_mu);

  Printf(" WARNING: Expected use of uninitialized value not found\n");
  StackTrace::PrintStack(stack->trace, stack->size, true, "", 0);
}

}  // namespace msan
