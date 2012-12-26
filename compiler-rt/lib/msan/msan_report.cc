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
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

using namespace __sanitizer;

static StaticSpinMutex report_mu;

namespace __msan {

static bool PrintsToTtyCached() {
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
  const char *Origin()     { return Magenta(); }
  const char *Name()   { return Green(); }
  const char *End()    { return Default(); }
};

static void DescribeOrigin(u32 origin) {
  Decorator d;
  if (flags()->verbosity)
    Printf("  raw origin id: %d\n", origin);
  if (const char *so = __msan_get_origin_descr_if_stack(origin)) {
    char* s = internal_strdup(so);
    char* sep = internal_strchr(s, '@');
    CHECK(sep);
    *sep = '\0';
    Printf("%s", d.Origin());
    Printf("  %sUninitialised value was created by an allocation of '%s%s%s'"
           " in the stack frame of function '%s%s%s'%s\n",
           d.Origin(), d.Name(), s, d.Origin(), d.Name(), sep + 1,
           d.Origin(), d.End());
    InternalFree(s);
  } else {
    uptr size = 0;
    const uptr *trace = StackDepotGet(origin, &size);
    Printf("  %sUninitialised value was created by a heap allocation%s\n",
           d.Origin(), d.End());
    StackTrace::PrintStack(trace, size, true, "", 0);
  }
}

void ReportUMR(StackTrace *stack, u32 origin) {
  if (!__msan::flags()->report_umrs) return;

  GenericScopedLock<StaticSpinMutex> lock(&report_mu);

  Decorator d;
  Printf("%s", d.Warning());
  Report(" WARNING: Use of uninitialized value\n");
  Printf("%s", d.End());
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
