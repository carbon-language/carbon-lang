//===-- msan_report.cc ----------------------------------------------------===//
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
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __sanitizer;

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

static void PrintStack(const uptr *trace, uptr size) {
  SymbolizerScope sym_scope;
  StackTrace::PrintStack(trace, size, true,
                         common_flags()->strip_path_prefix, 0);
}

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
    Printf("  %sUninitialized value was created by an allocation of '%s%s%s'"
           " in the stack frame of function '%s%s%s'%s\n",
           d.Origin(), d.Name(), s, d.Origin(), d.Name(), Demangle(sep + 1),
           d.Origin(), d.End());
    InternalFree(s);
  } else {
    uptr size = 0;
    const uptr *trace = StackDepotGet(origin, &size);
    Printf("  %sUninitialized value was created by a heap allocation%s\n",
           d.Origin(), d.End());
    PrintStack(trace, size);
  }
}

static void ReportSummary(const char *error_type, StackTrace *stack) {
  if (!stack->size || !IsSymbolizerAvailable()) return;
  AddressInfo ai;
  uptr pc = StackTrace::GetPreviousInstructionPc(stack->trace[0]);
  {
    SymbolizerScope sym_scope;
    SymbolizeCode(pc, &ai, 1);
  }
  ReportErrorSummary(error_type,
                     StripPathPrefix(ai.file,
                                     common_flags()->strip_path_prefix),
                     ai.line, ai.function);
}

void ReportUMR(StackTrace *stack, u32 origin) {
  if (!__msan::flags()->report_umrs) return;

  SpinMutexLock l(&CommonSanitizerReportMutex);

  Decorator d;
  Printf("%s", d.Warning());
  Report(" WARNING: MemorySanitizer: use-of-uninitialized-value\n");
  Printf("%s", d.End());
  PrintStack(stack->trace, stack->size);
  if (origin) {
    DescribeOrigin(origin);
  }
  ReportSummary("use-of-uninitialized-value", stack);
}

void ReportExpectedUMRNotFound(StackTrace *stack) {
  SpinMutexLock l(&CommonSanitizerReportMutex);

  Printf(" WARNING: Expected use of uninitialized value not found\n");
  PrintStack(stack->trace, stack->size);
}

void ReportAtExitStatistics() {
  SpinMutexLock l(&CommonSanitizerReportMutex);

  Decorator d;
  Printf("%s", d.Warning());
  Printf("MemorySanitizer: %d warnings reported.\n", msan_report_count);
  Printf("%s", d.End());
}


}  // namespace __msan
