//===-- sanitizer_common_libcdep.cc ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

bool PrintsToTty() {
  MaybeOpenReportFile();
  return internal_isatty(report_fd) != 0;
}

bool PrintsToTtyCached() {
  // FIXME: Add proper Windows support to AnsiColorDecorator and re-enable color
  // printing on Windows.
  if (SANITIZER_WINDOWS)
    return 0;

  static int cached = 0;
  static bool prints_to_tty;
  if (!cached) {  // Not thread-safe.
    prints_to_tty = PrintsToTty();
    cached = 1;
  }
  return prints_to_tty;
}

bool ColorizeReports() {
  const char *flag = common_flags()->color;
  return internal_strcmp(flag, "always") == 0 ||
         (internal_strcmp(flag, "auto") == 0 && PrintsToTtyCached());
}

static void (*sandboxing_callback)();
void SetSandboxingCallback(void (*f)()) {
  sandboxing_callback = f;
}

void ReportErrorSummary(const char *error_type, StackTrace *stack) {
  if (!common_flags()->print_summary)
    return;
  AddressInfo ai;
#if !SANITIZER_GO
  if (stack->size > 0 && Symbolizer::GetOrInit()->CanReturnFileLineInfo()) {
    // Currently, we include the first stack frame into the report summary.
    // Maybe sometimes we need to choose another frame (e.g. skip memcpy/etc).
    uptr pc = StackTrace::GetPreviousInstructionPc(stack->trace[0]);
    Symbolizer::GetOrInit()->SymbolizePC(pc, &ai, 1);
  }
#endif
  ReportErrorSummary(error_type, ai.file, ai.line, ai.function);
}

}  // namespace __sanitizer

void NOINLINE
__sanitizer_sandbox_on_notify(__sanitizer_sandbox_arguments *args) {
  PrepareForSandboxing(args);
  if (sandboxing_callback)
    sandboxing_callback();
}
