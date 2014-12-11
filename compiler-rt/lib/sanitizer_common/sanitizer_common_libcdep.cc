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

bool ReportFile::PrintsToTty() {
  SpinMutexLock l(mu);
  ReopenIfNecessary();
  return internal_isatty(fd) != 0;
}

bool ColorizeReports() {
  // FIXME: Add proper Windows support to AnsiColorDecorator and re-enable color
  // printing on Windows.
  if (SANITIZER_WINDOWS)
    return false;

  const char *flag = common_flags()->color;
  return internal_strcmp(flag, "always") == 0 ||
         (internal_strcmp(flag, "auto") == 0 && report_file.PrintsToTty());
}

static void (*sandboxing_callback)();
void SetSandboxingCallback(void (*f)()) {
  sandboxing_callback = f;
}

void ReportErrorSummary(const char *error_type, StackTrace *stack) {
  if (!common_flags()->print_summary)
    return;
#if !SANITIZER_GO
  if (stack->size > 0 && Symbolizer::GetOrInit()->CanReturnFileLineInfo()) {
    // Currently, we include the first stack frame into the report summary.
    // Maybe sometimes we need to choose another frame (e.g. skip memcpy/etc).
    uptr pc = StackTrace::GetPreviousInstructionPc(stack->trace[0]);
    SymbolizedStack *frame = Symbolizer::GetOrInit()->SymbolizePC(pc);
    const AddressInfo &ai = frame->info;
    ReportErrorSummary(error_type, ai.file, ai.line, ai.function);
    frame->ClearAll();
  }
#else
  AddressInfo ai;
  ReportErrorSummary(error_type, ai.file, ai.line, ai.function);
#endif
}

}  // namespace __sanitizer

void NOINLINE
__sanitizer_sandbox_on_notify(__sanitizer_sandbox_arguments *args) {
  PrepareForSandboxing(args);
  if (sandboxing_callback)
    sandboxing_callback();
}
