//===-- asan_win.cc -------------------------------------------------------===//
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
// Windows-specific details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_WINDOWS
#include <windows.h>

#include <dbghelp.h>
#include <stdlib.h>

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_report.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"

extern "C" {
  SANITIZER_INTERFACE_ATTRIBUTE
  int __asan_should_detect_stack_use_after_return() {
    __asan_init();
    return __asan_option_detect_stack_use_after_return;
  }
}

namespace __asan {

// ---------------------- TSD ---------------- {{{1
static bool tsd_key_inited = false;

static __declspec(thread) void *fake_tsd = 0;

void AsanTSDInit(void (*destructor)(void *tsd)) {
  // FIXME: we're ignoring the destructor for now.
  tsd_key_inited = true;
}

void *AsanTSDGet() {
  CHECK(tsd_key_inited);
  return fake_tsd;
}

void AsanTSDSet(void *tsd) {
  CHECK(tsd_key_inited);
  fake_tsd = tsd;
}

void PlatformTSDDtor(void *tsd) {
  AsanThread::TSDDtor(tsd);
}
// ---------------------- Various stuff ---------------- {{{1
void DisableReexec() {
  // No need to re-exec on Windows.
}

void MaybeReexec() {
  // No need to re-exec on Windows.
}

void *AsanDoesNotSupportStaticLinkage() {
#if defined(_DEBUG)
#error Please build the runtime with a non-debug CRT: /MD or /MT
#endif
  return 0;
}

void AsanCheckDynamicRTPrereqs() {}

void AsanCheckIncompatibleRT() {}

void AsanPlatformThreadInit() {
  // Nothing here for now.
}

void ReadContextStack(void *context, uptr *stack, uptr *ssize) {
  UNIMPLEMENTED();
}

void AsanOnSIGSEGV(int, void *siginfo, void *context) {
  UNIMPLEMENTED();
}

static LPTOP_LEVEL_EXCEPTION_FILTER default_seh_handler;

SignalContext SignalContext::Create(void *siginfo, void *context) {
  EXCEPTION_RECORD *exception_record = (EXCEPTION_RECORD*)siginfo;
  CONTEXT *context_record = (CONTEXT*)context;

  uptr pc = (uptr)exception_record->ExceptionAddress;
#ifdef _WIN64
  uptr bp = (uptr)context_record->Rbp;
  uptr sp = (uptr)context_record->Rsp;
#else
  uptr bp = (uptr)context_record->Ebp;
  uptr sp = (uptr)context_record->Esp;
#endif
  uptr access_addr = exception_record->ExceptionInformation[1];

  return SignalContext(context, access_addr, pc, sp, bp);
}

static long WINAPI SEHHandler(EXCEPTION_POINTERS *info) {
  EXCEPTION_RECORD *exception_record = info->ExceptionRecord;
  CONTEXT *context = info->ContextRecord;

  if (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION ||
      exception_record->ExceptionCode == EXCEPTION_IN_PAGE_ERROR) {
    const char *description =
        (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION)
            ? "access-violation"
            : "in-page-error";
    SignalContext sig = SignalContext::Create(exception_record, context);
    ReportSIGSEGV(description, sig);
  }

  // FIXME: Handle EXCEPTION_STACK_OVERFLOW here.

  return default_seh_handler(info);
}

// We want to install our own exception handler (EH) to print helpful reports
// on access violations and whatnot.  Unfortunately, the CRT initializers assume
// they are run before any user code and drop any previously-installed EHs on
// the floor, so we can't install our handler inside __asan_init.
// (See crt0dat.c in the CRT sources for the details)
//
// Things get even more complicated with the dynamic runtime, as it finishes its
// initialization before the .exe module CRT begins to initialize.
//
// For the static runtime (-MT), it's enough to put a callback to
// __asan_set_seh_filter in the last section for C initializers.
//
// For the dynamic runtime (-MD), we want link the same
// asan_dynamic_runtime_thunk.lib to all the modules, thus __asan_set_seh_filter
// will be called for each instrumented module.  This ensures that at least one
// __asan_set_seh_filter call happens after the .exe module CRT is initialized.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE
int __asan_set_seh_filter() {
  // We should only store the previous handler if it's not our own handler in
  // order to avoid loops in the EH chain.
  auto prev_seh_handler = SetUnhandledExceptionFilter(SEHHandler);
  if (prev_seh_handler != &SEHHandler)
    default_seh_handler = prev_seh_handler;
  return 0;
}

#if !ASAN_DYNAMIC
// Put a pointer to __asan_set_seh_filter at the end of the global list
// of C initializers, after the default EH is set by the CRT.
#pragma section(".CRT$XIZ", long, read)  // NOLINT
static __declspec(allocate(".CRT$XIZ"))
    int (*__intercept_seh)() = __asan_set_seh_filter;
#endif

}  // namespace __asan

#endif  // _WIN32
