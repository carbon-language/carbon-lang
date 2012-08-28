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
#ifdef _WIN32
#include <windows.h>

#include <dbghelp.h>
#include <stdlib.h>

#include <new>  // FIXME: temporarily needed for placement new in AsanLock.

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_lock.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_libc.h"

namespace __asan {

// ---------------------- Stacktraces, symbols, etc. ---------------- {{{1
static AsanLock dbghelp_lock(LINKER_INITIALIZED);
static bool dbghelp_initialized = false;
#pragma comment(lib, "dbghelp.lib")

void GetStackTrace(StackTrace *stack, uptr max_s, uptr pc, uptr bp) {
  stack->max_size = max_s;
  void *tmp[kStackTraceMax];

  // FIXME: CaptureStackBackTrace might be too slow for us.
  // FIXME: Compare with StackWalk64.
  // FIXME: Look at LLVMUnhandledExceptionFilter in Signals.inc
  uptr cs_ret = CaptureStackBackTrace(1, stack->max_size, tmp, 0);
  uptr offset = 0;
  // Skip the RTL frames by searching for the PC in the stacktrace.
  // FIXME: this doesn't work well for the malloc/free stacks yet.
  for (uptr i = 0; i < cs_ret; i++) {
    if (pc != (uptr)tmp[i])
      continue;
    offset = i;
    break;
  }

  stack->size = cs_ret - offset;
  for (uptr i = 0; i < stack->size; i++)
    stack->trace[i] = (uptr)tmp[i + offset];
}

bool WinSymbolize(const void *addr, char *out_buffer, int buffer_size) {
  ScopedLock lock(&dbghelp_lock);
  if (!dbghelp_initialized) {
    SymSetOptions(SYMOPT_DEFERRED_LOADS |
                  SYMOPT_UNDNAME |
                  SYMOPT_LOAD_LINES);
    CHECK(SymInitialize(GetCurrentProcess(), 0, TRUE));
    // FIXME: We don't call SymCleanup() on exit yet - should we?
    dbghelp_initialized = true;
  }

  // See http://msdn.microsoft.com/en-us/library/ms680578(VS.85).aspx
  char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(CHAR)];
  PSYMBOL_INFO symbol = (PSYMBOL_INFO)buffer;
  symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
  symbol->MaxNameLen = MAX_SYM_NAME;
  DWORD64 offset = 0;
  BOOL got_objname = SymFromAddr(GetCurrentProcess(),
                                 (DWORD64)addr, &offset, symbol);
  if (!got_objname)
    return false;

  DWORD  unused;
  IMAGEHLP_LINE64 info;
  info.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
  BOOL got_fileline = SymGetLineFromAddr64(GetCurrentProcess(),
                                           (DWORD64)addr, &unused, &info);
  int written = 0;
  out_buffer[0] = '\0';
  // FIXME: it might be useful to print out 'obj' or 'obj+offset' info too.
  if (got_fileline) {
    written += internal_snprintf(out_buffer + written, buffer_size - written,
                        " %s %s:%d", symbol->Name,
                        info.FileName, info.LineNumber);
  } else {
    written += internal_snprintf(out_buffer + written, buffer_size - written,
                        " %s+0x%p", symbol->Name, offset);
  }
  return true;
}

// ---------------------- AsanLock ---------------- {{{1
enum LockState {
  LOCK_UNINITIALIZED = 0,
  LOCK_READY = -1,
};

AsanLock::AsanLock(LinkerInitialized li) {
  // FIXME: see comments in AsanLock::Lock() for the details.
  CHECK(li == LINKER_INITIALIZED || owner_ == LOCK_UNINITIALIZED);

  CHECK(sizeof(CRITICAL_SECTION) <= sizeof(opaque_storage_));
  InitializeCriticalSection((LPCRITICAL_SECTION)opaque_storage_);
  owner_ = LOCK_READY;
}

void AsanLock::Lock() {
  if (owner_ == LOCK_UNINITIALIZED) {
    // FIXME: hm, global AsanLock objects are not initialized?!?
    // This might be a side effect of the clang+cl+link Frankenbuild...
    new(this) AsanLock((LinkerInitialized)(LINKER_INITIALIZED + 1));

    // FIXME: If it turns out the linker doesn't invoke our
    // constructors, we should probably manually Lock/Unlock all the global
    // locks while we're starting in one thread to avoid double-init races.
  }
  EnterCriticalSection((LPCRITICAL_SECTION)opaque_storage_);
  CHECK(owner_ == LOCK_READY);
  owner_ = GetThreadSelf();
}

void AsanLock::Unlock() {
  CHECK(owner_ == GetThreadSelf());
  owner_ = LOCK_READY;
  LeaveCriticalSection((LPCRITICAL_SECTION)opaque_storage_);
}

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

// ---------------------- Various stuff ---------------- {{{1
void MaybeReexec() {
  // No need to re-exec on Windows.
}

void *AsanDoesNotSupportStaticLinkage() {
#if defined(_DEBUG)
#error Please build the runtime with a non-debug CRT: /MD or /MT
#endif
  return 0;
}

void SetAlternateSignalStack() {
  // FIXME: Decide what to do on Windows.
}

void UnsetAlternateSignalStack() {
  // FIXME: Decide what to do on Windows.
}

void InstallSignalHandlers() {
  // FIXME: Decide what to do on Windows.
}

void AsanPlatformThreadInit() {
  // Nothing here for now.
}

}  // namespace __asan

#endif  // _WIN32
