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
#include <stdio.h>  // FIXME: get rid of this.
#include <stdlib.h>

#include <new>  // FIXME: temporarily needed for placement new in AsanLock.

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_lock.h"
#include "asan_procmaps.h"
#include "asan_thread.h"

// Should not add dependency on libstdc++,
// since most of the stuff here is inlinable.
#include <algorithm>

namespace __asan {

// ---------------------- Memory management ---------------- {{{1
void *AsanMmapFixedNoReserve(uintptr_t fixed_addr, size_t size) {
  return VirtualAlloc((LPVOID)fixed_addr, size,
                      MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void *AsanMmapSomewhereOrDie(size_t size, const char *mem_type) {
  void *rv = VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (rv == NULL)
    OutOfMemoryMessageAndDie(mem_type, size);
  return rv;
}

void *AsanMprotect(uintptr_t fixed_addr, size_t size) {
  return VirtualAlloc((LPVOID)fixed_addr, size,
                      MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
}

void AsanUnmapOrDie(void *addr, size_t size) {
  CHECK(VirtualFree(addr, size, MEM_DECOMMIT));
}

// ---------------------- IO ---------------- {{{1
size_t AsanWrite(int fd, const void *buf, size_t count) {
  if (fd != 2)
    UNIMPLEMENTED();

  HANDLE err = GetStdHandle(STD_ERROR_HANDLE);
  if (err == NULL)
    return 0;  // FIXME: this might not work on some apps.
  DWORD ret;
  if (!WriteFile(err, buf, count, &ret, NULL))
    return 0;
  return ret;
}

// FIXME: Looks like these functions are not needed and are linked in by the
// code unreachable on Windows. We should clean this up.
int AsanOpenReadonly(const char* filename) {
  UNIMPLEMENTED();
  return -1;
}

size_t AsanRead(int fd, void *buf, size_t count) {
  UNIMPLEMENTED();
  return -1;
}

int AsanClose(int fd) {
  UNIMPLEMENTED();
  return -1;
}

// ---------------------- Stacktraces, symbols, etc. ---------------- {{{1
static AsanLock dbghelp_lock(LINKER_INITIALIZED);
static bool dbghelp_initialized = false;
#pragma comment(lib, "dbghelp.lib")

void AsanThread::SetThreadStackTopAndBottom() {
  MEMORY_BASIC_INFORMATION mbi;
  CHECK(VirtualQuery(&mbi /* on stack */,
                    &mbi, sizeof(mbi)) != 0);
  // FIXME: is it possible for the stack to not be a single allocation?
  // Are these values what ASan expects to get (reserved, not committed;
  // including stack guard page) ?
  stack_top_ = (uintptr_t)mbi.BaseAddress + mbi.RegionSize;
  stack_bottom_ = (uintptr_t)mbi.AllocationBase;
}

void AsanStackTrace::GetStackTrace(size_t max_s, uintptr_t pc, uintptr_t bp) {
  max_size = max_s;
  void *tmp[kStackTraceMax];

  // FIXME: CaptureStackBackTrace might be too slow for us.
  // FIXME: Compare with StackWalk64.
  // FIXME: Look at LLVMUnhandledExceptionFilter in Signals.inc
  size_t cs_ret = CaptureStackBackTrace(1, max_size, tmp, NULL),
         offset = 0;
  // Skip the RTL frames by searching for the PC in the stacktrace.
  // FIXME: this doesn't work well for the malloc/free stacks yet.
  for (size_t i = 0; i < cs_ret; i++) {
    if (pc != (uintptr_t)tmp[i])
      continue;
    offset = i;
    break;
  }

  size = cs_ret - offset;
  for (size_t i = 0; i < size; i++)
    trace[i] = (uintptr_t)tmp[i + offset];
}

bool WinSymbolize(const void *addr, char *out_buffer, int buffer_size) {
  ScopedLock lock(&dbghelp_lock);
  if (!dbghelp_initialized) {
    SymSetOptions(SYMOPT_DEFERRED_LOADS |
                  SYMOPT_UNDNAME |
                  SYMOPT_LOAD_LINES);
    CHECK(SymInitialize(GetCurrentProcess(), NULL, TRUE));
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
    written += SNPrintf(out_buffer + written, buffer_size - written,
                        " %s %s:%d", symbol->Name,
                        info.FileName, info.LineNumber);
  } else {
    written += SNPrintf(out_buffer + written, buffer_size - written,
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

// FIXME: is __declspec enough?
static __declspec(thread) void *fake_tsd = NULL;

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
void *AsanDoesNotSupportStaticLinkage() {
#if !defined(_DLL) || defined(_DEBUG)
#error Please build the runtime with /MD
#endif
  return NULL;
}

bool AsanShadowRangeIsAvailable() {
  // FIXME: shall we do anything here on Windows?
  return true;
}

int AtomicInc(int *a) {
  return InterlockedExchangeAdd((LONG*)a, 1) + 1;
}

const char* AsanGetEnv(const char* name) {
  // FIXME: implement.
  return NULL;
}

void AsanDumpProcessMap() {
  UNIMPLEMENTED();
}

int GetPid() {
  return GetProcessId(GetCurrentProcess());
}

uintptr_t GetThreadSelf() {
  return GetCurrentThreadId();
}

void InstallSignalHandlers() {
  // FIXME: Decide what to do on Windows.
}

void AsanDisableCoreDumper() {
  UNIMPLEMENTED();
}

void SleepForSeconds(int seconds) {
  Sleep(seconds * 1000);
}

void Exit(int exitcode) {
  _exit(exitcode);
}

int Atexit(void (*function)(void)) {
  return atexit(function);
}

void SortArray(uintptr_t *array, size_t size) {
  std::sort(array, array + size);
}

}  // namespace __asan

#endif  // _WIN32
