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
#include "asan_procmaps.h"
#include "asan_thread.h"

// Should not add dependency on libstdc++,
// since most of the stuff here is inlinable.
#include <algorithm>

namespace __asan {

// ---------------------- Memory management ---------------- {{{1
void *AsanMmapFixedNoReserve(uptr fixed_addr, uptr size) {
  return VirtualAlloc((LPVOID)fixed_addr, size,
                      MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void *AsanMmapSomewhereOrDie(uptr size, const char *mem_type) {
  void *rv = VirtualAlloc(0, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (rv == 0)
    OutOfMemoryMessageAndDie(mem_type, size);
  return rv;
}

void *AsanMprotect(uptr fixed_addr, uptr size) {
  return VirtualAlloc((LPVOID)fixed_addr, size,
                      MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
}

void AsanUnmapOrDie(void *addr, uptr size) {
  CHECK(VirtualFree(addr, size, MEM_DECOMMIT));
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
  stack_top_ = (uptr)mbi.BaseAddress + mbi.RegionSize;
  stack_bottom_ = (uptr)mbi.AllocationBase;
}

void AsanStackTrace::GetStackTrace(uptr max_s, uptr pc, uptr bp) {
  max_size = max_s;
  void *tmp[kStackTraceMax];

  // FIXME: CaptureStackBackTrace might be too slow for us.
  // FIXME: Compare with StackWalk64.
  // FIXME: Look at LLVMUnhandledExceptionFilter in Signals.inc
  uptr cs_ret = CaptureStackBackTrace(1, max_size, tmp, 0),
         offset = 0;
  // Skip the RTL frames by searching for the PC in the stacktrace.
  // FIXME: this doesn't work well for the malloc/free stacks yet.
  for (uptr i = 0; i < cs_ret; i++) {
    if (pc != (uptr)tmp[i])
      continue;
    offset = i;
    break;
  }

  size = cs_ret - offset;
  for (uptr i = 0; i < size; i++)
    trace[i] = (uptr)tmp[i + offset];
}

bool __asan_WinSymbolize(const void *addr, char *out_buffer, int buffer_size) {
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
void *AsanDoesNotSupportStaticLinkage() {
#if defined(_DEBUG)
#error Please build the runtime with a non-debug CRT: /MD or /MT
#endif
  return 0;
}

bool AsanShadowRangeIsAvailable() {
  // FIXME: shall we do anything here on Windows?
  return true;
}

int AtomicInc(int *a) {
  return InterlockedExchangeAdd((LONG*)a, 1) + 1;
}

u16 AtomicExchange(u16 *a, u16 new_val) {
  // InterlockedExchange16 seems unavailable on some MSVS installations.
  // Everybody stand back, I pretend to know inline assembly!
  // FIXME: I assume VC is smart enough to save/restore eax/ecx?
  __asm {
    mov eax, a
    mov cx, new_val
    xchg [eax], cx  ; NOLINT
    mov new_val, cx
  }
  return new_val;
}

u16 AtomicExchange(u16 *a, u16 new_val) {
  // FIXME: can we do this with a proper xchg intrinsic?
  u8 t = *a;
  *a = new_val;
  return t;
}

const char* AsanGetEnv(const char* name) {
  static char env_buffer[32767] = {};

  // Note: this implementation stores the result in a static buffer so we only
  // allow it to be called just once.
  static bool called_once = false;
  if (called_once)
    UNIMPLEMENTED();
  called_once = true;

  DWORD rv = GetEnvironmentVariableA(name, env_buffer, sizeof(env_buffer));
  if (rv > 0 && rv < sizeof(env_buffer))
    return env_buffer;
  return 0;
}

void AsanDumpProcessMap() {
  UNIMPLEMENTED();
}

uptr GetThreadSelf() {
  return GetCurrentThreadId();
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

void AsanDisableCoreDumper() {
  UNIMPLEMENTED();
}

void SleepForSeconds(int seconds) {
  Sleep(seconds * 1000);
}

void Exit(int exitcode) {
  _exit(exitcode);
}

void Abort() {
  abort();
  _exit(-1);  // abort is not NORETURN on Windows.
}

int Atexit(void (*function)(void)) {
  return atexit(function);
}

void SortArray(uptr *array, uptr size) {
  std::sort(array, array + size);
}

}  // namespace __asan

#endif  // _WIN32
