//===-- sanitizer_win.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements windows-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_WINDOWS

#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <stdlib.h>
#include <io.h>
#include <windows.h>

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_stacktrace.h"

namespace __sanitizer {

#include "sanitizer_syscall_generic.inc"

// --------------------- sanitizer_common.h
uptr GetPageSize() {
  return 1U << 14;  // FIXME: is this configurable?
}

uptr GetMmapGranularity() {
  return 1U << 16;  // FIXME: is this configurable?
}

uptr GetMaxVirtualAddress() {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (uptr)si.lpMaximumApplicationAddress;
}

bool FileExists(const char *filename) {
  UNIMPLEMENTED();
}

uptr internal_getpid() {
  return GetProcessId(GetCurrentProcess());
}

// In contrast to POSIX, on Windows GetCurrentThreadId()
// returns a system-unique identifier.
uptr GetTid() {
  return GetCurrentThreadId();
}

uptr GetThreadSelf() {
  return GetTid();
}

void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom) {
  CHECK(stack_top);
  CHECK(stack_bottom);
  MEMORY_BASIC_INFORMATION mbi;
  CHECK_NE(VirtualQuery(&mbi /* on stack */, &mbi, sizeof(mbi)), 0);
  // FIXME: is it possible for the stack to not be a single allocation?
  // Are these values what ASan expects to get (reserved, not committed;
  // including stack guard page) ?
  *stack_top = (uptr)mbi.BaseAddress + mbi.RegionSize;
  *stack_bottom = (uptr)mbi.AllocationBase;
}

void *MmapOrDie(uptr size, const char *mem_type) {
  void *rv = VirtualAlloc(0, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (rv == 0) {
    Report("ERROR: Failed to allocate 0x%zx (%zd) bytes of %s\n",
           size, size, mem_type);
    CHECK("unable to mmap" && 0);
  }
  return rv;
}

void UnmapOrDie(void *addr, uptr size) {
  if (VirtualFree(addr, size, MEM_DECOMMIT) == 0) {
    Report("ERROR: Failed to deallocate 0x%zx (%zd) bytes at address %p\n",
           size, size, addr);
    CHECK("unable to unmap" && 0);
  }
}

void *MmapFixedNoReserve(uptr fixed_addr, uptr size) {
  // FIXME: is this really "NoReserve"? On Win32 this does not matter much,
  // but on Win64 it does.
  void *p = VirtualAlloc((LPVOID)fixed_addr, size,
      MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (p == 0)
    Report("ERROR: Failed to allocate 0x%zx (%zd) bytes at %p (%d)\n",
           size, size, fixed_addr, GetLastError());
  return p;
}

void *MmapFixedOrDie(uptr fixed_addr, uptr size) {
  return MmapFixedNoReserve(fixed_addr, size);
}

void *MmapNoReserveOrDie(uptr size, const char *mem_type) {
  // FIXME: make this really NoReserve?
  return MmapOrDie(size, mem_type);
}

void *Mprotect(uptr fixed_addr, uptr size) {
  return VirtualAlloc((LPVOID)fixed_addr, size,
                      MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
}

void FlushUnneededShadowMemory(uptr addr, uptr size) {
  // This is almost useless on 32-bits.
  // FIXME: add madvice-analog when we move to 64-bits.
}

bool MemoryRangeIsAvailable(uptr range_start, uptr range_end) {
  // FIXME: shall we do anything here on Windows?
  return true;
}

void *MapFileToMemory(const char *file_name, uptr *buff_size) {
  UNIMPLEMENTED();
}

static const int kMaxEnvNameLength = 128;
static const DWORD kMaxEnvValueLength = 32767;

namespace {

struct EnvVariable {
  char name[kMaxEnvNameLength];
  char value[kMaxEnvValueLength];
};

}  // namespace

static const int kEnvVariables = 5;
static EnvVariable env_vars[kEnvVariables];
static int num_env_vars;

const char *GetEnv(const char *name) {
  // Note: this implementation caches the values of the environment variables
  // and limits their quantity.
  for (int i = 0; i < num_env_vars; i++) {
    if (0 == internal_strcmp(name, env_vars[i].name))
      return env_vars[i].value;
  }
  CHECK_LT(num_env_vars, kEnvVariables);
  DWORD rv = GetEnvironmentVariableA(name, env_vars[num_env_vars].value,
                                     kMaxEnvValueLength);
  if (rv > 0 && rv < kMaxEnvValueLength) {
    CHECK_LT(internal_strlen(name), kMaxEnvNameLength);
    internal_strncpy(env_vars[num_env_vars].name, name, kMaxEnvNameLength);
    num_env_vars++;
    return env_vars[num_env_vars - 1].value;
  }
  return 0;
}

const char *GetPwd() {
  UNIMPLEMENTED();
}

u32 GetUid() {
  UNIMPLEMENTED();
}

void DumpProcessMap() {
  UNIMPLEMENTED();
}

void DisableCoreDumper() {
  UNIMPLEMENTED();
}

void ReExec() {
  UNIMPLEMENTED();
}

void PrepareForSandboxing() {
  // Nothing here for now.
}

bool StackSizeIsUnlimited() {
  UNIMPLEMENTED();
}

void SetStackSizeLimitInBytes(uptr limit) {
  UNIMPLEMENTED();
}

char *FindPathToBinary(const char *name) {
  // Nothing here for now.
  return 0;
}

void SleepForSeconds(int seconds) {
  Sleep(seconds * 1000);
}

void SleepForMillis(int millis) {
  Sleep(millis);
}

u64 NanoTime() {
  return 0;
}

void Abort() {
  abort();
  internal__exit(-1);  // abort is not NORETURN on Windows.
}

uptr GetListOfModules(LoadedModule *modules, uptr max_modules,
                      string_predicate_t filter) {
  UNIMPLEMENTED();
};

#ifndef SANITIZER_GO
int Atexit(void (*function)(void)) {
  return atexit(function);
}
#endif

// ------------------ sanitizer_libc.h
uptr internal_mmap(void *addr, uptr length, int prot, int flags,
                   int fd, u64 offset) {
  UNIMPLEMENTED();
}

uptr internal_munmap(void *addr, uptr length) {
  UNIMPLEMENTED();
}

uptr internal_close(fd_t fd) {
  UNIMPLEMENTED();
}

int internal_isatty(fd_t fd) {
  return _isatty(fd);
}

uptr internal_open(const char *filename, int flags) {
  UNIMPLEMENTED();
}

uptr internal_open(const char *filename, int flags, u32 mode) {
  UNIMPLEMENTED();
}

uptr OpenFile(const char *filename, bool write) {
  UNIMPLEMENTED();
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  UNIMPLEMENTED();
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  if (fd != kStderrFd)
    UNIMPLEMENTED();

  static HANDLE output_stream = 0;
  // Abort immediately if we know printing is not possible.
  if (output_stream == INVALID_HANDLE_VALUE)
    return 0;

  // If called for the first time, try to use stderr to output stuff,
  // falling back to stdout if anything goes wrong.
  bool fallback_to_stdout = false;
  if (output_stream == 0) {
    output_stream = GetStdHandle(STD_ERROR_HANDLE);
    // We don't distinguish "no such handle" from error.
    if (output_stream == 0)
      output_stream = INVALID_HANDLE_VALUE;

    if (output_stream == INVALID_HANDLE_VALUE) {
      // Retry with stdout?
      output_stream = GetStdHandle(STD_OUTPUT_HANDLE);
      if (output_stream == 0)
        output_stream = INVALID_HANDLE_VALUE;
      if (output_stream == INVALID_HANDLE_VALUE)
        return 0;
    } else {
      // Successfully got an stderr handle.  However, if WriteFile() fails,
      // we can still try to fallback to stdout.
      fallback_to_stdout = true;
    }
  }

  DWORD ret;
  if (WriteFile(output_stream, buf, count, &ret, 0))
    return ret;

  // Re-try with stdout if using a valid stderr handle fails.
  if (fallback_to_stdout) {
    output_stream = GetStdHandle(STD_OUTPUT_HANDLE);
    if (output_stream == 0)
      output_stream = INVALID_HANDLE_VALUE;
    if (output_stream != INVALID_HANDLE_VALUE)
      return internal_write(fd, buf, count);
  }
  return 0;
}

uptr internal_stat(const char *path, void *buf) {
  UNIMPLEMENTED();
}

uptr internal_lstat(const char *path, void *buf) {
  UNIMPLEMENTED();
}

uptr internal_fstat(fd_t fd, void *buf) {
  UNIMPLEMENTED();
}

uptr internal_filesize(fd_t fd) {
  UNIMPLEMENTED();
}

uptr internal_dup2(int oldfd, int newfd) {
  UNIMPLEMENTED();
}

uptr internal_readlink(const char *path, char *buf, uptr bufsize) {
  UNIMPLEMENTED();
}

uptr internal_sched_yield() {
  Sleep(0);
  return 0;
}

void internal__exit(int exitcode) {
  ExitProcess(exitcode);
}

// ---------------------- BlockingMutex ---------------- {{{1
const uptr LOCK_UNINITIALIZED = 0;
const uptr LOCK_READY = (uptr)-1;

BlockingMutex::BlockingMutex(LinkerInitialized li) {
  // FIXME: see comments in BlockingMutex::Lock() for the details.
  CHECK(li == LINKER_INITIALIZED || owner_ == LOCK_UNINITIALIZED);

  CHECK(sizeof(CRITICAL_SECTION) <= sizeof(opaque_storage_));
  InitializeCriticalSection((LPCRITICAL_SECTION)opaque_storage_);
  owner_ = LOCK_READY;
}

BlockingMutex::BlockingMutex() {
  CHECK(sizeof(CRITICAL_SECTION) <= sizeof(opaque_storage_));
  InitializeCriticalSection((LPCRITICAL_SECTION)opaque_storage_);
  owner_ = LOCK_READY;
}

void BlockingMutex::Lock() {
  if (owner_ == LOCK_UNINITIALIZED) {
    // FIXME: hm, global BlockingMutex objects are not initialized?!?
    // This might be a side effect of the clang+cl+link Frankenbuild...
    new(this) BlockingMutex((LinkerInitialized)(LINKER_INITIALIZED + 1));

    // FIXME: If it turns out the linker doesn't invoke our
    // constructors, we should probably manually Lock/Unlock all the global
    // locks while we're starting in one thread to avoid double-init races.
  }
  EnterCriticalSection((LPCRITICAL_SECTION)opaque_storage_);
  CHECK_EQ(owner_, LOCK_READY);
  owner_ = GetThreadSelf();
}

void BlockingMutex::Unlock() {
  CHECK_EQ(owner_, GetThreadSelf());
  owner_ = LOCK_READY;
  LeaveCriticalSection((LPCRITICAL_SECTION)opaque_storage_);
}

void BlockingMutex::CheckLocked() {
  CHECK_EQ(owner_, GetThreadSelf());
}

uptr GetTlsSize() {
  return 0;
}

void InitTlsSize() {
}

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
#ifdef SANITIZER_GO
  *stk_addr = 0;
  *stk_size = 0;
  *tls_addr = 0;
  *tls_size = 0;
#else
  uptr stack_top, stack_bottom;
  GetThreadStackTopAndBottom(main, &stack_top, &stack_bottom);
  *stk_addr = stack_bottom;
  *stk_size = stack_top - stack_bottom;
  *tls_addr = 0;
  *tls_size = 0;
#endif
}

void StackTrace::SlowUnwindStack(uptr pc, uptr max_depth) {
  // FIXME: CaptureStackBackTrace might be too slow for us.
  // FIXME: Compare with StackWalk64.
  // FIXME: Look at LLVMUnhandledExceptionFilter in Signals.inc
  size = CaptureStackBackTrace(2, Min(max_depth, kStackTraceMax),
                               (void**)trace, 0);
  if (size == 0)
    return;

  // Skip the RTL frames by searching for the PC in the stacktrace.
  uptr pc_location = LocatePcInTrace(pc);
  PopStackFrames(pc_location);
}

void MaybeOpenReportFile() {
  // Windows doesn't have native fork, and we don't support Cygwin or other
  // environments that try to fake it, so the initial report_fd will always be
  // correct.
}

void RawWrite(const char *buffer) {
  uptr length = (uptr)internal_strlen(buffer);
  if (length != internal_write(report_fd, buffer, length)) {
    // stderr may be closed, but we may be able to print to the debugger
    // instead.  This is the case when launching a program from Visual Studio,
    // and the following routine should write to its console.
    OutputDebugStringA(buffer);
  }
}

void SetAlternateSignalStack() {
  // FIXME: Decide what to do on Windows.
}

void UnsetAlternateSignalStack() {
  // FIXME: Decide what to do on Windows.
}

void InstallDeadlySignalHandlers(SignalHandlerType handler) {
  (void)handler;
  // FIXME: Decide what to do on Windows.
}

bool IsDeadlySignal(int signum) {
  // FIXME: Decide what to do on Windows.
  return false;
}

}  // namespace __sanitizer

#endif  // _WIN32
