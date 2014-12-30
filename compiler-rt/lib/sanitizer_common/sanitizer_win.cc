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
#include <windows.h>
#include <dbghelp.h>
#include <io.h>
#include <psapi.h>
#include <stdlib.h>

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

#if !SANITIZER_GO
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
#endif  // #if !SANITIZER_GO

void *MmapOrDie(uptr size, const char *mem_type) {
  void *rv = VirtualAlloc(0, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (rv == 0) {
    Report("ERROR: %s failed to "
           "allocate 0x%zx (%zd) bytes of %s (error code: %d)\n",
           SanitizerToolName, size, size, mem_type, GetLastError());
    CHECK("unable to mmap" && 0);
  }
  return rv;
}

void UnmapOrDie(void *addr, uptr size) {
  if (VirtualFree(addr, size, MEM_DECOMMIT) == 0) {
    Report("ERROR: %s failed to "
           "deallocate 0x%zx (%zd) bytes at address %p (error code: %d)\n",
           SanitizerToolName, size, size, addr, GetLastError());
    CHECK("unable to unmap" && 0);
  }
}

void *MmapFixedNoReserve(uptr fixed_addr, uptr size) {
  // FIXME: is this really "NoReserve"? On Win32 this does not matter much,
  // but on Win64 it does.
  void *p = VirtualAlloc((LPVOID)fixed_addr, size,
      MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (p == 0)
    Report("ERROR: %s failed to "
           "allocate %p (%zd) bytes at %p (error code: %d)\n",
           SanitizerToolName, size, size, fixed_addr, GetLastError());
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
  MEMORY_BASIC_INFORMATION mbi;
  CHECK(VirtualQuery((void *)range_start, &mbi, sizeof(mbi)));
  return mbi.Protect & PAGE_NOACCESS &&
         (uptr)mbi.BaseAddress + mbi.RegionSize >= range_end;
}

void *MapFileToMemory(const char *file_name, uptr *buff_size) {
  UNIMPLEMENTED();
}

void *MapWritableFileToMemory(void *addr, uptr size, uptr fd, uptr offset) {
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

namespace {
struct ModuleInfo {
  HMODULE handle;
  uptr base_address;
  uptr end_address;
};

int CompareModulesBase(const void *pl, const void *pr) {
  const ModuleInfo &l = *(ModuleInfo *)pl, &r = *(ModuleInfo *)pr;
  if (l.base_address < r.base_address)
    return -1;
  return l.base_address > r.base_address;
}
}  // namespace

void DumpProcessMap() {
  Report("Dumping process modules:\n");
  HANDLE cur_process = GetCurrentProcess();

  // Query the list of modules.  Start by assuming there are no more than 256
  // modules and retry if that's not sufficient.
  ModuleInfo *modules;
  size_t num_modules;
  {
    HMODULE *hmodules = 0;
    uptr modules_buffer_size = sizeof(HMODULE) * 256;
    DWORD bytes_required;
    while (!hmodules) {
      hmodules = (HMODULE *)MmapOrDie(modules_buffer_size, __FUNCTION__);
      CHECK(EnumProcessModules(cur_process, hmodules, modules_buffer_size,
                               &bytes_required));
      if (bytes_required > modules_buffer_size) {
        // Either there turned out to be more than 256 hmodules, or new hmodules
        // could have loaded since the last try.  Retry.
        UnmapOrDie(hmodules, modules_buffer_size);
        hmodules = 0;
        modules_buffer_size = bytes_required;
      }
    }

    num_modules = bytes_required / sizeof(HMODULE);
    modules =
        (ModuleInfo *)MmapOrDie(num_modules * sizeof(ModuleInfo), __FUNCTION__);
    for (size_t i = 0; i < num_modules; ++i) {
      modules[i].handle = hmodules[i];
      MODULEINFO mi;
      if (!GetModuleInformation(cur_process, hmodules[i], &mi, sizeof(mi)))
        continue;
      modules[i].base_address = (uptr)mi.lpBaseOfDll;
      modules[i].end_address = (uptr)mi.lpBaseOfDll + mi.SizeOfImage;
    }
    UnmapOrDie(hmodules, modules_buffer_size);
  }

  qsort(modules, num_modules, sizeof(ModuleInfo), CompareModulesBase);

  for (size_t i = 0; i < num_modules; ++i) {
    const ModuleInfo &mi = modules[i];
    char module_name[MAX_PATH];
    bool got_module_name = GetModuleFileNameEx(
        cur_process, mi.handle, module_name, sizeof(module_name));
    if (mi.end_address != 0) {
      Printf("\t%p-%p %s\n", mi.base_address, mi.end_address,
             got_module_name ? module_name : "[no name]");
    } else if (got_module_name) {
      Printf("\t??\?-??? %s\n", module_name);
    } else {
      Printf("\t???\n");
    }
  }
  UnmapOrDie(modules, num_modules * sizeof(ModuleInfo));
}

void DisableCoreDumperIfNecessary() {
  // Do nothing.
}

void ReExec() {
  UNIMPLEMENTED();
}

void PrepareForSandboxing(__sanitizer_sandbox_arguments *args) {
  (void)args;
  // Nothing here for now.
}

bool StackSizeIsUnlimited() {
  UNIMPLEMENTED();
}

void SetStackSizeLimitInBytes(uptr limit) {
  UNIMPLEMENTED();
}

bool AddressSpaceIsUnlimited() {
  UNIMPLEMENTED();
}

void SetAddressSpaceUnlimited() {
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
  if (::IsDebuggerPresent())
    __debugbreak();
  internal__exit(3);
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

uptr internal_ftruncate(fd_t fd, uptr size) {
  UNIMPLEMENTED();
}

uptr internal_rename(const char *oldpath, const char *newpath) {
  UNIMPLEMENTED();
}

uptr GetRSS() {
  return 0;
}

void *internal_start_thread(void (*func)(void *arg), void *arg) { return 0; }
void internal_join_thread(void *th) { }

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

#if !SANITIZER_GO
void BufferedStackTrace::SlowUnwindStack(uptr pc, uptr max_depth) {
  CHECK_GE(max_depth, 2);
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

void BufferedStackTrace::SlowUnwindStackWithContext(uptr pc, void *context,
                                                    uptr max_depth) {
  CONTEXT ctx = *(CONTEXT *)context;
  STACKFRAME64 stack_frame;
  memset(&stack_frame, 0, sizeof(stack_frame));
  size = 0;
#if defined(_WIN64)
  int machine_type = IMAGE_FILE_MACHINE_AMD64;
  stack_frame.AddrPC.Offset = ctx.Rip;
  stack_frame.AddrFrame.Offset = ctx.Rbp;
  stack_frame.AddrStack.Offset = ctx.Rsp;
#else
  int machine_type = IMAGE_FILE_MACHINE_I386;
  stack_frame.AddrPC.Offset = ctx.Eip;
  stack_frame.AddrFrame.Offset = ctx.Ebp;
  stack_frame.AddrStack.Offset = ctx.Esp;
#endif
  stack_frame.AddrPC.Mode = AddrModeFlat;
  stack_frame.AddrFrame.Mode = AddrModeFlat;
  stack_frame.AddrStack.Mode = AddrModeFlat;
  while (StackWalk64(machine_type, GetCurrentProcess(), GetCurrentThread(),
                     &stack_frame, &ctx, NULL, &SymFunctionTableAccess64,
                     &SymGetModuleBase64, NULL) &&
         size < Min(max_depth, kStackTraceMax)) {
    trace_buffer[size++] = (uptr)stack_frame.AddrPC.Offset;
  }
}
#endif  // #if !SANITIZER_GO

void ReportFile::Write(const char *buffer, uptr length) {
  SpinMutexLock l(mu);
  ReopenIfNecessary();
  if (length != internal_write(fd, buffer, length)) {
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

bool IsAccessibleMemoryRange(uptr beg, uptr size) {
  // FIXME: Actually implement this function.
  return true;
}

}  // namespace __sanitizer

#endif  // _WIN32
