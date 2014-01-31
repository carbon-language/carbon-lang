//===-- sanitizer_linux.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements linux-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_LINUX

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_linux.h"
#include "sanitizer_mutex.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"

#include <asm/param.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#if !SANITIZER_ANDROID
#include <link.h>
#endif
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <unwind.h>

#if !SANITIZER_ANDROID
#include <sys/signal.h>
#endif

#if SANITIZER_ANDROID
#include <android/log.h>
#endif

// <linux/time.h>
struct kernel_timeval {
  long tv_sec;
  long tv_usec;
};

// <linux/futex.h> is broken on some linux distributions.
const int FUTEX_WAIT = 0;
const int FUTEX_WAKE = 1;

// Are we using 32-bit or 64-bit syscalls?
// x32 (which defines __x86_64__) has SANITIZER_WORDSIZE == 32
// but it still needs to use 64-bit syscalls.
#if defined(__x86_64__) || SANITIZER_WORDSIZE == 64
# define SANITIZER_LINUX_USES_64BIT_SYSCALLS 1
#else
# define SANITIZER_LINUX_USES_64BIT_SYSCALLS 0
#endif

namespace __sanitizer {

#ifdef __x86_64__
#include "sanitizer_syscall_linux_x86_64.inc"
#else
#include "sanitizer_syscall_generic.inc"
#endif

// --------------- sanitizer_libc.h
uptr internal_mmap(void *addr, uptr length, int prot, int flags,
                    int fd, u64 offset) {
#if SANITIZER_LINUX_USES_64BIT_SYSCALLS
  return internal_syscall(__NR_mmap, (uptr)addr, length, prot, flags, fd,
                          offset);
#else
  return internal_syscall(__NR_mmap2, addr, length, prot, flags, fd, offset);
#endif
}

uptr internal_munmap(void *addr, uptr length) {
  return internal_syscall(__NR_munmap, (uptr)addr, length);
}

uptr internal_close(fd_t fd) {
  return internal_syscall(__NR_close, fd);
}

uptr internal_open(const char *filename, int flags) {
  return internal_syscall(__NR_open, (uptr)filename, flags);
}

uptr internal_open(const char *filename, int flags, u32 mode) {
  return internal_syscall(__NR_open, (uptr)filename, flags, mode);
}

uptr OpenFile(const char *filename, bool write) {
  return internal_open(filename,
      write ? O_WRONLY | O_CREAT /*| O_CLOEXEC*/ : O_RDONLY, 0660);
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  sptr res;
  HANDLE_EINTR(res, (sptr)internal_syscall(__NR_read, fd, (uptr)buf, count));
  return res;
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  sptr res;
  HANDLE_EINTR(res, (sptr)internal_syscall(__NR_write, fd, (uptr)buf, count));
  return res;
}

#if !SANITIZER_LINUX_USES_64BIT_SYSCALLS
static void stat64_to_stat(struct stat64 *in, struct stat *out) {
  internal_memset(out, 0, sizeof(*out));
  out->st_dev = in->st_dev;
  out->st_ino = in->st_ino;
  out->st_mode = in->st_mode;
  out->st_nlink = in->st_nlink;
  out->st_uid = in->st_uid;
  out->st_gid = in->st_gid;
  out->st_rdev = in->st_rdev;
  out->st_size = in->st_size;
  out->st_blksize = in->st_blksize;
  out->st_blocks = in->st_blocks;
  out->st_atime = in->st_atime;
  out->st_mtime = in->st_mtime;
  out->st_ctime = in->st_ctime;
  out->st_ino = in->st_ino;
}
#endif

uptr internal_stat(const char *path, void *buf) {
#if SANITIZER_LINUX_USES_64BIT_SYSCALLS
  return internal_syscall(__NR_stat, (uptr)path, (uptr)buf);
#else
  struct stat64 buf64;
  int res = internal_syscall(__NR_stat64, path, &buf64);
  stat64_to_stat(&buf64, (struct stat *)buf);
  return res;
#endif
}

uptr internal_lstat(const char *path, void *buf) {
#if SANITIZER_LINUX_USES_64BIT_SYSCALLS
  return internal_syscall(__NR_lstat, (uptr)path, (uptr)buf);
#else
  struct stat64 buf64;
  int res = internal_syscall(__NR_lstat64, path, &buf64);
  stat64_to_stat(&buf64, (struct stat *)buf);
  return res;
#endif
}

uptr internal_fstat(fd_t fd, void *buf) {
#if SANITIZER_LINUX_USES_64BIT_SYSCALLS
  return internal_syscall(__NR_fstat, fd, (uptr)buf);
#else
  struct stat64 buf64;
  int res = internal_syscall(__NR_fstat64, fd, &buf64);
  stat64_to_stat(&buf64, (struct stat *)buf);
  return res;
#endif
}

uptr internal_filesize(fd_t fd) {
  struct stat st;
  if (internal_fstat(fd, &st))
    return -1;
  return (uptr)st.st_size;
}

uptr internal_dup2(int oldfd, int newfd) {
  return internal_syscall(__NR_dup2, oldfd, newfd);
}

uptr internal_readlink(const char *path, char *buf, uptr bufsize) {
  return internal_syscall(__NR_readlink, (uptr)path, (uptr)buf, bufsize);
}

uptr internal_unlink(const char *path) {
  return internal_syscall(__NR_unlink, (uptr)path);
}

uptr internal_sched_yield() {
  return internal_syscall(__NR_sched_yield);
}

void internal__exit(int exitcode) {
  internal_syscall(__NR_exit_group, exitcode);
  Die();  // Unreachable.
}

uptr internal_execve(const char *filename, char *const argv[],
                     char *const envp[]) {
  return internal_syscall(__NR_execve, (uptr)filename, (uptr)argv, (uptr)envp);
}

// ----------------- sanitizer_common.h
bool FileExists(const char *filename) {
  struct stat st;
  if (internal_stat(filename, &st))
    return false;
  // Sanity check: filename is a regular file.
  return S_ISREG(st.st_mode);
}

uptr GetTid() {
  return internal_syscall(__NR_gettid);
}

u64 NanoTime() {
  kernel_timeval tv;
  internal_memset(&tv, 0, sizeof(tv));
  internal_syscall(__NR_gettimeofday, (uptr)&tv, 0);
  return (u64)tv.tv_sec * 1000*1000*1000 + tv.tv_usec * 1000;
}

// Like getenv, but reads env directly from /proc and does not use libc.
// This function should be called first inside __asan_init.
const char *GetEnv(const char *name) {
  static char *environ;
  static uptr len;
  static bool inited;
  if (!inited) {
    inited = true;
    uptr environ_size;
    len = ReadFileToBuffer("/proc/self/environ",
                           &environ, &environ_size, 1 << 26);
  }
  if (!environ || len == 0) return 0;
  uptr namelen = internal_strlen(name);
  const char *p = environ;
  while (*p != '\0') {  // will happen at the \0\0 that terminates the buffer
    // proc file has the format NAME=value\0NAME=value\0NAME=value\0...
    const char* endp =
        (char*)internal_memchr(p, '\0', len - (p - environ));
    if (endp == 0)  // this entry isn't NUL terminated
      return 0;
    else if (!internal_memcmp(p, name, namelen) && p[namelen] == '=')  // Match.
      return p + namelen + 1;  // point after =
    p = endp + 1;
  }
  return 0;  // Not found.
}

extern "C" {
  SANITIZER_WEAK_ATTRIBUTE extern void *__libc_stack_end;
}

#if !SANITIZER_GO
static void ReadNullSepFileToArray(const char *path, char ***arr,
                                   int arr_size) {
  char *buff;
  uptr buff_size = 0;
  *arr = (char **)MmapOrDie(arr_size * sizeof(char *), "NullSepFileArray");
  ReadFileToBuffer(path, &buff, &buff_size, 1024 * 1024);
  (*arr)[0] = buff;
  int count, i;
  for (count = 1, i = 1; ; i++) {
    if (buff[i] == 0) {
      if (buff[i+1] == 0) break;
      (*arr)[count] = &buff[i+1];
      CHECK_LE(count, arr_size - 1);  // FIXME: make this more flexible.
      count++;
    }
  }
  (*arr)[count] = 0;
}
#endif

static void GetArgsAndEnv(char*** argv, char*** envp) {
#if !SANITIZER_GO
  if (&__libc_stack_end) {
#endif
    uptr* stack_end = (uptr*)__libc_stack_end;
    int argc = *stack_end;
    *argv = (char**)(stack_end + 1);
    *envp = (char**)(stack_end + argc + 2);
#if !SANITIZER_GO
  } else {
    static const int kMaxArgv = 2000, kMaxEnvp = 2000;
    ReadNullSepFileToArray("/proc/self/cmdline", argv, kMaxArgv);
    ReadNullSepFileToArray("/proc/self/environ", envp, kMaxEnvp);
  }
#endif
}

void ReExec() {
  char **argv, **envp;
  GetArgsAndEnv(&argv, &envp);
  uptr rv = internal_execve("/proc/self/exe", argv, envp);
  int rverrno;
  CHECK_EQ(internal_iserror(rv, &rverrno), true);
  Printf("execve failed, errno %d\n", rverrno);
  Die();
}

void PrepareForSandboxing() {
  // Some kinds of sandboxes may forbid filesystem access, so we won't be able
  // to read the file mappings from /proc/self/maps. Luckily, neither the
  // process will be able to load additional libraries, so it's fine to use the
  // cached mappings.
  MemoryMappingLayout::CacheMemoryMappings();
  // Same for /proc/self/exe in the symbolizer.
#if !SANITIZER_GO
  if (Symbolizer *sym = Symbolizer::GetOrNull())
    sym->PrepareForSandboxing();
#endif
}

enum MutexState {
  MtxUnlocked = 0,
  MtxLocked = 1,
  MtxSleeping = 2
};

BlockingMutex::BlockingMutex(LinkerInitialized) {
  CHECK_EQ(owner_, 0);
}

BlockingMutex::BlockingMutex() {
  internal_memset(this, 0, sizeof(*this));
}

void BlockingMutex::Lock() {
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  if (atomic_exchange(m, MtxLocked, memory_order_acquire) == MtxUnlocked)
    return;
  while (atomic_exchange(m, MtxSleeping, memory_order_acquire) != MtxUnlocked)
    internal_syscall(__NR_futex, (uptr)m, FUTEX_WAIT, MtxSleeping, 0, 0, 0);
}

void BlockingMutex::Unlock() {
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  u32 v = atomic_exchange(m, MtxUnlocked, memory_order_relaxed);
  CHECK_NE(v, MtxUnlocked);
  if (v == MtxSleeping)
    internal_syscall(__NR_futex, (uptr)m, FUTEX_WAKE, 1, 0, 0, 0);
}

void BlockingMutex::CheckLocked() {
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  CHECK_NE(MtxUnlocked, atomic_load(m, memory_order_relaxed));
}

// ----------------- sanitizer_linux.h
// The actual size of this structure is specified by d_reclen.
// Note that getdents64 uses a different structure format. We only provide the
// 32-bit syscall here.
struct linux_dirent {
  unsigned long      d_ino;
  unsigned long      d_off;
  unsigned short     d_reclen;
  char               d_name[256];
};

// Syscall wrappers.
uptr internal_ptrace(int request, int pid, void *addr, void *data) {
  return internal_syscall(__NR_ptrace, request, pid, (uptr)addr, (uptr)data);
}

uptr internal_waitpid(int pid, int *status, int options) {
  return internal_syscall(__NR_wait4, pid, (uptr)status, options,
                          0 /* rusage */);
}

uptr internal_getpid() {
  return internal_syscall(__NR_getpid);
}

uptr internal_getppid() {
  return internal_syscall(__NR_getppid);
}

uptr internal_getdents(fd_t fd, struct linux_dirent *dirp, unsigned int count) {
  return internal_syscall(__NR_getdents, fd, (uptr)dirp, count);
}

uptr internal_lseek(fd_t fd, OFF_T offset, int whence) {
  return internal_syscall(__NR_lseek, fd, offset, whence);
}

uptr internal_prctl(int option, uptr arg2, uptr arg3, uptr arg4, uptr arg5) {
  return internal_syscall(__NR_prctl, option, arg2, arg3, arg4, arg5);
}

uptr internal_sigaltstack(const struct sigaltstack *ss,
                         struct sigaltstack *oss) {
  return internal_syscall(__NR_sigaltstack, (uptr)ss, (uptr)oss);
}

// Doesn't set sa_restorer, use with caution (see below).
int internal_sigaction_norestorer(int signum, const void *act, void *oldact) {
  __sanitizer_kernel_sigaction_t k_act, k_oldact;
  internal_memset(&k_act, 0, sizeof(__sanitizer_kernel_sigaction_t));
  internal_memset(&k_oldact, 0, sizeof(__sanitizer_kernel_sigaction_t));
  const __sanitizer_sigaction *u_act = (__sanitizer_sigaction *)act;
  __sanitizer_sigaction *u_oldact = (__sanitizer_sigaction *)oldact;
  if (u_act) {
    k_act.handler = u_act->handler;
    k_act.sigaction = u_act->sigaction;
    internal_memcpy(&k_act.sa_mask, &u_act->sa_mask,
                    sizeof(__sanitizer_kernel_sigset_t));
    k_act.sa_flags = u_act->sa_flags;
    // FIXME: most often sa_restorer is unset, however the kernel requires it
    // to point to a valid signal restorer that calls the rt_sigreturn syscall.
    // If sa_restorer passed to the kernel is NULL, the program may crash upon
    // signal delivery or fail to unwind the stack in the signal handler.
    // libc implementation of sigaction() passes its own restorer to
    // rt_sigaction, so we need to do the same (we'll need to reimplement the
    // restorers; for x86_64 the restorer address can be obtained from
    // oldact->sa_restorer upon a call to sigaction(xxx, NULL, oldact).
    k_act.sa_restorer = u_act->sa_restorer;
  }

  uptr result = internal_syscall(__NR_rt_sigaction, (uptr)signum,
      (uptr)(u_act ? &k_act : NULL),
      (uptr)(u_oldact ? &k_oldact : NULL),
      (uptr)sizeof(__sanitizer_kernel_sigset_t));

  if ((result == 0) && u_oldact) {
    u_oldact->handler = k_oldact.handler;
    u_oldact->sigaction = k_oldact.sigaction;
    internal_memcpy(&u_oldact->sa_mask, &k_oldact.sa_mask,
                    sizeof(__sanitizer_kernel_sigset_t));
    u_oldact->sa_flags = k_oldact.sa_flags;
    u_oldact->sa_restorer = k_oldact.sa_restorer;
  }
  return result;
}

uptr internal_sigprocmask(int how, __sanitizer_sigset_t *set,
    __sanitizer_sigset_t *oldset) {
  __sanitizer_kernel_sigset_t *k_set = (__sanitizer_kernel_sigset_t *)set;
  __sanitizer_kernel_sigset_t *k_oldset = (__sanitizer_kernel_sigset_t *)oldset;
  return internal_syscall(__NR_rt_sigprocmask, (uptr)how, &k_set->sig[0],
      &k_oldset->sig[0], sizeof(__sanitizer_kernel_sigset_t));
}

void internal_sigfillset(__sanitizer_sigset_t *set) {
  internal_memset(set, 0xff, sizeof(*set));
}

void internal_sigdelset(__sanitizer_sigset_t *set, int signum) {
  signum -= 1;
  CHECK_GE(signum, 0);
  CHECK_LT(signum, sizeof(*set) * 8);
  __sanitizer_kernel_sigset_t *k_set = (__sanitizer_kernel_sigset_t *)set;
  const uptr idx = signum / (sizeof(k_set->sig[0]) * 8);
  const uptr bit = signum % (sizeof(k_set->sig[0]) * 8);
  k_set->sig[idx] &= ~(1 << bit);
}

// ThreadLister implementation.
ThreadLister::ThreadLister(int pid)
  : pid_(pid),
    descriptor_(-1),
    buffer_(4096),
    error_(true),
    entry_((struct linux_dirent *)buffer_.data()),
    bytes_read_(0) {
  char task_directory_path[80];
  internal_snprintf(task_directory_path, sizeof(task_directory_path),
                    "/proc/%d/task/", pid);
  uptr openrv = internal_open(task_directory_path, O_RDONLY | O_DIRECTORY);
  if (internal_iserror(openrv)) {
    error_ = true;
    Report("Can't open /proc/%d/task for reading.\n", pid);
  } else {
    error_ = false;
    descriptor_ = openrv;
  }
}

int ThreadLister::GetNextTID() {
  int tid = -1;
  do {
    if (error_)
      return -1;
    if ((char *)entry_ >= &buffer_[bytes_read_] && !GetDirectoryEntries())
      return -1;
    if (entry_->d_ino != 0 && entry_->d_name[0] >= '0' &&
        entry_->d_name[0] <= '9') {
      // Found a valid tid.
      tid = (int)internal_atoll(entry_->d_name);
    }
    entry_ = (struct linux_dirent *)(((char *)entry_) + entry_->d_reclen);
  } while (tid < 0);
  return tid;
}

void ThreadLister::Reset() {
  if (error_ || descriptor_ < 0)
    return;
  internal_lseek(descriptor_, 0, SEEK_SET);
}

ThreadLister::~ThreadLister() {
  if (descriptor_ >= 0)
    internal_close(descriptor_);
}

bool ThreadLister::error() { return error_; }

bool ThreadLister::GetDirectoryEntries() {
  CHECK_GE(descriptor_, 0);
  CHECK_NE(error_, true);
  bytes_read_ = internal_getdents(descriptor_,
                                  (struct linux_dirent *)buffer_.data(),
                                  buffer_.size());
  if (internal_iserror(bytes_read_)) {
    Report("Can't read directory entries from /proc/%d/task.\n", pid_);
    error_ = true;
    return false;
  } else if (bytes_read_ == 0) {
    return false;
  }
  entry_ = (struct linux_dirent *)buffer_.data();
  return true;
}

uptr GetPageSize() {
#if defined(__x86_64__) || defined(__i386__)
  return EXEC_PAGESIZE;
#else
  return sysconf(_SC_PAGESIZE);  // EXEC_PAGESIZE may not be trustworthy.
#endif
}

static char proc_self_exe_cache_str[kMaxPathLength];
static uptr proc_self_exe_cache_len = 0;

uptr ReadBinaryName(/*out*/char *buf, uptr buf_len) {
  uptr module_name_len = internal_readlink(
      "/proc/self/exe", buf, buf_len);
  int readlink_error;
  if (internal_iserror(module_name_len, &readlink_error)) {
    if (proc_self_exe_cache_len) {
      // If available, use the cached module name.
      CHECK_LE(proc_self_exe_cache_len, buf_len);
      internal_strncpy(buf, proc_self_exe_cache_str, buf_len);
      module_name_len = internal_strlen(proc_self_exe_cache_str);
    } else {
      // We can't read /proc/self/exe for some reason, assume the name of the
      // binary is unknown.
      Report("WARNING: readlink(\"/proc/self/exe\") failed with errno %d, "
             "some stack frames may not be symbolized\n", readlink_error);
      module_name_len = internal_snprintf(buf, buf_len, "/proc/self/exe");
    }
    CHECK_LT(module_name_len, buf_len);
    buf[module_name_len] = '\0';
  }
  return module_name_len;
}

void CacheBinaryName() {
  if (!proc_self_exe_cache_len) {
    proc_self_exe_cache_len =
        ReadBinaryName(proc_self_exe_cache_str, kMaxPathLength);
  }
}

// Match full names of the form /path/to/base_name{-,.}*
bool LibraryNameIs(const char *full_name, const char *base_name) {
  const char *name = full_name;
  // Strip path.
  while (*name != '\0') name++;
  while (name > full_name && *name != '/') name--;
  if (*name == '/') name++;
  uptr base_name_length = internal_strlen(base_name);
  if (internal_strncmp(name, base_name, base_name_length)) return false;
  return (name[base_name_length] == '-' || name[base_name_length] == '.');
}

#if !SANITIZER_ANDROID
// Call cb for each region mapped by map.
void ForEachMappedRegion(link_map *map, void (*cb)(const void *, uptr)) {
  typedef ElfW(Phdr) Elf_Phdr;
  typedef ElfW(Ehdr) Elf_Ehdr;
  char *base = (char *)map->l_addr;
  Elf_Ehdr *ehdr = (Elf_Ehdr *)base;
  char *phdrs = base + ehdr->e_phoff;
  char *phdrs_end = phdrs + ehdr->e_phnum * ehdr->e_phentsize;

  // Find the segment with the minimum base so we can "relocate" the p_vaddr
  // fields.  Typically ET_DYN objects (DSOs) have base of zero and ET_EXEC
  // objects have a non-zero base.
  uptr preferred_base = (uptr)-1;
  for (char *iter = phdrs; iter != phdrs_end; iter += ehdr->e_phentsize) {
    Elf_Phdr *phdr = (Elf_Phdr *)iter;
    if (phdr->p_type == PT_LOAD && preferred_base > (uptr)phdr->p_vaddr)
      preferred_base = (uptr)phdr->p_vaddr;
  }

  // Compute the delta from the real base to get a relocation delta.
  sptr delta = (uptr)base - preferred_base;
  // Now we can figure out what the loader really mapped.
  for (char *iter = phdrs; iter != phdrs_end; iter += ehdr->e_phentsize) {
    Elf_Phdr *phdr = (Elf_Phdr *)iter;
    if (phdr->p_type == PT_LOAD) {
      uptr seg_start = phdr->p_vaddr + delta;
      uptr seg_end = seg_start + phdr->p_memsz;
      // None of these values are aligned.  We consider the ragged edges of the
      // load command as defined, since they are mapped from the file.
      seg_start = RoundDownTo(seg_start, GetPageSizeCached());
      seg_end = RoundUpTo(seg_end, GetPageSizeCached());
      cb((void *)seg_start, seg_end - seg_start);
    }
  }
}
#endif

#if defined(__x86_64__)
// We cannot use glibc's clone wrapper, because it messes with the child
// task's TLS. It writes the PID and TID of the child task to its thread
// descriptor, but in our case the child task shares the thread descriptor with
// the parent (because we don't know how to allocate a new thread
// descriptor to keep glibc happy). So the stock version of clone(), when
// used with CLONE_VM, would end up corrupting the parent's thread descriptor.
uptr internal_clone(int (*fn)(void *), void *child_stack, int flags, void *arg,
                    int *parent_tidptr, void *newtls, int *child_tidptr) {
  long long res;
  if (!fn || !child_stack)
    return -EINVAL;
  CHECK_EQ(0, (uptr)child_stack % 16);
  child_stack = (char *)child_stack - 2 * sizeof(unsigned long long);
  ((unsigned long long *)child_stack)[0] = (uptr)fn;
  ((unsigned long long *)child_stack)[1] = (uptr)arg;
  register void *r8 __asm__("r8") = newtls;
  register int *r10 __asm__("r10") = child_tidptr;
  __asm__ __volatile__(
                       /* %rax = syscall(%rax = __NR_clone,
                        *                %rdi = flags,
                        *                %rsi = child_stack,
                        *                %rdx = parent_tidptr,
                        *                %r8  = new_tls,
                        *                %r10 = child_tidptr)
                        */
                       "syscall\n"

                       /* if (%rax != 0)
                        *   return;
                        */
                       "testq  %%rax,%%rax\n"
                       "jnz    1f\n"

                       /* In the child. Terminate unwind chain. */
                       // XXX: We should also terminate the CFI unwind chain
                       // here. Unfortunately clang 3.2 doesn't support the
                       // necessary CFI directives, so we skip that part.
                       "xorq   %%rbp,%%rbp\n"

                       /* Call "fn(arg)". */
                       "popq   %%rax\n"
                       "popq   %%rdi\n"
                       "call   *%%rax\n"

                       /* Call _exit(%rax). */
                       "movq   %%rax,%%rdi\n"
                       "movq   %2,%%rax\n"
                       "syscall\n"

                       /* Return to parent. */
                     "1:\n"
                       : "=a" (res)
                       : "a"(__NR_clone), "i"(__NR_exit),
                         "S"(child_stack),
                         "D"(flags),
                         "d"(parent_tidptr),
                         "r"(r8),
                         "r"(r10)
                       : "rsp", "memory", "r11", "rcx");
  return res;
}
#endif  // defined(__x86_64__)

#if SANITIZER_ANDROID
// This thing is not, strictly speaking, async signal safe, but it does not seem
// to cause any issues. Alternative is writing to log devices directly, but
// their location and message format might change in the future, so we'd really
// like to avoid that.
void AndroidLogWrite(const char *buffer) {
  __android_log_write(ANDROID_LOG_INFO, NULL, buffer);
}
#endif

bool IsDeadlySignal(int signum) {
  return (signum == SIGSEGV) && common_flags()->handle_segv;
}

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX
