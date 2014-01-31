//===-- sanitizer_stoptheworld_linux_libcdep.cc ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// See sanitizer_stoptheworld.h for details.
// This implementation was inspired by Markus Gutschke's linuxthreads.cc.
//
//===----------------------------------------------------------------------===//


#include "sanitizer_platform.h"
#if SANITIZER_LINUX && defined(__x86_64__)

#include "sanitizer_stoptheworld.h"

#include "sanitizer_platform_limits_posix.h"

#include <errno.h>
#include <sched.h> // for CLONE_* definitions
#include <stddef.h>
#include <sys/prctl.h> // for PR_* definitions
#include <sys/ptrace.h> // for PTRACE_* definitions
#include <sys/types.h> // for pid_t
#if SANITIZER_ANDROID && defined(__arm__)
# include <linux/user.h>  // for pt_regs
#else
# include <sys/user.h>  // for user_regs_struct
#endif
#include <sys/wait.h> // for signal-related stuff

#ifdef sa_handler
# undef sa_handler
#endif

#ifdef sa_sigaction
# undef sa_sigaction
#endif

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_libc.h"
#include "sanitizer_linux.h"
#include "sanitizer_mutex.h"
#include "sanitizer_placement_new.h"

// This module works by spawning a Linux task which then attaches to every
// thread in the caller process with ptrace. This suspends the threads, and
// PTRACE_GETREGS can then be used to obtain their register state. The callback
// supplied to StopTheWorld() is run in the tracer task while the threads are
// suspended.
// The tracer task must be placed in a different thread group for ptrace to
// work, so it cannot be spawned as a pthread. Instead, we use the low-level
// clone() interface (we want to share the address space with the caller
// process, so we prefer clone() over fork()).
//
// We don't use any libc functions, relying instead on direct syscalls. There
// are two reasons for this:
// 1. calling a library function while threads are suspended could cause a
// deadlock, if one of the treads happens to be holding a libc lock;
// 2. it's generally not safe to call libc functions from the tracer task,
// because clone() does not set up a thread-local storage for it. Any
// thread-local variables used by libc will be shared between the tracer task
// and the thread which spawned it.

COMPILER_CHECK(sizeof(SuspendedThreadID) == sizeof(pid_t));

namespace __sanitizer {
// This class handles thread suspending/unsuspending in the tracer thread.
class ThreadSuspender {
 public:
  explicit ThreadSuspender(pid_t pid)
    : pid_(pid) {
      CHECK_GE(pid, 0);
    }
  bool SuspendAllThreads();
  void ResumeAllThreads();
  void KillAllThreads();
  SuspendedThreadsList &suspended_threads_list() {
    return suspended_threads_list_;
  }
 private:
  SuspendedThreadsList suspended_threads_list_;
  pid_t pid_;
  bool SuspendThread(SuspendedThreadID thread_id);
};

bool ThreadSuspender::SuspendThread(SuspendedThreadID thread_id) {
  // Are we already attached to this thread?
  // Currently this check takes linear time, however the number of threads is
  // usually small.
  if (suspended_threads_list_.Contains(thread_id))
    return false;
  int pterrno;
  if (internal_iserror(internal_ptrace(PTRACE_ATTACH, thread_id, NULL, NULL),
                       &pterrno)) {
    // Either the thread is dead, or something prevented us from attaching.
    // Log this event and move on.
    VReport(1, "Could not attach to thread %d (errno %d).\n", thread_id,
            pterrno);
    return false;
  } else {
    VReport(1, "Attached to thread %d.\n", thread_id);
    // The thread is not guaranteed to stop before ptrace returns, so we must
    // wait on it.
    uptr waitpid_status;
    HANDLE_EINTR(waitpid_status, internal_waitpid(thread_id, NULL, __WALL));
    int wperrno;
    if (internal_iserror(waitpid_status, &wperrno)) {
      // Got a ECHILD error. I don't think this situation is possible, but it
      // doesn't hurt to report it.
      VReport(1, "Waiting on thread %d failed, detaching (errno %d).\n",
              thread_id, wperrno);
      internal_ptrace(PTRACE_DETACH, thread_id, NULL, NULL);
      return false;
    }
    suspended_threads_list_.Append(thread_id);
    return true;
  }
}

void ThreadSuspender::ResumeAllThreads() {
  for (uptr i = 0; i < suspended_threads_list_.thread_count(); i++) {
    pid_t tid = suspended_threads_list_.GetThreadID(i);
    int pterrno;
    if (!internal_iserror(internal_ptrace(PTRACE_DETACH, tid, NULL, NULL),
                          &pterrno)) {
      VReport(1, "Detached from thread %d.\n", tid);
    } else {
      // Either the thread is dead, or we are already detached.
      // The latter case is possible, for instance, if this function was called
      // from a signal handler.
      VReport(1, "Could not detach from thread %d (errno %d).\n", tid, pterrno);
    }
  }
}

void ThreadSuspender::KillAllThreads() {
  for (uptr i = 0; i < suspended_threads_list_.thread_count(); i++)
    internal_ptrace(PTRACE_KILL, suspended_threads_list_.GetThreadID(i),
                    NULL, NULL);
}

bool ThreadSuspender::SuspendAllThreads() {
  ThreadLister thread_lister(pid_);
  bool added_threads;
  do {
    // Run through the directory entries once.
    added_threads = false;
    pid_t tid = thread_lister.GetNextTID();
    while (tid >= 0) {
      if (SuspendThread(tid))
        added_threads = true;
      tid = thread_lister.GetNextTID();
    }
    if (thread_lister.error()) {
      // Detach threads and fail.
      ResumeAllThreads();
      return false;
    }
    thread_lister.Reset();
  } while (added_threads);
  return true;
}

// Pointer to the ThreadSuspender instance for use in signal handler.
static ThreadSuspender *thread_suspender_instance = NULL;

// Signals that should not be blocked (this is used in the parent thread as well
// as the tracer thread).
static const int kUnblockedSignals[] = { SIGABRT, SIGILL, SIGFPE, SIGSEGV,
                                         SIGBUS, SIGXCPU, SIGXFSZ };

// Structure for passing arguments into the tracer thread.
struct TracerThreadArgument {
  StopTheWorldCallback callback;
  void *callback_argument;
  // The tracer thread waits on this mutex while the parent finishes its
  // preparations.
  BlockingMutex mutex;
  uptr parent_pid;
};

static DieCallbackType old_die_callback;

// Signal handler to wake up suspended threads when the tracer thread dies.
void TracerThreadSignalHandler(int signum, void *siginfo, void *) {
  if (thread_suspender_instance != NULL) {
    if (signum == SIGABRT)
      thread_suspender_instance->KillAllThreads();
    else
      thread_suspender_instance->ResumeAllThreads();
  }
  internal__exit((signum == SIGABRT) ? 1 : 2);
}

static void TracerThreadDieCallback() {
  // Generally a call to Die() in the tracer thread should be fatal to the
  // parent process as well, because they share the address space.
  // This really only works correctly if all the threads are suspended at this
  // point. So we correctly handle calls to Die() from within the callback, but
  // not those that happen before or after the callback. Hopefully there aren't
  // a lot of opportunities for that to happen...
  if (thread_suspender_instance)
    thread_suspender_instance->KillAllThreads();
  if (old_die_callback)
    old_die_callback();
}

// Size of alternative stack for signal handlers in the tracer thread.
static const int kHandlerStackSize = 4096;

// This function will be run as a cloned task.
static int TracerThread(void* argument) {
  TracerThreadArgument *tracer_thread_argument =
      (TracerThreadArgument *)argument;

  internal_prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0);
  // Check if parent is already dead.
  if (internal_getppid() != tracer_thread_argument->parent_pid)
    internal__exit(4);

  // Wait for the parent thread to finish preparations.
  tracer_thread_argument->mutex.Lock();
  tracer_thread_argument->mutex.Unlock();

  SetDieCallback(TracerThreadDieCallback);

  ThreadSuspender thread_suspender(internal_getppid());
  // Global pointer for the signal handler.
  thread_suspender_instance = &thread_suspender;

  // Alternate stack for signal handling.
  InternalScopedBuffer<char> handler_stack_memory(kHandlerStackSize);
  struct sigaltstack handler_stack;
  internal_memset(&handler_stack, 0, sizeof(handler_stack));
  handler_stack.ss_sp = handler_stack_memory.data();
  handler_stack.ss_size = kHandlerStackSize;
  internal_sigaltstack(&handler_stack, NULL);

  // Install our handler for fatal signals. Other signals should be blocked by
  // the mask we inherited from the caller thread.
  for (uptr signal_index = 0; signal_index < ARRAY_SIZE(kUnblockedSignals);
       signal_index++) {
    __sanitizer_sigaction new_sigaction;
    internal_memset(&new_sigaction, 0, sizeof(new_sigaction));
    new_sigaction.sigaction = TracerThreadSignalHandler;
    new_sigaction.sa_flags = SA_ONSTACK | SA_SIGINFO;
    internal_sigfillset(&new_sigaction.sa_mask);
    internal_sigaction_norestorer(kUnblockedSignals[signal_index],
                                  &new_sigaction, NULL);
  }

  int exit_code = 0;
  if (!thread_suspender.SuspendAllThreads()) {
    VReport(1, "Failed suspending threads.\n");
    exit_code = 3;
  } else {
    tracer_thread_argument->callback(thread_suspender.suspended_threads_list(),
                                     tracer_thread_argument->callback_argument);
    thread_suspender.ResumeAllThreads();
    exit_code = 0;
  }
  thread_suspender_instance = NULL;
  handler_stack.ss_flags = SS_DISABLE;
  internal_sigaltstack(&handler_stack, NULL);
  return exit_code;
}

class ScopedStackSpaceWithGuard {
 public:
  explicit ScopedStackSpaceWithGuard(uptr stack_size) {
    stack_size_ = stack_size;
    guard_size_ = GetPageSizeCached();
    // FIXME: Omitting MAP_STACK here works in current kernels but might break
    // in the future.
    guard_start_ = (uptr)MmapOrDie(stack_size_ + guard_size_,
                                   "ScopedStackWithGuard");
    CHECK_EQ(guard_start_, (uptr)Mprotect((uptr)guard_start_, guard_size_));
  }
  ~ScopedStackSpaceWithGuard() {
    UnmapOrDie((void *)guard_start_, stack_size_ + guard_size_);
  }
  void *Bottom() const {
    return (void *)(guard_start_ + stack_size_ + guard_size_);
  }

 private:
  uptr stack_size_;
  uptr guard_size_;
  uptr guard_start_;
};

// We have a limitation on the stack frame size, so some stuff had to be moved
// into globals.
static __sanitizer_sigset_t blocked_sigset;
static __sanitizer_sigset_t old_sigset;
static __sanitizer_sigaction old_sigactions
    [ARRAY_SIZE(kUnblockedSignals)];

class StopTheWorldScope {
 public:
  StopTheWorldScope() {
    // Block all signals that can be blocked safely, and install
    // default handlers for the remaining signals.
    // We cannot allow user-defined handlers to run while the ThreadSuspender
    // thread is active, because they could conceivably call some libc functions
    // which modify errno (which is shared between the two threads).
    internal_sigfillset(&blocked_sigset);
    for (uptr signal_index = 0; signal_index < ARRAY_SIZE(kUnblockedSignals);
         signal_index++) {
      // Remove the signal from the set of blocked signals.
      internal_sigdelset(&blocked_sigset, kUnblockedSignals[signal_index]);
      // Install the default handler.
      __sanitizer_sigaction new_sigaction;
      internal_memset(&new_sigaction, 0, sizeof(new_sigaction));
      new_sigaction.handler = SIG_DFL;
      internal_sigfillset(&new_sigaction.sa_mask);
      internal_sigaction_norestorer(kUnblockedSignals[signal_index],
          &new_sigaction, &old_sigactions[signal_index]);
    }
    int sigprocmask_status =
        internal_sigprocmask(SIG_BLOCK, &blocked_sigset, &old_sigset);
    CHECK_EQ(sigprocmask_status, 0); // sigprocmask should never fail
    // Make this process dumpable. Processes that are not dumpable cannot be
    // attached to.
    process_was_dumpable_ = internal_prctl(PR_GET_DUMPABLE, 0, 0, 0, 0);
    if (!process_was_dumpable_)
      internal_prctl(PR_SET_DUMPABLE, 1, 0, 0, 0);
    old_die_callback = GetDieCallback();
  }

  ~StopTheWorldScope() {
    SetDieCallback(old_die_callback);
    // Restore the dumpable flag.
    if (!process_was_dumpable_)
      internal_prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
    // Restore the signal handlers.
    for (uptr signal_index = 0; signal_index < ARRAY_SIZE(kUnblockedSignals);
         signal_index++) {
      internal_sigaction_norestorer(kUnblockedSignals[signal_index],
                                    &old_sigactions[signal_index], NULL);
    }
    internal_sigprocmask(SIG_SETMASK, &old_sigset, &old_sigset);
  }

 private:
  int process_was_dumpable_;
};

// When sanitizer output is being redirected to file (i.e. by using log_path),
// the tracer should write to the parent's log instead of trying to open a new
// file. Alert the logging code to the fact that we have a tracer.
struct ScopedSetTracerPID {
  explicit ScopedSetTracerPID(uptr tracer_pid) {
    stoptheworld_tracer_pid = tracer_pid;
    stoptheworld_tracer_ppid = internal_getpid();
  }
  ~ScopedSetTracerPID() {
    stoptheworld_tracer_pid = 0;
    stoptheworld_tracer_ppid = 0;
  }
};

void StopTheWorld(StopTheWorldCallback callback, void *argument) {
  StopTheWorldScope in_stoptheworld;
  // Prepare the arguments for TracerThread.
  struct TracerThreadArgument tracer_thread_argument;
  tracer_thread_argument.callback = callback;
  tracer_thread_argument.callback_argument = argument;
  tracer_thread_argument.parent_pid = internal_getpid();
  const uptr kTracerStackSize = 2 * 1024 * 1024;
  ScopedStackSpaceWithGuard tracer_stack(kTracerStackSize);
  // Block the execution of TracerThread until after we have set ptrace
  // permissions.
  tracer_thread_argument.mutex.Lock();
  uptr tracer_pid = internal_clone(
      TracerThread, tracer_stack.Bottom(),
      CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_UNTRACED,
      &tracer_thread_argument, 0 /* parent_tidptr */, 0 /* newtls */, 0
      /* child_tidptr */);
  int local_errno = 0;
  if (internal_iserror(tracer_pid, &local_errno)) {
    VReport(1, "Failed spawning a tracer thread (errno %d).\n", local_errno);
    tracer_thread_argument.mutex.Unlock();
  } else {
    ScopedSetTracerPID scoped_set_tracer_pid(tracer_pid);
    // On some systems we have to explicitly declare that we want to be traced
    // by the tracer thread.
#ifdef PR_SET_PTRACER
    internal_prctl(PR_SET_PTRACER, tracer_pid, 0, 0, 0);
#endif
    // Allow the tracer thread to start.
    tracer_thread_argument.mutex.Unlock();
    // Since errno is shared between this thread and the tracer thread, we
    // must avoid using errno while the tracer thread is running.
    // At this point, any signal will either be blocked or kill us, so waitpid
    // should never return (and set errno) while the tracer thread is alive.
    uptr waitpid_status = internal_waitpid(tracer_pid, NULL, __WALL);
    if (internal_iserror(waitpid_status, &local_errno))
      VReport(1, "Waiting on the tracer thread failed (errno %d).\n",
              local_errno);
  }
}

// Platform-specific methods from SuspendedThreadsList.
#if SANITIZER_ANDROID && defined(__arm__)
typedef pt_regs regs_struct;
#define REG_SP ARM_sp

#elif SANITIZER_LINUX && defined(__arm__)
typedef user_regs regs_struct;
#define REG_SP uregs[13]

#elif defined(__i386__) || defined(__x86_64__)
typedef user_regs_struct regs_struct;
#if defined(__i386__)
#define REG_SP esp
#else
#define REG_SP rsp
#endif

#elif defined(__powerpc__) || defined(__powerpc64__)
typedef pt_regs regs_struct;
#define REG_SP gpr[PT_R1]

#elif defined(__mips__)
typedef struct user regs_struct;
#define REG_SP regs[EF_REG29]

#else
#error "Unsupported architecture"
#endif // SANITIZER_ANDROID && defined(__arm__)

int SuspendedThreadsList::GetRegistersAndSP(uptr index,
                                            uptr *buffer,
                                            uptr *sp) const {
  pid_t tid = GetThreadID(index);
  regs_struct regs;
  int pterrno;
  if (internal_iserror(internal_ptrace(PTRACE_GETREGS, tid, NULL, &regs),
                       &pterrno)) {
    VReport(1, "Could not get registers from thread %d (errno %d).\n", tid,
            pterrno);
    return -1;
  }

  *sp = regs.REG_SP;
  internal_memcpy(buffer, &regs, sizeof(regs));
  return 0;
}

uptr SuspendedThreadsList::RegisterCount() {
  return sizeof(regs_struct) / sizeof(uptr);
}
}  // namespace __sanitizer

#endif  // SANITIZER_LINUX && defined(__x86_64__)
