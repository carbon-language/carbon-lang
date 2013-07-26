//===- llvm/Support/Unix/Program.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific portion of the Program class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FileSystem.h"
#include <llvm/Config/config.h>
#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#if HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif
#if HAVE_SIGNAL_H
#include <signal.h>
#endif
#if HAVE_FCNTL_H
#include <fcntl.h>
#endif
#if HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_POSIX_SPAWN
#include <spawn.h>
#if !defined(__APPLE__)
  extern char **environ;
#else
#include <crt_externs.h> // _NSGetEnviron
#endif
#endif

namespace llvm {
using namespace sys;

// This function just uses the PATH environment variable to find the program.
std::string
sys::FindProgramByName(const std::string& progName) {

  // Check some degenerate cases
  if (progName.length() == 0) // no program
    return "";
  std::string temp = progName;
  // Use the given path verbatim if it contains any slashes; this matches
  // the behavior of sh(1) and friends.
  if (progName.find('/') != std::string::npos)
    return temp;

  // At this point, the file name is valid and does not contain slashes. Search
  // for it through the directories specified in the PATH environment variable.

  // Get the path. If its empty, we can't do anything to find it.
  const char *PathStr = getenv("PATH");
  if (PathStr == 0)
    return "";

  // Now we have a colon separated list of directories to search; try them.
  size_t PathLen = strlen(PathStr);
  while (PathLen) {
    // Find the first colon...
    const char *Colon = std::find(PathStr, PathStr+PathLen, ':');

    // Check to see if this first directory contains the executable...
    SmallString<128> FilePath(PathStr,Colon);
    sys::path::append(FilePath, progName);
    if (sys::fs::can_execute(Twine(FilePath)))
      return FilePath.str();                    // Found the executable!

    // Nope it wasn't in this directory, check the next path in the list!
    PathLen -= Colon-PathStr;
    PathStr = Colon;

    // Advance past duplicate colons
    while (*PathStr == ':') {
      PathStr++;
      PathLen--;
    }
  }
  return "";
}

static bool RedirectIO(const StringRef *Path, int FD, std::string* ErrMsg) {
  if (Path == 0) // Noop
    return false;
  std::string File;
  if (Path->empty())
    // Redirect empty paths to /dev/null
    File = "/dev/null";
  else
    File = *Path;

  // Open the file
  int InFD = open(File.c_str(), FD == 0 ? O_RDONLY : O_WRONLY|O_CREAT, 0666);
  if (InFD == -1) {
    MakeErrMsg(ErrMsg, "Cannot open file '" + File + "' for "
              + (FD == 0 ? "input" : "output"));
    return true;
  }

  // Install it as the requested FD
  if (dup2(InFD, FD) == -1) {
    MakeErrMsg(ErrMsg, "Cannot dup2");
    close(InFD);
    return true;
  }
  close(InFD);      // Close the original FD
  return false;
}

#ifdef HAVE_POSIX_SPAWN
static bool RedirectIO_PS(const std::string *Path, int FD, std::string *ErrMsg,
                          posix_spawn_file_actions_t *FileActions) {
  if (Path == 0) // Noop
    return false;
  const char *File;
  if (Path->empty())
    // Redirect empty paths to /dev/null
    File = "/dev/null";
  else
    File = Path->c_str();

  if (int Err = posix_spawn_file_actions_addopen(
          FileActions, FD, File,
          FD == 0 ? O_RDONLY : O_WRONLY | O_CREAT, 0666))
    return MakeErrMsg(ErrMsg, "Cannot dup2", Err);
  return false;
}
#endif

static void TimeOutHandler(int Sig) {
}

static void SetMemoryLimits (unsigned size)
{
#if HAVE_SYS_RESOURCE_H && HAVE_GETRLIMIT && HAVE_SETRLIMIT
  struct rlimit r;
  __typeof__ (r.rlim_cur) limit = (__typeof__ (r.rlim_cur)) (size) * 1048576;

  // Heap size
  getrlimit (RLIMIT_DATA, &r);
  r.rlim_cur = limit;
  setrlimit (RLIMIT_DATA, &r);
#ifdef RLIMIT_RSS
  // Resident set size.
  getrlimit (RLIMIT_RSS, &r);
  r.rlim_cur = limit;
  setrlimit (RLIMIT_RSS, &r);
#endif
#ifdef RLIMIT_AS  // e.g. NetBSD doesn't have it.
  // Don't set virtual memory limit if built with any Sanitizer. They need 80Tb
  // of virtual memory for shadow memory mapping.
#if !LLVM_MEMORY_SANITIZER_BUILD && !LLVM_ADDRESS_SANITIZER_BUILD
  // Virtual memory.
  getrlimit (RLIMIT_AS, &r);
  r.rlim_cur = limit;
  setrlimit (RLIMIT_AS, &r);
#endif
#endif
#endif
}

}

static bool Execute(void **Data, StringRef Program, const char **args,
                    const char **envp, const StringRef **redirects,
                    unsigned memoryLimit, std::string *ErrMsg) {
  // If this OS has posix_spawn and there is no memory limit being implied, use
  // posix_spawn.  It is more efficient than fork/exec.
#ifdef HAVE_POSIX_SPAWN
  if (memoryLimit == 0) {
    posix_spawn_file_actions_t FileActionsStore;
    posix_spawn_file_actions_t *FileActions = 0;

    // If we call posix_spawn_file_actions_addopen we have to make sure the
    // c strings we pass to it stay alive until the call to posix_spawn,
    // so we copy any StringRefs into this variable.
    std::string RedirectsStorage[3];

    if (redirects) {
      std::string *RedirectsStr[3] = {0, 0, 0};
      for (int I = 0; I < 3; ++I) {
        if (redirects[I]) {
          RedirectsStorage[I] = *redirects[I];
          RedirectsStr[I] = &RedirectsStorage[I];
        }
      }

      FileActions = &FileActionsStore;
      posix_spawn_file_actions_init(FileActions);

      // Redirect stdin/stdout.
      if (RedirectIO_PS(RedirectsStr[0], 0, ErrMsg, FileActions) ||
          RedirectIO_PS(RedirectsStr[1], 1, ErrMsg, FileActions))
        return false;
      if (redirects[1] == 0 || redirects[2] == 0 ||
          *redirects[1] != *redirects[2]) {
        // Just redirect stderr
        if (RedirectIO_PS(RedirectsStr[2], 2, ErrMsg, FileActions))
          return false;
      } else {
        // If stdout and stderr should go to the same place, redirect stderr
        // to the FD already open for stdout.
        if (int Err = posix_spawn_file_actions_adddup2(FileActions, 1, 2))
          return !MakeErrMsg(ErrMsg, "Can't redirect stderr to stdout", Err);
      }
    }

    if (!envp)
#if !defined(__APPLE__)
      envp = const_cast<const char **>(environ);
#else
      // environ is missing in dylibs.
      envp = const_cast<const char **>(*_NSGetEnviron());
#endif

    // Explicitly initialized to prevent what appears to be a valgrind false
    // positive.
    pid_t PID = 0;
    int Err = posix_spawn(&PID, Program.str().c_str(), FileActions, /*attrp*/0,
                          const_cast<char **>(args), const_cast<char **>(envp));

    if (FileActions)
      posix_spawn_file_actions_destroy(FileActions);

    if (Err)
     return !MakeErrMsg(ErrMsg, "posix_spawn failed", Err);

    if (Data)
      *Data = reinterpret_cast<void*>(PID);
    return true;
  }
#endif

  // Create a child process.
  int child = fork();
  switch (child) {
    // An error occurred:  Return to the caller.
    case -1:
      MakeErrMsg(ErrMsg, "Couldn't fork");
      return false;

    // Child process: Execute the program.
    case 0: {
      // Redirect file descriptors...
      if (redirects) {
        // Redirect stdin
        if (RedirectIO(redirects[0], 0, ErrMsg)) { return false; }
        // Redirect stdout
        if (RedirectIO(redirects[1], 1, ErrMsg)) { return false; }
        if (redirects[1] && redirects[2] &&
            *(redirects[1]) == *(redirects[2])) {
          // If stdout and stderr should go to the same place, redirect stderr
          // to the FD already open for stdout.
          if (-1 == dup2(1,2)) {
            MakeErrMsg(ErrMsg, "Can't redirect stderr to stdout");
            return false;
          }
        } else {
          // Just redirect stderr
          if (RedirectIO(redirects[2], 2, ErrMsg)) { return false; }
        }
      }

      // Set memory limits
      if (memoryLimit!=0) {
        SetMemoryLimits(memoryLimit);
      }

      // Execute!
      std::string PathStr = Program;
      if (envp != 0)
        execve(PathStr.c_str(),
               const_cast<char **>(args),
               const_cast<char **>(envp));
      else
        execv(PathStr.c_str(),
              const_cast<char **>(args));
      // If the execve() failed, we should exit. Follow Unix protocol and
      // return 127 if the executable was not found, and 126 otherwise.
      // Use _exit rather than exit so that atexit functions and static
      // object destructors cloned from the parent process aren't
      // redundantly run, and so that any data buffered in stdio buffers
      // cloned from the parent aren't redundantly written out.
      _exit(errno == ENOENT ? 127 : 126);
    }

    // Parent process: Break out of the switch to do our processing.
    default:
      break;
  }

  if (Data)
    *Data = reinterpret_cast<void*>(child);

  return true;
}

static int Wait(void *&Data, StringRef Program, unsigned secondsToWait,
                std::string *ErrMsg) {
#ifdef HAVE_SYS_WAIT_H
  struct sigaction Act, Old;
  assert(Data && "invalid pid to wait on, process not started?");

  // Install a timeout handler.  The handler itself does nothing, but the simple
  // fact of having a handler at all causes the wait below to return with EINTR,
  // unlike if we used SIG_IGN.
  if (secondsToWait) {
    memset(&Act, 0, sizeof(Act));
    Act.sa_handler = TimeOutHandler;
    sigemptyset(&Act.sa_mask);
    sigaction(SIGALRM, &Act, &Old);
    alarm(secondsToWait);
  }

  // Parent process: Wait for the child process to terminate.
  int status;
  uint64_t pid = reinterpret_cast<uint64_t>(Data);
  pid_t child = static_cast<pid_t>(pid);
  while (waitpid(pid, &status, 0) != child)
    if (secondsToWait && errno == EINTR) {
      // Kill the child.
      kill(child, SIGKILL);

      // Turn off the alarm and restore the signal handler
      alarm(0);
      sigaction(SIGALRM, &Old, 0);

      // Wait for child to die
      if (wait(&status) != child)
        MakeErrMsg(ErrMsg, "Child timed out but wouldn't die");
      else
        MakeErrMsg(ErrMsg, "Child timed out", 0);

      return -2;   // Timeout detected
    } else if (errno != EINTR) {
      MakeErrMsg(ErrMsg, "Error waiting for child process");
      return -1;
    }

  // We exited normally without timeout, so turn off the timer.
  if (secondsToWait) {
    alarm(0);
    sigaction(SIGALRM, &Old, 0);
  }

  // Return the proper exit status. Detect error conditions
  // so we can return -1 for them and set ErrMsg informatively.
  int result = 0;
  if (WIFEXITED(status)) {
    result = WEXITSTATUS(status);
#ifdef HAVE_POSIX_SPAWN
    // The posix_spawn child process returns 127 on any kind of error.
    // Following the POSIX convention for command-line tools (which posix_spawn
    // itself apparently does not), check to see if the failure was due to some
    // reason other than the file not existing, and return 126 in this case.
    bool Exists;
    if (result == 127 && !llvm::sys::fs::exists(Program, Exists) && Exists)
      result = 126;
#endif
    if (result == 127) {
      if (ErrMsg)
        *ErrMsg = llvm::sys::StrError(ENOENT);
      return -1;
    }
    if (result == 126) {
      if (ErrMsg)
        *ErrMsg = "Program could not be executed";
      return -1;
    }
  } else if (WIFSIGNALED(status)) {
    if (ErrMsg) {
      *ErrMsg = strsignal(WTERMSIG(status));
#ifdef WCOREDUMP
      if (WCOREDUMP(status))
        *ErrMsg += " (core dumped)";
#endif
    }
    // Return a special value to indicate that the process received an unhandled
    // signal during execution as opposed to failing to execute.
    return -2;
  }
  return result;
#else
  if (ErrMsg)
    *ErrMsg = "Program::Wait is not implemented on this platform yet!";
  return -1;
#endif
}

namespace llvm {

error_code sys::ChangeStdinToBinary(){
  // Do nothing, as Unix doesn't differentiate between text and binary.
  return make_error_code(errc::success);
}

error_code sys::ChangeStdoutToBinary(){
  // Do nothing, as Unix doesn't differentiate between text and binary.
  return make_error_code(errc::success);
}

error_code sys::ChangeStderrToBinary(){
  // Do nothing, as Unix doesn't differentiate between text and binary.
  return make_error_code(errc::success);
}

bool llvm::sys::argumentsFitWithinSystemLimits(ArrayRef<const char*> Args) {
  static long ArgMax = sysconf(_SC_ARG_MAX);

  // System says no practical limit.
  if (ArgMax == -1)
    return true;

  // Conservatively account for space required by environment variables.
  ArgMax /= 2;

  size_t ArgLength = 0;
  for (ArrayRef<const char*>::iterator I = Args.begin(), E = Args.end();
       I != E; ++I) {
    ArgLength += strlen(*I) + 1;
    if (ArgLength > size_t(ArgMax)) {
      return false;
    }
  }
  return true;
}

}
