//===- llvm/System/Unix/Program.cpp -----------------------------*- C++ -*-===//
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

#include <llvm/Config/config.h>
#include "Unix.h"
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

Program::Program() : Data_(0) {}

Program::~Program() {}

unsigned Program::GetPid() const {
  uint64_t pid = reinterpret_cast<uint64_t>(Data_);
  return static_cast<unsigned>(pid);
}

// This function just uses the PATH environment variable to find the program.
Path
Program::FindProgramByName(const std::string& progName) {

  // Check some degenerate cases
  if (progName.length() == 0) // no program
    return Path();
  Path temp;
  if (!temp.set(progName)) // invalid name
    return Path();
  // Use the given path verbatim if it contains any slashes; this matches
  // the behavior of sh(1) and friends.
  if (progName.find('/') != std::string::npos)
    return temp;

  // At this point, the file name does not contain slashes. Search for it
  // through the directories specified in the PATH environment variable.

  // Get the path. If its empty, we can't do anything to find it.
  const char *PathStr = getenv("PATH");
  if (PathStr == 0)
    return Path();

  // Now we have a colon separated list of directories to search; try them.
  size_t PathLen = strlen(PathStr);
  while (PathLen) {
    // Find the first colon...
    const char *Colon = std::find(PathStr, PathStr+PathLen, ':');

    // Check to see if this first directory contains the executable...
    Path FilePath;
    if (FilePath.set(std::string(PathStr,Colon))) {
      FilePath.appendComponent(progName);
      if (FilePath.canExecute())
        return FilePath;                    // Found the executable!
    }

    // Nope it wasn't in this directory, check the next path in the list!
    PathLen -= Colon-PathStr;
    PathStr = Colon;

    // Advance past duplicate colons
    while (*PathStr == ':') {
      PathStr++;
      PathLen--;
    }
  }
  return Path();
}

static bool RedirectIO(const Path *Path, int FD, std::string* ErrMsg) {
  if (Path == 0) // Noop
    return false;
  const char *File;
  if (Path->isEmpty())
    // Redirect empty paths to /dev/null
    File = "/dev/null";
  else
    File = Path->c_str();

  // Open the file
  int InFD = open(File, FD == 0 ? O_RDONLY : O_WRONLY|O_CREAT, 0666);
  if (InFD == -1) {
    MakeErrMsg(ErrMsg, "Cannot open file '" + std::string(File) + "' for "
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
static bool RedirectIO_PS(const Path *Path, int FD, std::string *ErrMsg,
                          posix_spawn_file_actions_t &FileActions) {
  if (Path == 0) // Noop
    return false;
  const char *File;
  if (Path->isEmpty())
    // Redirect empty paths to /dev/null
    File = "/dev/null";
  else
    File = Path->c_str();

  if (int Err = posix_spawn_file_actions_addopen(&FileActions, FD,
                            File, FD == 0 ? O_RDONLY : O_WRONLY|O_CREAT, 0666))
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
  // Virtual memory.
  getrlimit (RLIMIT_AS, &r);
  r.rlim_cur = limit;
  setrlimit (RLIMIT_AS, &r);
#endif
#endif
}

bool
Program::Execute(const Path &path, const char **args, const char **envp,
                 const Path **redirects, unsigned memoryLimit,
                  std::string *ErrMsg) {
  // If this OS has posix_spawn and there is no memory limit being implied, use
  // posix_spawn.  It is more efficient than fork/exec.
#ifdef HAVE_POSIX_SPAWN
  if (memoryLimit == 0) {
    posix_spawn_file_actions_t FileActions;
    posix_spawn_file_actions_init(&FileActions);

    if (redirects) {
      // Redirect stdin/stdout.
      if (RedirectIO_PS(redirects[0], 0, ErrMsg, FileActions) ||
          RedirectIO_PS(redirects[1], 1, ErrMsg, FileActions))
        return false;
      if (redirects[1] == 0 || redirects[2] == 0 ||
          *redirects[1] != *redirects[2]) {
        // Just redirect stderr
        if (RedirectIO_PS(redirects[2], 2, ErrMsg, FileActions)) return false;
      } else {       
        // If stdout and stderr should go to the same place, redirect stderr
        // to the FD already open for stdout.
        if (int Err = posix_spawn_file_actions_adddup2(&FileActions, 1, 2))
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

    pid_t PID;
    int Err = posix_spawn(&PID, path.c_str(), &FileActions, /*attrp*/0,
                          const_cast<char **>(args), const_cast<char **>(envp));
                          
    posix_spawn_file_actions_destroy(&FileActions);

    if (Err)
     return !MakeErrMsg(ErrMsg, "posix_spawn failed", Err);
      
    Data_ = reinterpret_cast<void*>(PID);
    return true;
  }
#endif
  
  if (!path.canExecute()) {
    if (ErrMsg)
      *ErrMsg = path.str() + " is not executable";
    return false;
  }

  // Create a child process.
  int child = fork();
  switch (child) {
    // An error occured:  Return to the caller.
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
      if (envp != 0)
        execve(path.c_str(),
               const_cast<char **>(args),
               const_cast<char **>(envp));
      else
        execv(path.c_str(),
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

  Data_ = reinterpret_cast<void*>(child);

  return true;
}

int
Program::Wait(unsigned secondsToWait,
              std::string* ErrMsg)
{
#ifdef HAVE_SYS_WAIT_H
  struct sigaction Act, Old;

  if (Data_ == 0) {
    MakeErrMsg(ErrMsg, "Process not started!");
    return -1;
  }

  // Install a timeout handler.  The handler itself does nothing, but the simple
  // fact of having a handler at all causes the wait below to return with EINTR,
  // unlike if we used SIG_IGN.
  if (secondsToWait) {
#ifndef __HAIKU__
    Act.sa_sigaction = 0;
#endif
    Act.sa_handler = TimeOutHandler;
    sigemptyset(&Act.sa_mask);
    Act.sa_flags = 0;
    sigaction(SIGALRM, &Act, &Old);
    alarm(secondsToWait);
  }

  // Parent process: Wait for the child process to terminate.
  int status;
  uint64_t pid = reinterpret_cast<uint64_t>(Data_);
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

      return -1;   // Timeout detected
    } else if (errno != EINTR) {
      MakeErrMsg(ErrMsg, "Error waiting for child process");
      return -1;
    }

  // We exited normally without timeout, so turn off the timer.
  if (secondsToWait) {
    alarm(0);
    sigaction(SIGALRM, &Old, 0);
  }

  // Return the proper exit status. 0=success, >0 is programs' exit status,
  // <0 means a signal was returned, -9999999 means the program dumped core.
  int result = 0;
  if (WIFEXITED(status))
    result = WEXITSTATUS(status);
  else if (WIFSIGNALED(status))
    result = 0 - WTERMSIG(status);
#ifdef WCOREDUMP
  else if (WCOREDUMP(status))
    result |= 0x01000000;
#endif
  return result;
#else
  return -99;
#endif

}

bool
Program::Kill(std::string* ErrMsg) {
  if (Data_ == 0) {
    MakeErrMsg(ErrMsg, "Process not started!");
    return true;
  }

  uint64_t pid64 = reinterpret_cast<uint64_t>(Data_);
  pid_t pid = static_cast<pid_t>(pid64);

  if (kill(pid, SIGKILL) != 0) {
    MakeErrMsg(ErrMsg, "The process couldn't be killed!");
    return true;
  }

  return false;
}

bool Program::ChangeStdinToBinary(){
  // Do nothing, as Unix doesn't differentiate between text and binary.
  return false;
}

bool Program::ChangeStdoutToBinary(){
  // Do nothing, as Unix doesn't differentiate between text and binary.
  return false;
}

bool Program::ChangeStderrToBinary(){
  // Do nothing, as Unix doesn't differentiate between text and binary.
  return false;
}

}
