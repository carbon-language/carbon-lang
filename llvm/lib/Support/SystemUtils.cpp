//===- SystemUtils.cpp - Utilities for low-level system tasks -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains functions used to do a variety of low-level, often
// system-specific, tasks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SystemUtils.h"
#include "llvm/System/Program.h"
#include <unistd.h>
#include <wait.h>
#include <sys/fcntl.h>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <signal.h>
using namespace llvm;

/// isStandardOutAConsole - Return true if we can tell that the standard output
/// stream goes to a terminal window or console.
bool llvm::isStandardOutAConsole() {
#if HAVE_ISATTY
  return isatty(1);
#endif
  // If we don't have isatty, just return false.
  return false;
}


/// FindExecutable - Find a named executable, giving the argv[0] of program
/// being executed. This allows us to find another LLVM tool if it is built
/// into the same directory, but that directory is neither the current
/// directory, nor in the PATH.  If the executable cannot be found, return an
/// empty string.
/// 
#undef FindExecutable   // needed on windows :(
sys::Path llvm::FindExecutable(const std::string &ExeName,
                                 const std::string &ProgramPath) {
  // First check the directory that the calling program is in.  We can do this 
  // if ProgramPath contains at least one / character, indicating that it is a
  // relative path to bugpoint itself.
  //
  sys::Path Result ( ProgramPath );
  Result.elideFile();

  if (!Result.isEmpty()) {
    Result.appendFile(ExeName);
    if (Result.executable()) return Result;
  }

  return sys::Program::FindProgramByName(ExeName);
}

static void RedirectFD(const std::string &File, int FD) {
  if (File.empty()) return;  // Noop

  // Open the file
  int InFD = open(File.c_str(), FD == 0 ? O_RDONLY : O_WRONLY|O_CREAT, 0666);
  if (InFD == -1) {
    std::cerr << "Error opening file '" << File << "' for "
              << (FD == 0 ? "input" : "output") << "!\n";
    exit(1);
  }

  dup2(InFD, FD);   // Install it as the requested FD
  close(InFD);      // Close the original FD
}

static bool Timeout = false;
static void TimeOutHandler(int Sig) {
  Timeout = true;
}

/// RunProgramWithTimeout - This function executes the specified program, with
/// the specified null-terminated argument array, with the stdin/out/err fd's
/// redirected, with a timeout specified by the last argument.  This terminates
/// the calling program if there is an error executing the specified program.
/// It returns the return value of the program, or -1 if a timeout is detected.
///
int llvm::RunProgramWithTimeout(const std::string &ProgramPath,
                                const char **Args,
                                const std::string &StdInFile,
                                const std::string &StdOutFile,
                                const std::string &StdErrFile,
                                unsigned NumSeconds) {
#ifdef HAVE_SYS_WAIT_H
  int Child = fork();
  switch (Child) {
  case -1:
    std::cerr << "ERROR forking!\n";
    exit(1);
  case 0:               // Child
    RedirectFD(StdInFile, 0);      // Redirect file descriptors...
    RedirectFD(StdOutFile, 1);
    if (StdOutFile != StdErrFile)
      RedirectFD(StdErrFile, 2);
    else
      dup2(1, 2);

    execv(ProgramPath.c_str(), (char *const *)Args);
    std::cerr << "Error executing program: '" << ProgramPath;
    for (; *Args; ++Args)
      std::cerr << " " << *Args;
    std::cerr << "'\n";
    exit(1);

  default: break;
  }

  // Make sure all output has been written while waiting
  std::cout << std::flush;

  // Install a timeout handler.
  Timeout = false;
  struct sigaction Act, Old;
  Act.sa_sigaction = 0;
  Act.sa_handler = TimeOutHandler;
  sigemptyset(&Act.sa_mask);
  Act.sa_flags = 0;
  sigaction(SIGALRM, &Act, &Old);

  // Set the timeout if one is set.
  if (NumSeconds)
    alarm(NumSeconds);

  int Status;
  while (wait(&Status) != Child)
    if (errno == EINTR) {
      if (Timeout) {
        // Kill the child.
        kill(Child, SIGKILL);
        
        if (wait(&Status) != Child)
          std::cerr << "Something funny happened waiting for the child!\n";
        
        alarm(0);
        sigaction(SIGALRM, &Old, 0);
        return -1;   // Timeout detected
      } else {
        std::cerr << "Error waiting for child process!\n";
        exit(1);
      }
    }

  alarm(0);
  sigaction(SIGALRM, &Old, 0);
  return Status;

#else
  std::cerr << "RunProgramWithTimeout not implemented on this platform!\n";
  return -1;
#endif
}


// ExecWait - executes a program with the specified arguments and environment.
// It then waits for the progarm to termiante and then returns to the caller.
//
// Inputs:
//  argv - The arguments to the program as an array of C strings.  The first
//         argument should be the name of the program to execute, and the
//         last argument should be a pointer to NULL.
//
//  envp - The environment passes to the program as an array of C strings in
//         the form of "name=value" pairs.  The last element should be a
//         pointer to NULL.
//
// Outputs:
//  None.
//
// Return value:
//  0 - No errors.
//  1 - The program could not be executed.
//  1 - The program returned a non-zero exit status.
//  1 - The program terminated abnormally.
//
// Notes:
//  The program will inherit the stdin, stdout, and stderr file descriptors
//  as well as other various configuration settings (umask).
//
//  This function should not print anything to stdout/stderr on its own.  It is
//  a generic library function.  The caller or executed program should report
//  errors in the way it sees fit.
//
//  This function does not use $PATH to find programs.
//
int llvm::ExecWait(const char * const old_argv[],
                   const char * const old_envp[]) {
#ifdef HAVE_SYS_WAIT_H
  // Create local versions of the parameters that can be passed into execve()
  // without creating const problems.
  char ** const argv = (char ** const) old_argv;
  char ** const envp = (char ** const) old_envp;

  // Create a child process.
  switch (fork()) {
    // An error occured:  Return to the caller.
    case -1:
      return 1;
      break;

    // Child process: Execute the program.
    case 0:
      execve (argv[0], argv, envp);
      // If the execve() failed, we should exit and let the parent pick up
      // our non-zero exit status.
      exit (1);

    // Parent process: Break out of the switch to do our processing.
    default:
      break;
  }

  // Parent process: Wait for the child process to terminate.
  int status;
  if ((wait (&status)) == -1)
    return 1;

  // If the program exited normally with a zero exit status, return success!
  if (WIFEXITED (status) && (WEXITSTATUS(status) == 0))
    return 0;
#else
  std::cerr << "llvm::ExecWait not implemented on this platform!\n";
#endif

  // Otherwise, return failure.
  return 1;
}
