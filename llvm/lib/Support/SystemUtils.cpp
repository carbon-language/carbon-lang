//===- SystemUtils.h - Utilities to do low-level system stuff --*- C++ -*--===//
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

#include "Support/SystemUtils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include "Config/sys/types.h"
#include "Config/sys/stat.h"
#include "Config/fcntl.h"
#include "Config/sys/wait.h"
#include "Config/unistd.h"
#include "Config/errno.h"
using namespace llvm;

/// isExecutableFile - This function returns true if the filename specified
/// exists and is executable.
///
bool llvm::isExecutableFile(const std::string &ExeFileName) {
  struct stat Buf;
  if (stat(ExeFileName.c_str(), &Buf))
    return false;  // Must not be executable!

  if (!(Buf.st_mode & S_IFREG))
    return false;                    // Not a regular file?

  if (Buf.st_uid == getuid())        // Owner of file?
    return Buf.st_mode & S_IXUSR;
  else if (Buf.st_gid == getgid())   // In group of file?
    return Buf.st_mode & S_IXGRP;
  else                               // Unrelated to file?
    return Buf.st_mode & S_IXOTH;
}

/// FindExecutable - Find a named executable, giving the argv[0] of program
/// being executed. This allows us to find another LLVM tool if it is built
/// into the same directory, but that directory is neither the current
/// directory, nor in the PATH.  If the executable cannot be found, return an
/// empty string.
/// 
std::string llvm::FindExecutable(const std::string &ExeName,
                                 const std::string &ProgramPath) {
  // First check the directory that bugpoint is in.  We can do this if
  // BugPointPath contains at least one / character, indicating that it is a
  // relative path to bugpoint itself.
  //
  std::string Result = ProgramPath;
  while (!Result.empty() && Result[Result.size()-1] != '/')
    Result.erase(Result.size()-1, 1);

  if (!Result.empty()) {
    Result += ExeName;
    if (isExecutableFile(Result)) return Result; // Found it?
  }

  // Okay, if the path to the program didn't tell us anything, try using the
  // PATH environment variable.
  const char *PathStr = getenv("PATH");
  if (PathStr == 0) return "";

  // Now we have a colon separated list of directories to search... try them...
  unsigned PathLen = strlen(PathStr);
  while (PathLen) {
    // Find the first colon...
    const char *Colon = std::find(PathStr, PathStr+PathLen, ':');
    
    // Check to see if this first directory contains the executable...
    std::string FilePath = std::string(PathStr, Colon) + '/' + ExeName;
    if (isExecutableFile(FilePath))
      return FilePath;                    // Found the executable!
   
    // Nope it wasn't in this directory, check the next range!
    PathLen -= Colon-PathStr;
    PathStr = Colon;
    while (*PathStr == ':') {   // Advance past colons
      PathStr++;
      PathLen--;
    }
  }

  // If we fell out, we ran out of directories in PATH to search, return failure
  return "";
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

/// RunProgramWithTimeout - This function executes the specified program, with
/// the specified null-terminated argument array, with the stdin/out/err fd's
/// redirected, with a timeout specified on the command line.  This terminates
/// the calling program if there is an error executing the specified program.
/// It returns the return value of the program, or -1 if a timeout is detected.
///
int llvm::RunProgramWithTimeout(const std::string &ProgramPath,
                                const char **Args,
                                const std::string &StdInFile,
                                const std::string &StdOutFile,
                                const std::string &StdErrFile) {
  // FIXME: install sigalarm handler here for timeout...

  int Child = fork();
  switch (Child) {
  case -1:
    std::cerr << "ERROR forking!\n";
    exit(1);
  case 0:               // Child
    RedirectFD(StdInFile, 0);      // Redirect file descriptors...
    RedirectFD(StdOutFile, 1);
    RedirectFD(StdErrFile, 2);

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

  int Status;
  if (wait(&Status) != Child) {
    if (errno == EINTR) {
      static bool FirstTimeout = true;
      if (FirstTimeout) {
        std::cout <<
 "*** Program execution timed out!  This mechanism is designed to handle\n"
 "    programs stuck in infinite loops gracefully.  The -timeout option\n"
 "    can be used to change the timeout threshold or disable it completely\n"
 "    (with -timeout=0).  This message is only displayed once.\n";
        FirstTimeout = false;
      }
      return -1;   // Timeout detected
    }

    std::cerr << "Error waiting for child process!\n";
    exit(1);
  }
  return Status;
}


//
// Function: ExecWait ()
//
// Description:
//  This function executes a program with the specified arguments and
//  environment.  It then waits for the progarm to termiante and then returns
//  to the caller.
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
  // Child process ID
  register int child;

  // Status from child process when it exits
  int status;
 
  //
  // Create local versions of the parameters that can be passed into execve()
  // without creating const problems.
  //
  char ** const argv = (char ** const) old_argv;
  char ** const envp = (char ** const) old_envp;

  //
  // Create a child process.
  //
  switch (child=fork())
  {
    //
    // An error occured:  Return to the caller.
    //
    case -1:
      return 1;
      break;

    //
    // Child process: Execute the program.
    //
    case 0:
      execve (argv[0], argv, envp);

      //
      // If the execve() failed, we should exit and let the parent pick up
      // our non-zero exit status.
      //
      exit (1);
      break;

    //
    // Parent process: Break out of the switch to do our processing.
    //
    default:
      break;
  }

  //
  // Parent process: Wait for the child process to termiante.
  //
  if ((wait (&status)) == -1)
  {
    return 1;
  }

  //
  // If the program exited normally with a zero exit status, return success!
  //
  if (WIFEXITED (status) && (WEXITSTATUS(status) == 0))
  {
    return 0;
  }

  //
  // Otherwise, return failure.
  //
  return 1;
}
