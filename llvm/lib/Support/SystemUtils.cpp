//===- SystemUtils.h - Utilities to do low-level system stuff --*- C++ -*--===//
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

/// isExecutableFile - This function returns true if the filename specified
/// exists and is executable.
///
bool isExecutableFile(const std::string &ExeFileName) {
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


// FindExecutable - Find a named executable, giving the argv[0] of bugpoint.
// This assumes the executable is in the same directory as bugpoint itself.
// If the executable cannot be found, return an empty string.
//
std::string FindExecutable(const std::string &ExeName,
			   const std::string &BugPointPath) {
  // First check the directory that bugpoint is in.  We can do this if
  // BugPointPath contains at least one / character, indicating that it is a
  // relative path to bugpoint itself.
  //
  std::string Result = BugPointPath;
  while (!Result.empty() && Result[Result.size()-1] != '/')
    Result.erase(Result.size()-1, 1);

  if (!Result.empty()) {
    Result += ExeName;
    if (isExecutableFile(Result)) return Result; // Found it?
  }

  // Okay, if the path to bugpoint didn't tell us anything, try using the PATH
  // environment variable.
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
/// redirected, with a timeout specified on the commandline.  This terminates
/// the calling program if there is an error executing the specified program.
/// It returns the return value of the program, or -1 if a timeout is detected.
///
int RunProgramWithTimeout(const std::string &ProgramPath, const char **Args,
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
    std::cerr << "Error executing program '" << ProgramPath;
    for (; *Args; ++Args)
      std::cerr << " " << *Args;
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
