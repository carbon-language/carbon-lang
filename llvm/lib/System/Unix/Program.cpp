//===- llvm/System/Unix/Program.cpp -----------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include <sys/stat.h>
#include <fcntl.h>
#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif

extern char** environ;

namespace llvm {
using namespace sys;

// This function just uses the PATH environment variable to find the program.
Path
Program::FindProgramByName(const std::string& progName) {

  // Check some degenerate cases
  if (progName.length() == 0) // no program
    return Path();
  Path temp;
  if (!temp.setFile(progName)) // invalid name
    return Path();
  if (temp.executable()) // already executable as is
    return temp;

  // At this point, the file name is valid and its not executable
 
  // Get the path. If its empty, we can't do anything to find it.
  const char *PathStr = getenv("PATH");
  if (PathStr == 0) 
    return Path();

  // Now we have a colon separated list of directories to search; try them.
  unsigned PathLen = strlen(PathStr);
  while (PathLen) {
    // Find the first colon...
    const char *Colon = std::find(PathStr, PathStr+PathLen, ':');

    // Check to see if this first directory contains the executable...
    Path FilePath;
    if (FilePath.setDirectory(std::string(PathStr,Colon))) {
      FilePath.appendFile(progName);
      if (FilePath.executable())
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

//
int 
Program::ExecuteAndWait(const Path& path, 
                        const std::vector<std::string>& args) {
  if (!path.executable())
    throw path.get() + " is not executable"; 

#ifdef HAVE_SYS_WAIT_H
  // Create local versions of the parameters that can be passed into execve()
  // without creating const problems.
  const char* argv[ args.size() + 2 ];
  unsigned index = 0;
  std::string progname(path.getLast());
  argv[index++] = progname.c_str();
  for (unsigned i = 0; i < args.size(); i++)
    argv[index++] = args[i].c_str();
  argv[index] = 0;

  // Create a child process.
  switch (fork()) {
    // An error occured:  Return to the caller.
    case -1:
      ThrowErrno(std::string("Couldn't execute program '") + path.get() + "'");
      break;

    // Child process: Execute the program.
    case 0:
      execve (path.c_str(), (char** const)argv, environ);
      // If the execve() failed, we should exit and let the parent pick up
      // our non-zero exit status.
      exit (errno);

    // Parent process: Break out of the switch to do our processing.
    default:
      break;
  }

  // Parent process: Wait for the child process to terminate.
  int status;
  if ((::wait (&status)) == -1)
    ThrowErrno(std::string("Failed waiting for program '") + path.get() + "'");

  // If the program exited normally with a zero exit status, return success!
  if (WIFEXITED (status))
    return WEXITSTATUS(status);
  else if (WIFSIGNALED(status))
    throw std::string("Program '") + path.get() + "' received terminating signal.";
  else
    return 0;
    
#else
  throw std::string("Program::ExecuteAndWait not implemented on this platform!\n");
#endif

}

}
// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
