//===- Win32/Program.cpp - Win32 Program Implementation ------- -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Jeff Cohen and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the Program class.
//
//===----------------------------------------------------------------------===//

#include "Win32.h"
#include <cstdio>
#include <malloc.h>
#include <io.h>
#include <fcntl.h>

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code 
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//

namespace llvm {
using namespace sys;

// This function just uses the PATH environment variable to find the program.
Path
Program::FindProgramByName(const std::string& progName) {

  // Check some degenerate cases
  if (progName.length() == 0) // no program
    return Path();
  Path temp;
  if (!temp.set(progName)) // invalid name
    return Path();
  if (temp.canExecute()) // already executable as is
    return temp;

  // At this point, the file name is valid and its not executable.
  // Let Windows search for it.
  char buffer[MAX_PATH];
  char *dummy = NULL;
  DWORD len = SearchPath(NULL, progName.c_str(), ".exe", MAX_PATH,
                         buffer, &dummy);

  // See if it wasn't found.
  if (len == 0)
    return Path();

  // See if we got the entire path.
  if (len < MAX_PATH)
    return Path(buffer);

  // Buffer was too small; grow and retry.
  while (true) {
    char *b = reinterpret_cast<char *>(_alloca(len+1));
    DWORD len2 = SearchPath(NULL, progName.c_str(), ".exe", len+1, b, &dummy);

    // It is unlikely the search failed, but it's always possible some file
    // was added or removed since the last search, so be paranoid...
    if (len2 == 0)
      return Path();
    else if (len2 <= len)
      return Path(b);

    len = len2;
  }
}

static HANDLE RedirectIO(const Path *path, int fd, std::string* ErrMsg) {
  HANDLE h;
  if (path == 0) {
    DuplicateHandle(GetCurrentProcess(), (HANDLE)_get_osfhandle(fd),
                    GetCurrentProcess(), &h,
                    0, TRUE, DUPLICATE_SAME_ACCESS);
    return h;
  }

  const char *fname = path->toString().c_str();
  if (*fname == 0)
    fname = "NUL";

  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = 0;
  sa.bInheritHandle = TRUE;

  h = CreateFile(fname, fd ? GENERIC_WRITE : GENERIC_READ, FILE_SHARE_READ,
                 &sa, fd == 0 ? OPEN_EXISTING : CREATE_ALWAYS,
                 FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) {
    MakeErrMsg(ErrMsg, std::string(fname) + ": Can't open file for " +
        (fd ? "input: " : "output: "));
    return h;
  }
  return h;
}

int 
Program::ExecuteAndWait(const Path& path, 
                        const char** args,
                        const char** envp,
                        const Path** redirects,
                        unsigned secondsToWait,
                        std::string* ErrMsg) {
  if (!path.canExecute()) {
    if (ErrMsg)
      *ErrMsg = "program not executable";
    return -1;
  }

  // Windows wants a command line, not an array of args, to pass to the new
  // process.  We have to concatenate them all, while quoting the args that
  // have embedded spaces.

  // First, determine the length of the command line.
  unsigned len = 0;
  for (unsigned i = 0; args[i]; i++) {
    len += strlen(args[i]) + 1;
    if (strchr(args[i], ' '))
      len += 2;
  }

  // Now build the command line.
  char *command = reinterpret_cast<char *>(_alloca(len));
  char *p = command;

  for (unsigned i = 0; args[i]; i++) {
    const char *arg = args[i];
    size_t len = strlen(arg);
    bool needsQuoting = strchr(arg, ' ') != 0;
    if (needsQuoting)
      *p++ = '"';
    memcpy(p, arg, len);
    p += len;
    if (needsQuoting)
      *p++ = '"';
    *p++ = ' ';
  }

  *p = 0;

  // Create a child process.
  STARTUPINFO si;
  memset(&si, 0, sizeof(si));
  si.cb = sizeof(si);
  si.hStdInput = INVALID_HANDLE_VALUE;
  si.hStdOutput = INVALID_HANDLE_VALUE;
  si.hStdError = INVALID_HANDLE_VALUE;

  if (redirects) {
    si.dwFlags = STARTF_USESTDHANDLES;
    
    si.hStdInput = RedirectIO(redirects[0], 0, ErrMsg);
    if (si.hStdInput == INVALID_HANDLE_VALUE) {
      MakeErrMsg(ErrMsg, "can't redirect stdin");
      return -1;
    }
    si.hStdOutput = RedirectIO(redirects[1], 1);
    if (si.hStdOutput == INVALID_HANDLE_VALUE) {
      CloseHandle(si.hStdInput);
      MakeErrMsg(ErrMsg, "can't redirect stdout");
      return -1;
    }
    if (redirects[1] && redirects[2] && *(redirects[1]) != *(redirects[2])) {
      si.hStdError = RedirectIO(redirects[2], 2);
      if (si.hStdError == INVALID_HANDLE_VALUE) {
        CloseHandle(si.hStdInput);
        CloseHandle(si.hStdOutput);
        MakeErrMsg(ErrMsg, "can't redirect stderr");
        return -1;
      }
    } else {
      DuplicateHandle(GetCurrentProcess(), si.hStdOutput,
                      GetCurrentProcess(), &si.hStdError,
                      0, TRUE, DUPLICATE_SAME_ACCESS);
    }
  }

  PROCESS_INFORMATION pi;
  memset(&pi, 0, sizeof(pi));

  fflush(stdout);
  fflush(stderr);
  BOOL rc = CreateProcess(path.c_str(), command, NULL, NULL, FALSE, 0,
                          envp, NULL, &si, &pi);
  DWORD err = GetLastError();

  // Regardless of whether the process got created or not, we are done with
  // the handles we created for it to inherit.
  CloseHandle(si.hStdInput);
  CloseHandle(si.hStdOutput);
  CloseHandle(si.hStdError);

  // Now return an error if the process didn't get created.
  if (!rc)
  {
    SetLastError(err);
    MakeErrMsg(ErrMsg, std::string("Couldn't execute program '") + 
               path.toString() + "'");
    return -1;
  }

  // Wait for it to terminate.
  DWORD millisecondsToWait = INFINITE;
  if (secondsToWait > 0)
    millisecondsToWait = secondsToWait * 1000;

  if (WaitForSingleObject(pi.hProcess, millisecondsToWait) == WAIT_TIMEOUT) {
    if (!TerminateProcess(pi.hProcess, 1)) {
      MakeErrMsg(ErrMsg, std::string("Failed to terminate timed-out program '")
          + path.toString() + "'");
      return -1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
  }
  
  // Get its exit status.
  DWORD status;
  rc = GetExitCodeProcess(pi.hProcess, &status);
  err = GetLastError();

  // Done with the handles; go close them.
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  if (!rc) {
    SetLastError(err);
    MakeErrMsg(ErrMsg, std::string("Failed getting status for program '") + 
               path.toString() + "'");
    return -1;
  }

  return status;
}

bool Program::ChangeStdinToBinary(){
  int result = _setmode( _fileno(stdin), _O_BINARY );
  return result == -1;
}

bool Program::ChangeStdoutToBinary(){
  int result = _setmode( _fileno(stdout), _O_BINARY );
  return result == -1;
}

}
