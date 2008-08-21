//===--- raw_ostream.cpp - Implement the raw_ostream classes --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements support for bulk buffered stream output.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include <ostream>
using namespace llvm;

#include <fcntl.h>

#if defined(_MSC_VER)
#include <io.h>
#ifndef STDIN_FILENO
# define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
# define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
# define STDERR_FILENO 2
#endif
#endif

// An out of line virtual method to provide a home for the class vtable.
void raw_ostream::handle() {}

//===----------------------------------------------------------------------===//
//  raw_fd_ostream
//===----------------------------------------------------------------------===//

/// raw_fd_ostream - Open the specified file for writing.  If an error occurs,
/// information about the error is put into ErrorInfo, and the stream should
/// be immediately destroyed.
raw_fd_ostream::raw_fd_ostream(const char *Filename, std::string &ErrorInfo) {
  // Handle "-" as stdout.
  if (Filename[0] == '-' && Filename[1] == 0) {
    FD = STDOUT_FILENO;
    ShouldClose = false;
    return;
  }
  
  FD = open(Filename, O_WRONLY|O_CREAT|O_TRUNC, 0644);
  if (FD < 0) {
    ErrorInfo = "Error opening output file '" + std::string(Filename) + "'";
    ShouldClose = false;
  } else {
    ShouldClose = true;
  }
}

raw_fd_ostream::~raw_fd_ostream() {
  flush();
  if (ShouldClose)
    close(FD);
}

void raw_fd_ostream::flush_impl() {
  if (OutBufCur-OutBufStart)
    ::write(FD, OutBufStart, OutBufCur-OutBufStart);
  HandleFlush();
}

//===----------------------------------------------------------------------===//
//  raw_stdout/err_ostream
//===----------------------------------------------------------------------===//

raw_stdout_ostream::raw_stdout_ostream():raw_fd_ostream(STDOUT_FILENO, false) {}
raw_stderr_ostream::raw_stderr_ostream():raw_fd_ostream(STDERR_FILENO, false) {}

// An out of line virtual method to provide a home for the class vtable.
void raw_stdout_ostream::handle() {}
void raw_stderr_ostream::handle() {}

/// outs() - This returns a reference to a raw_ostream for standard output.
/// Use it like: outs() << "foo" << "bar";
raw_ostream &llvm::outs() {
  static raw_stdout_ostream S;
  return S;
}

/// errs() - This returns a reference to a raw_ostream for standard error.
/// Use it like: errs() << "foo" << "bar";
raw_ostream &llvm::errs() {
  static raw_stderr_ostream S;
  return S;
}

//===----------------------------------------------------------------------===//
//  raw_os_ostream
//===----------------------------------------------------------------------===//

/// flush_impl - The is the piece of the class that is implemented by
/// subclasses.  This outputs the currently buffered data and resets the
/// buffer to empty.
void raw_os_ostream::flush_impl() {
  if (OutBufCur-OutBufStart)
    OS.write(OutBufStart, OutBufCur-OutBufStart);
  HandleFlush();
}
