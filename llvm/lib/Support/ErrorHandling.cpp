//===- lib/Support/ErrorHandling.cpp - Callbacks for errors -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an API for error handling, it supersedes cerr+abort(), and 
// cerr+exit() style error handling.
// Callbacks can be registered for these errors through this API.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Threading.h"
#include <cassert>
#include <cstdlib>

using namespace llvm;
using namespace std;

static llvm_error_handler_t ErrorHandler = 0;
namespace llvm {
void llvm_install_error_handler(llvm_error_handler_t handler) {
  assert(!llvm_is_multithreaded() &&
         "Cannot register error handlers after starting multithreaded mode!\n");
  assert(!ErrorHandler && "Error handler already registered!\n");
  ErrorHandler = handler;
}

void llvm_remove_error_handler(void) {
  ErrorHandler = 0;
}

void llvm_report_error(const std::string &reason) {
  if (!ErrorHandler) {
    errs() << "LLVM ERROR: " << reason << "\n";
  } else {
    ErrorHandler(reason);
  }
  exit(1);
}

void llvm_unreachable(const char *msg) {
  if (msg)
    errs() << msg << "\n";
  abort();
}
}

