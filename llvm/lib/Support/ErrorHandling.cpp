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

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Threading.h"
#include <cassert>
#include <cstdlib>

using namespace llvm;
using namespace std;

static llvm_error_handler_t ErrorHandler = 0;
static void *ErrorHandlerUserData = 0;

namespace llvm {
void llvm_install_error_handler(llvm_error_handler_t handler,
                                void *user_data) {
  assert(!llvm_is_multithreaded() &&
         "Cannot register error handlers after starting multithreaded mode!\n");
  assert(!ErrorHandler && "Error handler already registered!\n");
  ErrorHandler = handler;
  ErrorHandlerUserData = user_data;
}

void llvm_remove_error_handler() {
  ErrorHandler = 0;
}

void llvm_report_error(const char *reason) {
  llvm_report_error(Twine(reason));
}

void llvm_report_error(const std::string &reason) {
  llvm_report_error(Twine(reason));
}

void llvm_report_error(const Twine &reason) {
  if (!ErrorHandler) {
    errs() << "LLVM ERROR: " << reason << "\n";
  } else {
    ErrorHandler(ErrorHandlerUserData, reason.str());
  }
  exit(1);
}

void llvm_unreachable_internal(const char *msg, const char *file, 
                               unsigned line) {
  if (msg)
    errs() << msg << "\n";
  errs() << "UNREACHABLE executed";
  if (file)
    errs() << " at " << file << ":" << line;
  errs() << "!\n";
  abort();
}
}

