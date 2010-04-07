//===- lib/Support/ErrorHandling.cpp - Callbacks for errors ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an API used to indicate fatal error conditions.  Non-fatal
// errors (most of them) should be handled through LLVMContext.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Threading.h"
#include <cassert>
#include <cstdlib>
using namespace llvm;
using namespace std;

static fatal_error_handler_t ErrorHandler = 0;
static void *ErrorHandlerUserData = 0;

void llvm::install_fatal_error_handler(fatal_error_handler_t handler,
                                       void *user_data) {
  assert(!llvm_is_multithreaded() &&
         "Cannot register error handlers after starting multithreaded mode!\n");
  assert(!ErrorHandler && "Error handler already registered!\n");
  ErrorHandler = handler;
  ErrorHandlerUserData = user_data;
}

void llvm::remove_fatal_error_handler() {
  ErrorHandler = 0;
}

void llvm::report_fatal_error(const char *reason) {
  report_fatal_error(Twine(reason));
}

void llvm::report_fatal_error(const std::string &reason) {
  report_fatal_error(Twine(reason));
}

void llvm::report_fatal_error(const Twine &reason) {
  if (!ErrorHandler) {
    errs() << "LLVM ERROR: " << reason << "\n";
  } else {
    ErrorHandler(ErrorHandlerUserData, reason.str());
  }
  exit(1);
}

void llvm::llvm_unreachable_internal(const char *msg, const char *file,
                                     unsigned line) {
  // This code intentionally doesn't call the ErrorHandler callback, because
  // llvm_unreachable is intended to be used to indicate "impossible"
  // situations, and not legitimate runtime errors.
  if (msg)
    dbgs() << msg << "\n";
  dbgs() << "UNREACHABLE executed";
  if (file)
    dbgs() << " at " << file << ":" << line;
  dbgs() << "!\n";
  abort();
}
