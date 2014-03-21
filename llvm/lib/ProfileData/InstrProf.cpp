//=-- InstrProf.cpp - Instrumented profiling format support -----------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for clang's instrumentation based PGO and
// coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class InstrProfErrorCategoryType : public error_category {
  const char *name() const override { return "llvm.instrprof"; }
  std::string message(int IE) const override {
    instrprof_error::ErrorType E = static_cast<instrprof_error::ErrorType>(IE);
    switch (E) {
    case instrprof_error::success:
      return "Success";
    case instrprof_error::eof:
      return "End of File";
    case instrprof_error::bad_magic:
      return "Invalid file format (bad magic)";
    case instrprof_error::unsupported_version:
      return "Unsupported format version";
    case instrprof_error::too_large:
      return "Too much profile data";
    case instrprof_error::truncated:
      return "Truncated profile data";
    case instrprof_error::malformed:
      return "Malformed profile data";
    case instrprof_error::unknown_function:
      return "No profile data available for function";
    }
    llvm_unreachable("A value of instrprof_error has no message.");
  }
  error_condition default_error_condition(int EV) const {
    if (EV == instrprof_error::success)
      return errc::success;
    return errc::invalid_argument;
  }
};
}

const error_category &llvm::instrprof_category() {
  static InstrProfErrorCategoryType C;
  return C;
}
