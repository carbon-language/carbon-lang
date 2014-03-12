//=-- ProfileData.cpp - Instrumented profiling format support ---------------=//
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

#include "llvm/Profile/ProfileData.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class ProfileDataErrorCategoryType : public _do_message {
  const char *name() const override { return "llvm.profiledata"; }
  std::string message(int IE) const {
    profiledata_error::ErrorType E =
        static_cast<profiledata_error::ErrorType>(IE);
    switch (E) {
    case profiledata_error::success: return "Success";
    case profiledata_error::bad_magic:
      return "Invalid file format (bad magic)";
    case profiledata_error::unsupported_version:
      return "Unsupported format version";
    case profiledata_error::too_large:
      return "Too much profile data";
    case profiledata_error::truncated:
      return "Truncated profile data";
    case profiledata_error::malformed:
      return "Malformed profile data";
    case profiledata_error::unknown_function:
      return "No profile data available for function";
    }
    llvm_unreachable("A value of profiledata_error has no message.");
  }
  error_condition default_error_condition(int EV) const {
    if (EV == profiledata_error::success)
      return errc::success;
    return errc::invalid_argument;
  }
};
}

const error_category &llvm::profiledata_category() {
  static ProfileDataErrorCategoryType C;
  return C;
}
