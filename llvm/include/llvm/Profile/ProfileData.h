//=-- ProfileData.h - Instrumented profiling format support -------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for instrumentation based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILE_PROFILEDATA_H__
#define LLVM_PROFILE_PROFILEDATA_H__

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/system_error.h"

#include <vector>

namespace llvm {

const char PROFILEDATA_MAGIC[4] = {'L', 'P', 'R', 'F'};
const uint32_t PROFILEDATA_VERSION = 1;
const uint32_t PROFILEDATA_HEADER_SIZE = 24;

const error_category &profiledata_category();

struct profiledata_error {
  enum ErrorType {
    success = 0,
    bad_magic,
    unsupported_version,
    too_large,
    truncated,
    malformed,
    unknown_function
  };
  ErrorType V;

  profiledata_error(ErrorType V) : V(V) {}
  operator ErrorType() const { return V; }
};

inline error_code make_error_code(profiledata_error E) {
  return error_code(static_cast<int>(E), profiledata_category());
}

template <> struct is_error_code_enum<profiledata_error> : std::true_type {};
template <> struct is_error_code_enum<profiledata_error::ErrorType>
  : std::true_type {};

} // end namespace llvm

#endif // LLVM_PROFILE_PROFILEDATA_H__
