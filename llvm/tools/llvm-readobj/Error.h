//===- Error.h - system_error extensions for llvm-readobj -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This declares a new error_category for the llvm-readobj tool.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_READOBJ_ERROR_H
#define LLVM_READOBJ_ERROR_H

#include "llvm/Support/system_error.h"

namespace llvm {

const error_category &readobj_category();

enum class readobj_error {
  success = 0,
  file_not_found,
  unsupported_file_format,
  unrecognized_file_format,
  unsupported_obj_file_format,
  unknown_symbol
};

inline error_code make_error_code(readobj_error e) {
  return error_code(static_cast<int>(e), readobj_category());
}

template <> struct is_error_code_enum<readobj_error> : std::true_type { };

} // namespace llvm

#endif
