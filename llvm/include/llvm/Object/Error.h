//===- Error.h - system_error extensions for Object -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This declares a new error_category for the Object library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ERROR_H
#define LLVM_OBJECT_ERROR_H

#include "llvm/Support/system_error.h"

namespace llvm {
namespace object {

const error_category &object_category();

enum class object_error {
  success = 0,
  arch_not_found,
  invalid_file_type,
  parse_failed,
  unexpected_eof
};

inline error_code make_error_code(object_error e) {
  return error_code(static_cast<int>(e), object_category());
}

} // end namespace object.

template <> struct is_error_code_enum<object::object_error> : std::true_type {};

} // end namespace llvm.

#endif
