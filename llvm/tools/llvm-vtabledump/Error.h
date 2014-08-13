//===- Error.h - system_error extensions for llvm-vtabledump ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This declares a new error_category for the llvm-vtabledump tool.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_VTABLEDUMP_ERROR_H
#define LLVM_TOOLS_LLVM_VTABLEDUMP_ERROR_H

#include <system_error>

namespace llvm {
const std::error_category &vtabledump_category();

enum class vtabledump_error {
  success = 0,
  file_not_found,
  unrecognized_file_format,
};

inline std::error_code make_error_code(vtabledump_error e) {
  return std::error_code(static_cast<int>(e), vtabledump_category());
}

} // namespace llvm

namespace std {
template <>
struct is_error_code_enum<llvm::vtabledump_error> : std::true_type {};
}

#endif
