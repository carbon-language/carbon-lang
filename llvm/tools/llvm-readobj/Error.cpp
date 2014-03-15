//===- Error.cpp - system_error extensions for llvm-readobj -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines a new error_category for the llvm-readobj tool.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class _readobj_error_category : public error_category {
public:
  const char* name() const override;
  std::string message(int ev) const override;
  error_condition default_error_condition(int ev) const override;
};
} // namespace

const char *_readobj_error_category::name() const {
  return "llvm.readobj";
}

std::string _readobj_error_category::message(int ev) const {
  switch (ev) {
  case readobj_error::success: return "Success";
  case readobj_error::file_not_found:
    return "No such file.";
  case readobj_error::unsupported_file_format:
    return "The file was not recognized as a valid object file.";
  case readobj_error::unrecognized_file_format:
    return "Unrecognized file type.";
  case readobj_error::unsupported_obj_file_format:
    return "Unsupported object file format.";
  case readobj_error::unknown_symbol:
    return "Unknown symbol.";
  default:
    llvm_unreachable("An enumerator of readobj_error does not have a message "
                     "defined.");
  }
}

error_condition _readobj_error_category::default_error_condition(int ev) const {
  if (ev == readobj_error::success)
    return errc::success;
  return errc::invalid_argument;
}

namespace llvm {
const error_category &readobj_category() {
  static _readobj_error_category o;
  return o;
}
} // namespace llvm
