//===- Error.cpp - system_error extensions for Object -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines a new error_category for the Object library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Error.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace object;

namespace {
class _object_error_category : public error_category {
public:
  const char* name() const override;
  std::string message(int ev) const override;
  error_condition default_error_condition(int ev) const override;
};
}

const char *_object_error_category::name() const {
  return "llvm.object";
}

std::string _object_error_category::message(int ev) const {
  object_error::Impl E = static_cast<object_error::Impl>(ev);
  switch (E) {
  case object_error::success: return "Success";
  case object_error::arch_not_found:
    return "No object file for requested architecture";
  case object_error::invalid_file_type:
    return "The file was not recognized as a valid object file";
  case object_error::parse_failed:
    return "Invalid data was encountered while parsing the file";
  case object_error::unexpected_eof:
    return "The end of the file was unexpectedly encountered";
  }
  llvm_unreachable("An enumerator of object_error does not have a message "
                   "defined.");
}

error_condition _object_error_category::default_error_condition(int ev) const {
  if (ev == object_error::success)
    return errc::success;
  return errc::invalid_argument;
}

const error_category &object::object_category() {
  static _object_error_category o;
  return o;
}
