//===- Error.cpp - system_error extensions for llvm-vtabledump --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines a new error_category for the llvm-vtabledump tool.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class vtabledump_error_category : public std::error_category {
public:
  const char *name() const LLVM_NOEXCEPT override { return "llvm.vtabledump"; }
  std::string message(int ev) const override {
    switch (static_cast<vtabledump_error>(ev)) {
    case vtabledump_error::success:
      return "Success";
    case vtabledump_error::file_not_found:
      return "No such file.";
    case vtabledump_error::unrecognized_file_format:
      return "Unrecognized file type.";
    }
    llvm_unreachable(
        "An enumerator of vtabledump_error does not have a message defined.");
  }
};
} // namespace

namespace llvm {
const std::error_category &vtabledump_category() {
  static vtabledump_error_category o;
  return o;
}
} // namespace llvm
