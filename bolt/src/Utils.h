//===--- Utils.h - Common helper functions --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common helper functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_UTILS_H
#define LLVM_TOOLS_LLVM_BOLT_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace bolt {

/// Free memory allocated for \p List.
template<typename T> void clearList(T& List) {
  T TempList;
  TempList.swap(List);
}

void report_error(StringRef Message, std::error_code EC);

void report_error(StringRef Message, Error E);

void check_error(std::error_code EC, StringRef Message);

void check_error(Error E, Twine Message);

} // namespace bolt
} // namespace llvm

#endif
