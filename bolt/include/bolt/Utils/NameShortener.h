//===--- NameShortener.h - Helper class for shortening names --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper class for shortening names.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_NAME_SHORTENER_H
#define LLVM_TOOLS_LLVM_BOLT_NAME_SHORTENER_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"

namespace llvm {
namespace bolt {

class NameShortener {
  StringMap<uint64_t> IDs;

public:
  uint64_t getID(StringRef Name) {
    return IDs.insert({Name, IDs.size()}).first->getValue();
  }
};

} // namespace bolt
} // namespace llvm

#endif
