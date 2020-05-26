//===--- NameShortener.h - Helper class for shortening names --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
