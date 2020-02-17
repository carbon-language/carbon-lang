//===--- NameResolver.h - Helper class for names deduplication ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Helper class for names deduplication.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_NAME_RESOLVER_H
#define LLVM_TOOLS_LLVM_BOLT_NAME_RESOLVER_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"

namespace llvm {
namespace bolt {

class NameResolver {
  /// Track the number of duplicate names.
  StringMap<uint64_t> Counters;

public:
  /// Return unique version of a symbol name in the form "<name>/<number>".
  std::string uniquify(StringRef Name) {
    const auto ID = ++Counters[Name];
    return (Name + "/" + Twine(ID)).str();
  }
};

} // namespace bolt
} // namespace llvm

#endif
