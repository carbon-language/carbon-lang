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

  /// Character guaranteed not to be used by any "native" name passed to
  /// uniquify() function.
  static constexpr char Sep = '/';

public:
  /// Return unique version of the \p Name in the form "Name<Sep><Number>".
  std::string uniquify(StringRef Name) {
    const auto ID = ++Counters[Name];
    return (Name + Twine(Sep) + Twine(ID)).str();
  }

  /// For uniquified \p Name, return the original form (that may no longer be
  /// unique).
  static StringRef restore(StringRef Name) {
    return Name.substr(0, Name.find_first_of(Sep));
  }

  /// Append \p Suffix to the original string in \p UniqueName  preserving the
  /// deduplication form. E.g. append("Name<Sep>42", "Suffix") will return
  /// "NameSuffix<Sep>42".
  static std::string append(StringRef UniqueName, StringRef Suffix) {
    StringRef LHS, RHS;
    std::tie(LHS, RHS) = UniqueName.split(Sep);
    return (LHS + Suffix + Twine(Sep) + RHS).str();
  }
};

} // namespace bolt
} // namespace llvm

#endif
