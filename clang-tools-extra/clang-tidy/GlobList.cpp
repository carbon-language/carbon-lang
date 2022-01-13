//===--- tools/extra/clang-tidy/GlobList.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobList.h"
#include "llvm/ADT/SmallString.h"

namespace clang {
namespace tidy {

// Returns true if GlobList starts with the negative indicator ('-'), removes it
// from the GlobList.
static bool consumeNegativeIndicator(StringRef &GlobList) {
  GlobList = GlobList.trim();
  if (GlobList.startswith("-")) {
    GlobList = GlobList.substr(1);
    return true;
  }
  return false;
}

// Converts first glob from the comma-separated list of globs to Regex and
// removes it and the trailing comma from the GlobList.
static llvm::Regex consumeGlob(StringRef &GlobList) {
  StringRef UntrimmedGlob = GlobList.substr(0, GlobList.find(','));
  StringRef Glob = UntrimmedGlob.trim();
  GlobList = GlobList.substr(UntrimmedGlob.size() + 1);
  SmallString<128> RegexText("^");
  StringRef MetaChars("()^$|*+?.[]\\{}");
  for (char C : Glob) {
    if (C == '*')
      RegexText.push_back('.');
    else if (MetaChars.contains(C))
      RegexText.push_back('\\');
    RegexText.push_back(C);
  }
  RegexText.push_back('$');
  return llvm::Regex(RegexText);
}

GlobList::GlobList(StringRef Globs, bool KeepNegativeGlobs /* =true */) {
  Items.reserve(Globs.count(',') + 1);
  do {
    GlobListItem Item;
    Item.IsPositive = !consumeNegativeIndicator(Globs);
    Item.Regex = consumeGlob(Globs);
    if (Item.IsPositive || KeepNegativeGlobs)
      Items.push_back(std::move(Item));
  } while (!Globs.empty());
}

bool GlobList::contains(StringRef S) const {
  // Iterating the container backwards as the last match determins if S is in
  // the list.
  for (const GlobListItem &Item : llvm::reverse(Items)) {
    if (Item.Regex.match(S))
      return Item.IsPositive;
  }
  return false;
}

bool CachedGlobList::contains(StringRef S) const {
  switch (auto &Result = Cache[S]) {
  case Yes:
    return true;
  case No:
    return false;
  case None:
    Result = GlobList::contains(S) ? Yes : No;
    return Result == Yes;
  }
  llvm_unreachable("invalid enum");
}

} // namespace tidy
} // namespace clang
