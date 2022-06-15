//===- CommonConfig.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjCopy/CommonConfig.h"

namespace llvm {
namespace objcopy {

Expected<NameOrPattern>
NameOrPattern::create(StringRef Pattern, MatchStyle MS,
                      function_ref<Error(Error)> ErrorCallback) {
  switch (MS) {
  case MatchStyle::Literal:
    return NameOrPattern(Pattern);
  case MatchStyle::Wildcard: {
    SmallVector<char, 32> Data;
    bool IsPositiveMatch = true;
    if (Pattern[0] == '!') {
      IsPositiveMatch = false;
      Pattern = Pattern.drop_front();
    }
    Expected<GlobPattern> GlobOrErr = GlobPattern::create(Pattern);

    // If we couldn't create it as a glob, report the error, but try again
    // with a literal if the error reporting is non-fatal.
    if (!GlobOrErr) {
      if (Error E = ErrorCallback(GlobOrErr.takeError()))
        return std::move(E);
      return create(Pattern, MatchStyle::Literal, ErrorCallback);
    }

    return NameOrPattern(std::make_shared<GlobPattern>(*GlobOrErr),
                         IsPositiveMatch);
  }
  case MatchStyle::Regex: {
    SmallVector<char, 32> Data;
    return NameOrPattern(std::make_shared<Regex>(
        ("^" + Pattern.ltrim('^').rtrim('$') + "$").toStringRef(Data)));
  }
  }
  llvm_unreachable("Unhandled llvm.objcopy.MatchStyle enum");
}

} // end namespace objcopy
} // end namespace llvm
