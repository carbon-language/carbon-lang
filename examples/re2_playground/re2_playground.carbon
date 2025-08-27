// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Cpp library "re2/re2.h";

import Cpp inline '''c++
// A wrapper around RE2 working around lack of:
//
// * Overloading support for the constructor.
// * Non-null pointer support for the `submatch` parameter.
struct WrapRE2 : re2::RE2 {
  WrapRE2(std::string_view v) : RE2(v) {}
  bool Match(absl::string_view text, size_t startpos, size_t endpos,
             Anchor re_anchor, absl::string_view* _Nonnull submatch,
             int nsubmatch) const {
    return RE2::Match(text, startpos, endpos, re_anchor, submatch, nsubmatch);
  }
};
''';

class RE2 {
  extend adapt Cpp.WrapRE2;
  fn Make(re: str) -> Self {
    return WrapRE2(re) as Self;
  }
  fn Match[self: Self](text: str) -> bool {
    var submatches: str;
    return (self as Cpp.WrapRE2).Match(text, 0, text.Size(), UNANCHORED, &submatches, 0);
  }
}

fn Run() -> i32 {
  var re: RE2 = RE2.Make("^a*$");
  if (re.Match("aaaaaaaab")) {
    return 1;
  }
  if (not re.Match("aaaaaaaaa")) {
    return 1;
  }
  return 0;
}
