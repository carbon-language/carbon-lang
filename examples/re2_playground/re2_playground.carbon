// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Cpp library "llvm/LineEditor/LineEditor.h";
import Cpp library "llvm/Support/raw_ostream.h";
import Cpp library "re2/re2.h";

import Cpp inline "using OptionalString = std::optional<std::string>;";
import Cpp inline "std::string_view AsStringView(const std::string &s) { return s; }";

import Core library "io";

class RE2 {
  extend adapt Cpp.re2.RE2;
  fn Make(re: str) -> Self {
    return RE2(re) as Self;
  }
  fn Match[self: Self](text: str) -> bool {
    return FullMatch(text, self as Cpp.re2.RE2);
  }
}

impl str as Core.ImplicitAs(Cpp.llvm.StringRef) {
  fn Convert[self: Self]() -> Cpp.llvm.StringRef {
    return Cpp.llvm.StringRef.StringRef(self);
  }
}

fn Print(var s: str) {
  *Cpp.llvm.outs() << s;
}

fn Run() -> i32 {
  var editor: Cpp.llvm.LineEditor = Cpp.llvm.LineEditor.LineEditor("re2_playground");
  editor.setPrompt(Cpp.std.string.basic_string("re2> "));

  Print('''
    Usage:
      /regex/        -- sets the current regular expression to regex
      anything else  -- matches the entered text against the specified regex
    ''');

  var re: RE2 = RE2.Make("^a*$");
  while (true) {
    var line: Cpp.OptionalString = editor.readLine();
    if (not line.has_value()) {
      break;
    }
    if (line.value()->starts_with("/") and line.value()->ends_with("/")) {
      re = RE2.Make(Cpp.AsStringView(line.value()->substr(1, line.value()->size() - 2)));
      continue;
    }
    if (re.Match(Cpp.AsStringView(*line.value()))) {
      Print("Match\n");
    } else {
      Print("No match\n");
    }
  }
  return 0;
}
