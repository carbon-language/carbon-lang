// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstring>
#include <iostream>

#include "executable_semantics/syntax_helpers.h"

extern FILE* yyin;
extern auto yyparse() -> int;  // NOLINT(readability-identifier-naming)

int main(int argc, char* argv[]) {
  // yydebug = 1;

  if (argc > 1) {
    Carbon::input_filename = argv[1];
    yyin = fopen(argv[1], "r");
    if (yyin == nullptr) {
      std::cerr << "Error opening '" << argv[1] << "': " << strerror(errno)
                << std::endl;
      return 1;
    }
  }
  return yyparse();
}
