// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstring>
#include <iostream>

#include "executable_semantics/syntax.tab.h"
#include "executable_semantics/syntax_helpers.h"
#include "executable_semantics/tracing_flag.h"
#include "llvm/Support/CommandLine.h"

extern FILE* yyin;

int main(int argc, char* argv[]) {
  // yydebug = 1;

  using llvm::cl::desc;
  using llvm::cl::opt;
  opt<bool> quiet_option("quiet", desc("Disable tracing"));
  opt<std::string> input_filename(llvm::cl::Positional, desc("<input file>"));
  llvm::cl::ParseCommandLineOptions(argc, argv);

  if (input_filename.getNumOccurrences() > 0) {
    Carbon::input_filename = input_filename.c_str();
    yyin = fopen(input_filename.c_str(), "r");
    if (yyin == nullptr) {
      std::cerr << "Error opening '" << input_filename
                << "': " << strerror(errno) << std::endl;
      return 1;
    }
  }
  if (quiet_option) {
    Carbon::tracing_output = false;
  }
  yy::parser parse;
  parse.set_debug_level(10);
  return parse();
}
