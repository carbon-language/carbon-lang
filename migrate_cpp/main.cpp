// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstring>
#include <iostream>

#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;
using clang::tooling::CompilationDatabase;
using clang::tooling::FixedCompilationDatabase;

int main(int argc, char* argv[]) {
  cl::opt<std::string> source_path(cl::Positional, cl::desc("<source>"),
                                   cl::Required);

  std::string err;
  std::unique_ptr<CompilationDatabase> comp_db =
      FixedCompilationDatabase::loadFromCommandLine(argc, argv, err);
  if (!comp_db && !err.empty()) {
    llvm::errs() << err;
  }
  if (!cl::ParseCommandLineOptions(argc, argv)) {
    cl::PrintOptionValues();
    return 1;
  }
  std::uniqu_ptr<ASTUnit> source = GetAST(comp_db, source_path);
  return 0;
}
