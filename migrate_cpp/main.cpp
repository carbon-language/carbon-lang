// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstring>
#include <iostream>

#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;
using clang::tooling::ClangTool;
using clang::tooling::CompilationDatabase;
using clang::tooling::JSONCommandLineSyntax;
using clang::tooling::JSONCompilationDatabase;

auto GetAST(const std::unique_ptr<CompilationDatabase>& common_comp_db,
            const llvm::StringRef build_path, const llvm::StringRef filename)
    -> std::unique_ptr<clang::ASTUnit> {
  std::string err;
  std::array<std::string, 1> files = {{std::string(filename)}};
  ClangTool tool(*common_comp_db, files);
  std::vector<std::unique_ptr<clang::ASTUnit>> asts;
  tool.buildASTs(asts);
  if (asts.size() != files.size()) {
    return nullptr;
  }
  return std::move(asts[0]);
}

int main(int argc, char* argv[]) {
  cl::opt<std::string> source_path(cl::Positional, cl::desc("<source>"),
                                   cl::Required);
  cl::opt<std::string> common_comp_db_path("comp-db", cl::desc("<path>"),
                                           cl::Required);
  cl::opt<std::string> build_path("p", cl::desc("Build path"), cl::init(""),
                                  cl::Optional);

  if (!cl::ParseCommandLineOptions(argc, argv)) {
    cl::PrintOptionValues();
    return 1;
  }

  std::string err;
  std::unique_ptr<CompilationDatabase> common_comp_db =
      JSONCompilationDatabase::loadFromFile(common_comp_db_path, err,
                                            JSONCommandLineSyntax::AutoDetect);
  if (!common_comp_db && !err.empty()) {
    llvm::errs() << err;
  }
  std::unique_ptr<clang::ASTUnit> source =
      GetAST(common_comp_db, build_path, source_path);
  return 0;
}
