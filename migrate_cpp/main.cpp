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
using clang::tooling::ClangTool;
using clang::tooling::CompilationDatabase;
using clang::tooling::FixedCompilationDatabase;

auto GetAST(const std::unique_ptr<CompilationDatabase>& common_comp_db,
            const llvm::StringRef build_path, const llvm::StringRef filename)
    -> std::unique_ptr<clang::ASTUnit> {
  std::string err;
  std::unique_ptr<CompilationDatabase> comp_db;
  if (!common_comp_db) {
    comp_db = CompilationDatabase::autoDetectFromSource(
        build_path.empty() ? filename : build_path, err);
    if (!comp_db) {
      llvm::errs()
          << "Error while trying to load a compilation database, running "
             "without flags.\n"
          << err;
      comp_db = std::make_unique<clang::tooling::FixedCompilationDatabase>(
          ".", std::vector<std::string>());
    }
  }
  // addExtraArgs(Compilations);
  std::array<std::string, 1> files = {{std::string(filename)}};
  ClangTool tool(comp_db ? *comp_db : *common_comp_db, files);
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
  cl::opt<std::string> build_path("p", cl::desc("Build path"), cl::init(""),
                                  cl::Optional);

  std::string err;
  std::unique_ptr<CompilationDatabase> common_comp_db =
      FixedCompilationDatabase::loadFromCommandLine(argc, argv, err);
  if (!common_comp_db && !err.empty()) {
    llvm::errs() << err;
  }
  if (!cl::ParseCommandLineOptions(argc, argv)) {
    cl::PrintOptionValues();
    return 1;
  }
  std::unique_ptr<clang::ASTUnit> source =
      GetAST(common_comp_db, build_path, source_path);
  return 0;
}
