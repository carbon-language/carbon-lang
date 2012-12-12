//===-- cpp11-migrate/Cpp11Migrate.cpp - Main file C++11 migration tool ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the C++11 feature migration tool main function
/// and transformation framework.
///
/// Usage:
/// cpp11-migrate [-p <build-path>] <file1> <file2> ... [-- [compiler-options]]
///
/// Where <build-path> is a CMake build directory containing a file named
/// compile_commands.json which provides compiler options for building each
/// sourc file. If <build-path> is not provided the compile_commands.json file
/// is searched for through all parent directories.
///
/// Alternatively, one can provide compile options to be applied to every source
/// file after the optional '--'.
///
/// <file1>... specify the paths of files in the CMake source tree, with the
/// same requirements as other tools built on LibTooling.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CommonOptionsParser.h"

namespace cl = llvm::cl;
using namespace clang::tooling;


int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv);

  // TODO: Create transforms requested by command-line.
  ClangTool SyntaxTool(OptionsParser.GetCompilations(),
                       OptionsParser.GetSourcePathList());

  // First, let's check to make sure there were no errors.
  if (SyntaxTool.run(newFrontendActionFactory<clang::SyntaxOnlyAction>()) !=
      0) {
    return 1;
  }

  // TODO: Apply transforms

  return 0;
}
