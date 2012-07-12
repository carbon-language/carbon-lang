//===- examples/Tooling/ClangCheck.cpp - Clang check tool -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a clang-check tool that runs the
//  clang::SyntaxOnlyAction over a number of translation units.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"

using namespace clang::tooling;
using namespace llvm;

cl::opt<std::string> BuildPath(
  "p",
  cl::desc("<build-path>"),
  cl::Optional);

cl::list<std::string> SourcePaths(
  cl::Positional,
  cl::desc("<source0> [... <sourceN>]"),
  cl::OneOrMore);

static cl::extrahelp MoreHelp(
    "\n"
    "<build-path> is used to read a compile command database.\n"
    "\n"
    "For example, it can be a CMake build directory in which a file named\n"
    "compile_commands.json exists (use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\n"
    "CMake option to get this output). When no build path is specified,\n"
    "clang-check will attempt to locate it automatically using all parent\n"
    "paths of the first input file.\n"
    "\n"
    "<source0> ... specify the paths of source files. These paths are looked\n"
    "up in the compile command database. If the path of a file is absolute,\n"
    "it needs to point into CMake's source tree. If the path is relative,\n"
    "the current working directory needs to be in the CMake source tree and\n"
    "the file must be in a subdirectory of the current working directory.\n"
    "\"./\" prefixes in the relative files will be automatically removed,\n"
    "but the rest of a relative path must be a suffix of a path in the\n"
    "compile command database.\n"
    "\n"
    "For example, to use clang-check on all files in a subtree of the source\n"
    "tree, use:\n"
    "\n"
    "  find path/in/subtree -name '*.cpp'|xargs clang-check\n"
    "\n"
    "or using a specific build path:\n"
    "\n"
    "  find path/in/subtree -name '*.cpp'|xargs clang-check -p build/path\n"
    "\n"
    "Note, that path/in/subtree and current directory should follow the\n"
    "rules described above.\n"
    "\n"
);

int main(int argc, const char **argv) {
  llvm::OwningPtr<CompilationDatabase> Compilations(
    FixedCompilationDatabase::loadFromCommandLine(argc, argv));
  cl::ParseCommandLineOptions(argc, argv);
  if (!Compilations) {
    std::string ErrorMessage;
    if (!BuildPath.empty()) {
      Compilations.reset(
         CompilationDatabase::autoDetectFromDirectory(BuildPath, ErrorMessage));
    } else {
      Compilations.reset(CompilationDatabase::autoDetectFromSource(
          SourcePaths[0], ErrorMessage));
    }
    if (!Compilations)
      llvm::report_fatal_error(ErrorMessage);
  }
  ClangTool Tool(*Compilations, SourcePaths);
  return Tool.run(newFrontendActionFactory<clang::SyntaxOnlyAction>());
}
