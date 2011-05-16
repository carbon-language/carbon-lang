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
//  Usage:
//  clang-check <cmake-output-dir> <file1> <file2> ...
//
//  Where <cmake-output-dir> is a CMake build directory in which a file named
//  compile_commands.json exists (enable -DCMAKE_EXPORT_COMPILE_COMMANDS in
//  CMake to get this output).
//
//  <file1> ... specify the paths of files in the CMake source tree. This  path
//  is looked up in the compile command database. If the path of a file is
//  absolute, it needs to point into CMake's source tree. If the path is
//  relative, the current working directory needs to be in the CMake source
//  tree and the file must be in a subdirectory of the current working
//  directory. "./" prefixes in the relative files will be automatically
//  removed, but the rest of a relative path must be a suffix of a path in
//  the compile command line database.
//
//  For example, to use clang-check on all files in a subtree of the source
//  tree, use:
//
//    /path/in/subtree $ find . -name '*.cpp'| xargs clang-check /path/to/source
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

class SyntaxOnlyActionFactory : public clang::tooling::FrontendActionFactory {
 public:
  virtual clang::FrontendAction *New() { return new clang::SyntaxOnlyAction; }
};

int main(int argc, char **argv) {
  clang::tooling::ClangTool Tool(argc, argv);
  return Tool.Run(new SyntaxOnlyActionFactory);
}
