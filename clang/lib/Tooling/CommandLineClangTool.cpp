//===--- CommandLineClangTool.cpp - command-line clang tools driver -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the CommandLineClangTool class used to run clang
//  tools as separate command-line applications with a consistent common
//  interface for handling compilation database and input files.
//
//  It provides a common subset of command-line options, common algorithm
//  for locating a compilation database and source files, and help messages
//  for the basic command-line interface.
//
//  It creates a CompilationDatabase, initializes a ClangTool and runs a
//  user-specified FrontendAction over all TUs in which the given files are
//  compiled.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommandLineClangTool.h"
#include "clang/Tooling/Tooling.h"

using namespace clang::tooling;
using namespace llvm;

static const char *MoreHelpText =
    "\n"
    "-p <build-path> is used to read a compile command database.\n"
    "\n"
    "\tFor example, it can be a CMake build directory in which a file named\n"
    "\tcompile_commands.json exists (use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\n"
    "\tCMake option to get this output). When no build path is specified,\n"
    "\tclang-check will attempt to locate it automatically using all parent\n"
    "\tpaths of the first input file. See:\n"
    "\thttp://clang.llvm.org/docs/HowToSetupToolingForLLVM.html for an\n"
    "\texample of setting up Clang Tooling on a source tree.\n"
    "\n"
    "<source0> ... specify the paths of source files. These paths are looked\n"
    "\tup in the compile command database. If the path of a file is absolute,\n"
    "\tit needs to point into CMake's source tree. If the path is relative,\n"
    "\tthe current working directory needs to be in the CMake source tree and\n"
    "\tthe file must be in a subdirectory of the current working directory.\n"
    "\t\"./\" prefixes in the relative files will be automatically removed,\n"
    "\tbut the rest of a relative path must be a suffix of a path in the\n"
    "\tcompile command database.\n"
    "\n";

CommandLineClangTool::CommandLineClangTool() :
    BuildPath("p", cl::desc("Build path"), cl::Optional),
    SourcePaths(cl::Positional, cl::desc("<source0> [... <sourceN>]"),
                cl::OneOrMore),
    MoreHelp(MoreHelpText) {
}

void CommandLineClangTool::initialize(int argc, const char **argv) {
  Compilations.reset(FixedCompilationDatabase::loadFromCommandLine(argc, argv));
  cl::ParseCommandLineOptions(argc, argv);
  if (!Compilations) {
    std::string ErrorMessage;
    if (!BuildPath.empty()) {
      Compilations.reset(CompilationDatabase::autoDetectFromDirectory(
                              BuildPath, ErrorMessage));
    } else {
      Compilations.reset(CompilationDatabase::autoDetectFromSource(
                              SourcePaths[0], ErrorMessage));
    }
    if (!Compilations)
      llvm::report_fatal_error(ErrorMessage);
  }
}

int CommandLineClangTool::run(FrontendActionFactory *ActionFactory) {
  ClangTool Tool(*Compilations, SourcePaths);
  return Tool.run(ActionFactory);
}
