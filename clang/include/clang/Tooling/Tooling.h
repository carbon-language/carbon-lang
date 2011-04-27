//===--- Tooling.h - Framework for standalone Clang tools -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements functions to run clang tools standalone instead
//  of running them as a plugin.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_TOOLING_H
#define LLVM_CLANG_TOOLING_TOOLING_H

#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace clang {

class FrontendAction;

namespace tooling {

/// \brief Runs (and deletes) the tool on 'Code' with the -fsynatx-only flag.
///
/// \param ToolAction The action to run over the code.
/// \param Code C++ code.
///
/// \return - True if 'ToolAction' was successfully executed.
bool RunSyntaxOnlyToolOnCode(
    clang::FrontendAction *ToolAction, llvm::StringRef Code);

/// \brief Runs (and deletes) the tool with the given Clang flags.
///
/// \param ToolAction The action to run over the code.
/// \param Argc The number of elements in Argv.
/// \param Argv The command line arguments, including the path the binary
/// was started with (Argv[0]).
bool RunToolWithFlags(
    clang::FrontendAction* ToolAction, int Argc, char *Argv[]);

/// \brief Converts a vector<string> into a vector<char*> suitable to pass
/// to main-style functions taking (int Argc, char *Argv[]).
std::vector<char*> CommandLineToArgv(const std::vector<std::string>* Command);

/// \brief Specifies the working directory and command of a compilation.
struct CompileCommand {
  /// \brief The working directory the command was executed from.
  std::string Directory;

  /// \brief The command line that was executed.
  std::vector<std::string> CommandLine;
};

/// \brief Looks up the compile command for 'FileName' in 'JsonDatabase'.
///
/// \param FileName The path to an input file for which we want the compile
/// command line. If the 'JsonDatabase' was created by CMake, this must be
/// an absolute path inside the CMake source directory which does not have
/// symlinks resolved.
///
/// \param JsonDatabase A JSON formatted list of compile commands. This lookup
/// command supports only a subset of the JSON standard as written by CMake.
///
/// \param ErrorMessage If non-empty, an error occurred and 'ErrorMessage' will
/// be set to contain the error message. In this case CompileCommand will
/// contain an empty directory and command line.
///
/// \see JsonCompileCommandLineDatabase
CompileCommand FindCompileArgsInJsonDatabase(
    llvm::StringRef FileName, llvm::StringRef JsonDatabase,
    std::string &ErrorMessage);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_TOOLING_H
