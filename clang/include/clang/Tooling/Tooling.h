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
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/MemoryBuffer.h"
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
    clang::FrontendAction *ToolAction, int Argc, char *Argv[]);

/// \brief Converts a vector<string> into a vector<char*> suitable to pass
/// to main-style functions taking (int Argc, char *Argv[]).
std::vector<char*> CommandLineToArgv(const std::vector<std::string> *Command);

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

// Interface to generate clang::FrontendActions.
class FrontendActionFactory {
 public:
  virtual ~FrontendActionFactory();

  // Returns a new clang::FrontendAction. The caller takes ownership of the
  // returned action.
  virtual clang::FrontendAction* New() = 0;
};

/// \brief Utility to run a FrontendAction over a set of files.
///
/// This class is written to be usable for command line utilities.
class ClangTool {
 public:
  /// \brief Construct a clang tool from a command line.
  ///
  /// This will parse the command line parameters and print an error message
  /// and exit the program if the command line does not specify the required
  /// parameters.
  ///
  /// Usage:
  /// $ tool-name <cmake-output-dir> <file1> <file2> ...
  ///
  /// where <cmake-output-dir> is a CMake build directory in which a file named
  /// compile_commands.json exists (enable -DCMAKE_EXPORT_COMPILE_COMMANDS in
  /// CMake to get this output).
  ///
  /// <file1> ... specify the paths of files in the CMake source tree. This
  /// path is looked up in the compile command database. If the path of a file
  /// is absolute, it needs to point into CMake's source tree. If the path is
  /// relative, the current working directory needs to be in the CMake source
  /// tree and the file must be in a subdirectory of the current working
  /// directory. "./" prefixes in the relative files will be automatically
  /// removed, but the rest of a relative path must be a suffix of a path in
  /// the compile command line database.
  ///
  /// For example, to use a tool on all files in a subtree of the source
  /// tree, use:
  ///
  ///   /path/in/subtree $ find . -name '*.cpp' |
  ///       xargs tool-name /path/to/source
  ///
  /// \param argc The argc argument from main.
  /// \param argv The argv argument from main.
  ClangTool(int argc, char **argv);

  /// Runs a frontend action over all files specified in the command line.
  ///
  /// \param ActionFactory Factory generating the frontend actions. The function
  /// takes ownership of this parameter. A new action is generated for every
  /// processed translation unit.
  int Run(FrontendActionFactory *ActionFactory);

 private:
  std::vector<std::string> Files;
  llvm::OwningPtr<llvm::MemoryBuffer> JsonDatabase;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_TOOLING_H
