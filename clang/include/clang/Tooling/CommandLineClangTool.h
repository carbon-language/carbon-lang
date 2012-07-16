//===- CommandLineClangTool.h - command-line clang tools driver -*- C++ -*-===//
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
//  This class uses the Clang Tooling infrastructure, see
//    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
//  for details on setting it up with LLVM source tree.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_CLANG_INCLUDE_CLANG_TOOLING_COMMANDLINECLANGTOOL_H
#define LLVM_TOOLS_CLANG_INCLUDE_CLANG_TOOLING_COMMANDLINECLANGTOOL_H

#include "llvm/Support/CommandLine.h"
#include "clang/Tooling/CompilationDatabase.h"

namespace clang {

namespace tooling {

class CompilationDatabase;
class FrontendActionFactory;

/// \brief A common driver for command-line Clang tools.
///
/// Parses a common subset of command-line arguments, locates and loads a
/// compilation commands database, runs a tool with user-specified action. It
/// also contains a help message for the common command-line options.
/// An example of usage:
/// @code
/// int main(int argc, const char **argv) {
///   CommandLineClangTool Tool;
///   cl::extrahelp MoreHelp("\nMore help text...");
///   Tool.initialize(argc, argv);
///   return Tool.run(newFrontendActionFactory<clang::SyntaxOnlyAction>());
/// }
/// @endcode
///
class CommandLineClangTool {
public:
  /// Sets up command-line options and help messages.
  /// Add your own help messages after constructing this tool.
  CommandLineClangTool();

  /// Parses command-line, initializes a compilation database.
  /// This method exits program in case of error.
  void initialize(int argc, const char **argv);

  /// Runs a clang tool with an action created by \c ActionFactory.
  int run(FrontendActionFactory *ActionFactory);

private:
  llvm::OwningPtr<CompilationDatabase> Compilations;
  llvm::cl::opt<std::string> BuildPath;
  llvm::cl::list<std::string> SourcePaths;
  llvm::cl::extrahelp MoreHelp;
};

} // namespace tooling

} // namespace clang

#endif  // LLVM_TOOLS_CLANG_INCLUDE_CLANG_TOOLING_COMMANDLINECLANGTOOL_H
