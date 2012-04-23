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
//  A ClangTool is initialized with a CompilationDatabase and a set of files
//  to run over. The tool will then run a user-specified FrontendAction over
//  all TUs in which the given files are compiled.
//
//  It is also possible to run a FrontendAction over a snippet of code by
//  calling runToolOnCode, which is useful for unit testing.
//
//  Applications that need more fine grained control over how to run
//  multiple FrontendActions over code can use ToolInvocation.
//
//  Example tools:
//  - running clang -fsyntax-only over source code from an editor to get
//    fast syntax checks
//  - running match/replace tools over C++ code
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_TOOLING_H
#define LLVM_CLANG_TOOLING_TOOLING_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LLVM.h"
#include "clang/Driver/Util.h"
#include <string>
#include <vector>

namespace clang {

namespace driver {
class Compilation;
} // end namespace driver

class CompilerInvocation;
class SourceManager;
class FrontendAction;

namespace tooling {

class CompilationDatabase;

/// \brief Interface to generate clang::FrontendActions.
class FrontendActionFactory {
public:
  virtual ~FrontendActionFactory();

  /// \brief Returns a new clang::FrontendAction.
  ///
  /// The caller takes ownership of the returned action.
  virtual clang::FrontendAction *create() = 0;
};

/// \brief Returns a new FrontendActionFactory for a given type.
///
/// T must extend clang::FrontendAction.
///
/// Example:
/// FrontendActionFactory *Factory =
///   newFrontendActionFactory<clang::SyntaxOnlyAction>();
template <typename T>
FrontendActionFactory *newFrontendActionFactory();

/// \brief Returns a new FrontendActionFactory for any type that provides an
/// implementation of newFrontendAction().
///
/// FactoryT must implement: FrontendAction *newFrontendAction().
///
/// Example:
/// struct ProvidesFrontendActions {
///   FrontendAction *newFrontendAction();
/// } Factory;
/// FrontendActionFactory *FactoryAdapter =
///   newFrontendActionFactory(&Factory);
template <typename FactoryT>
FrontendActionFactory *newFrontendActionFactory(FactoryT *ActionFactory);

/// \brief Runs (and deletes) the tool on 'Code' with the -fsyntax-only flag.
///
/// \param ToolAction The action to run over the code.
/// \param Code C++ code.
/// \param FileName The file name which 'Code' will be mapped as.
///
/// \return - True if 'ToolAction' was successfully executed.
bool runToolOnCode(clang::FrontendAction *ToolAction, const Twine &Code,
                   const Twine &FileName = "input.cc");

/// \brief Utility to run a FrontendAction in a single clang invocation.
class ToolInvocation {
 public:
  /// \brief Create a tool invocation.
  ///
  /// \param CommandLine The command line arguments to clang.
  /// \param ToolAction The action to be executed. Class takes ownership.
  /// \param Files The FileManager used for the execution. Class does not take
  /// ownership.
  ToolInvocation(ArrayRef<std::string> CommandLine, FrontendAction *ToolAction,
                 FileManager *Files);

  /// \brief Map a virtual file to be used while running the tool.
  ///
  /// \param FilePath The path at which the content will be mapped.
  /// \param Content A null terminated buffer of the file's content.
  void mapVirtualFile(StringRef FilePath, StringRef Content);

  /// \brief Run the clang invocation.
  ///
  /// \returns True if there were no errors during execution.
  bool run();

 private:
  void addFileMappingsTo(SourceManager &SourceManager);

  bool runInvocation(const char *BinaryName,
                     clang::driver::Compilation *Compilation,
                     clang::CompilerInvocation *Invocation,
                     const clang::driver::ArgStringList &CC1Args,
                     clang::FrontendAction *ToolAction);

  std::vector<std::string> CommandLine;
  llvm::OwningPtr<FrontendAction> ToolAction;
  FileManager *Files;
  // Maps <file name> -> <file content>.
  llvm::StringMap<StringRef> MappedFileContents;
};

/// \brief Utility to run a FrontendAction over a set of files.
///
/// This class is written to be usable for command line utilities.
class ClangTool {
 public:
  /// \brief Constructs a clang tool to run over a list of files.
  ///
  /// \param Compilations The CompilationDatabase which contains the compile
  ///        command lines for the given source paths.
  /// \param SourcePaths The source files to run over. If a source files is
  ///        not found in Compilations, it is skipped.
  ClangTool(const CompilationDatabase &Compilations,
            ArrayRef<std::string> SourcePaths);

  /// \brief Map a virtual file to be used while running the tool.
  ///
  /// \param FilePath The path at which the content will be mapped.
  /// \param Content A null terminated buffer of the file's content.
  void mapVirtualFile(StringRef FilePath, StringRef Content);

  /// Runs a frontend action over all files specified in the command line.
  ///
  /// \param ActionFactory Factory generating the frontend actions. The function
  /// takes ownership of this parameter. A new action is generated for every
  /// processed translation unit.
  int run(FrontendActionFactory *ActionFactory);

  /// \brief Returns the file manager used in the tool.
  ///
  /// The file manager is shared between all translation units.
  FileManager &getFiles() { return Files; }

 private:
  // We store command lines as pair (file name, command line).
  typedef std::pair< std::string, std::vector<std::string> > CommandLine;
  std::vector<CommandLine> CommandLines;

  FileManager Files;
  // Contains a list of pairs (<file name>, <file content>).
  std::vector< std::pair<StringRef, StringRef> > MappedFileContents;
};

template <typename T>
FrontendActionFactory *newFrontendActionFactory() {
  class SimpleFrontendActionFactory : public FrontendActionFactory {
  public:
    virtual clang::FrontendAction *create() { return new T; }
  };

  return new SimpleFrontendActionFactory;
}

template <typename FactoryT>
FrontendActionFactory *newFrontendActionFactory(FactoryT *ActionFactory) {
  class FrontendActionFactoryAdapter : public FrontendActionFactory {
  public:
    explicit FrontendActionFactoryAdapter(FactoryT *ActionFactory)
      : ActionFactory(ActionFactory) {}

    virtual clang::FrontendAction *create() {
      return ActionFactory->newFrontendAction();
    }

  private:
    FactoryT *ActionFactory;
  };

  return new FrontendActionFactoryAdapter(ActionFactory);
}

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_TOOLING_H

