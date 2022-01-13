//===- DependencyScanningWorker.h - clang-scan-deps worker ===---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_WORKER_H
#define LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_WORKER_H

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Lex/PreprocessorExcludedConditionalDirectiveSkipMapping.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include <string>

namespace clang {

class DependencyOutputOptions;

namespace tooling {
namespace dependencies {

class DependencyScanningWorkerFilesystem;

/// Compilation database that holds and reports a single compile command.
class SingleCommandCompilationDatabase : public CompilationDatabase {
  CompileCommand Command;

public:
  SingleCommandCompilationDatabase(CompileCommand Cmd)
      : Command(std::move(Cmd)) {}

  std::vector<CompileCommand>
  getCompileCommands(StringRef FilePath) const override {
    return {Command};
  }

  std::vector<CompileCommand> getAllCompileCommands() const override {
    return {Command};
  }
};

class DependencyConsumer {
public:
  virtual ~DependencyConsumer() {}

  virtual void
  handleDependencyOutputOpts(const DependencyOutputOptions &Opts) = 0;

  virtual void handleFileDependency(StringRef Filename) = 0;

  virtual void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) = 0;

  virtual void handleModuleDependency(ModuleDeps MD) = 0;

  virtual void handleContextHash(std::string Hash) = 0;
};

/// An individual dependency scanning worker that is able to run on its own
/// thread.
///
/// The worker computes the dependencies for the input files by preprocessing
/// sources either using a fast mode where the source files are minimized, or
/// using the regular processing run.
class DependencyScanningWorker {
public:
  DependencyScanningWorker(DependencyScanningService &Service);

  /// Run the dependency scanning tool for a given clang driver invocation (as
  /// specified for the given Input in the CDB), and report the discovered
  /// dependencies to the provided consumer.
  ///
  /// \returns A \c StringError with the diagnostic output if clang errors
  /// occurred, success otherwise.
  llvm::Error computeDependencies(const std::string &Input,
                                  StringRef WorkingDirectory,
                                  const CompilationDatabase &CDB,
                                  DependencyConsumer &Consumer);

private:
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  std::shared_ptr<PCHContainerOperations> PCHContainerOps;
  std::unique_ptr<ExcludedPreprocessorDirectiveSkipMapping> PPSkipMappings;

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> RealFS;
  /// The file system that is used by each worker when scanning for
  /// dependencies. This filesystem persists accross multiple compiler
  /// invocations.
  llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;
  /// The file manager that is reused accross multiple invocations by this
  /// worker. If null, the file manager will not be reused.
  llvm::IntrusiveRefCntPtr<FileManager> Files;
  ScanningOutputFormat Format;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_WORKER_H
