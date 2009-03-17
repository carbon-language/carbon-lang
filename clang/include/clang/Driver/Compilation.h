//===--- Compilation.h - Compilation Task Data Structure --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_COMPILATION_H_
#define CLANG_DRIVER_COMPILATION_H_

#include "clang/Driver/Job.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
namespace driver {
  class ArgList;
  class JobList;
  class ToolChain;

/// Compilation - A set of tasks to perform for a single driver
/// invocation.
class Compilation {
  /// The default tool chain.
  ToolChain &DefaultToolChain;

  /// The original (untranslated) input argument list.
  ArgList *Args;

  /// The root list of jobs.
  JobList Jobs;

  /// Cache of translated arguments for a particular tool chain.
  llvm::DenseMap<const ToolChain*, ArgList*> TCArgs;

  /// Temporary files which should be removed on exit.
  llvm::SmallVector<const char*, 4> TempFiles;

  /// Result files which should be removed on failure.
  llvm::SmallVector<const char*, 4> ResultFiles;

public:
  Compilation(ToolChain &DefaultToolChain, ArgList *Args);
  ~Compilation();

  const ArgList &getArgs() const { return *Args; }
  JobList &getJobs() { return Jobs; }

  /// getArgsForToolChain - Return the argument list, possibly
  /// translated by the tool chain \arg TC (or by the default tool
  /// chain, if TC is not specified).
  const ArgList &getArgsForToolChain(const ToolChain *TC = 0);

  /// addTempFile - Add a file to remove on exit, and returns its
  /// argument.
  const char *addTempFile(const char *Name) { 
    TempFiles.push_back(Name); 
    return Name;
  }

  /// addResultFile - Add a file to remove on failure, and returns its
  /// argument.
  const char *addResultFile(const char *Name) {
    ResultFiles.push_back(Name);
    return Name;
  }

  /// Execute - Execute the compilation jobs and return an
  /// appropriate exit code.
  int Execute() const;
};

} // end namespace driver
} // end namespace clang

#endif
