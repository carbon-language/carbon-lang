//===--- IncrementalExecutor.h - Incremental Execution ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class which performs incremental code execution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H
#define LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"

#include <memory>

namespace llvm {
class Error;
class Module;
namespace orc {
class LLJIT;
class ThreadSafeContext;
} // namespace orc
} // namespace llvm

namespace clang {
class IncrementalExecutor {
  using CtorDtorIterator = llvm::orc::CtorDtorIterator;
  std::unique_ptr<llvm::orc::LLJIT> Jit;
  llvm::orc::ThreadSafeContext &TSCtx;

public:
  enum SymbolNameKind { IRName, LinkerName };

  IncrementalExecutor(llvm::orc::ThreadSafeContext &TSC, llvm::Error &Err,
                      const llvm::Triple &Triple);
  ~IncrementalExecutor();

  llvm::Error addModule(std::unique_ptr<llvm::Module> M);
  llvm::Error runCtors() const;
  llvm::Expected<llvm::JITTargetAddress>
  getSymbolAddress(llvm::StringRef Name, SymbolNameKind NameKind) const;
};

} // end namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H
