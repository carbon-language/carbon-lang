//===- Debugify.h - Attach synthetic debug info to everything -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file Interface to the `debugify` synthetic debug info testing utility.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OPT_DEBUGIFY_H
#define LLVM_TOOLS_OPT_DEBUGIFY_H

#include "llvm/IR/PassManager.h"

llvm::ModulePass *createDebugifyModulePass();
llvm::FunctionPass *createDebugifyFunctionPass();

struct NewPMDebugifyPass : public llvm::PassInfoMixin<NewPMDebugifyPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

llvm::ModulePass *
createCheckDebugifyModulePass(bool Strip = false,
                              llvm::StringRef NameOfWrappedPass = "");

llvm::FunctionPass *
createCheckDebugifyFunctionPass(bool Strip = false,
                                llvm::StringRef NameOfWrappedPass = "");

struct NewPMCheckDebugifyPass
    : public llvm::PassInfoMixin<NewPMCheckDebugifyPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

#endif // LLVM_TOOLS_OPT_DEBUGIFY_H
