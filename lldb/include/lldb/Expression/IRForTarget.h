//===-- IRForTarget.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IRForTarget_h_
#define liblldb_IRForTarget_h_

#include "llvm/Pass.h"

namespace llvm {
    class BasicBlock;
    class Function;
    class Module;
    class TargetData;
    class Value;
}

namespace lldb_private {
    class ClangExpressionDeclMap;
}

class IRForTarget : public llvm::ModulePass
{
public:
    IRForTarget(const void *pid,
                lldb_private::ClangExpressionDeclMap *decl_map,
                const llvm::TargetData *target_data);
    ~IRForTarget();
    bool runOnModule(llvm::Module &M);
    void assignPassManager(llvm::PMStack &PMS,
                           llvm::PassManagerType T = llvm::PMT_ModulePassManager);
    llvm::PassManagerType getPotentialPassManagerType() const;
private:
    bool MaybeHandleVariable(llvm::Module &M, 
                             lldb_private::ClangExpressionDeclMap *DM,
                             llvm::Value *V,
                             bool Store);
    bool runOnBasicBlock(llvm::Module &M,
                         llvm::BasicBlock &BB);
    bool replaceVariables(llvm::Module &M,
                          llvm::Function *F);
    
    lldb_private::ClangExpressionDeclMap *m_decl_map;
    const llvm::TargetData *m_target_data;
};

#endif