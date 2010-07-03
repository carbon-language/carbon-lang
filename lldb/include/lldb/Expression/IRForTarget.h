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
#include "llvm/PassManager.h"

namespace llvm {
    class BasicBlock;
    class Module;
}

namespace lldb_private {
    class ClangExpressionDeclMap;
}

class IRForTarget : public llvm::ModulePass
{
public:
    IRForTarget(const void *pid,
                lldb_private::ClangExpressionDeclMap *decl_map);
    ~IRForTarget();
    bool runOnModule(llvm::Module &M);
    void assignPassManager(llvm::PMStack &PMS,
                           llvm::PassManagerType T = llvm::PMT_ModulePassManager);
    llvm::PassManagerType getPotentialPassManagerType() const;
private:
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    
    lldb_private::ClangExpressionDeclMap *m_decl_map;
};

#endif