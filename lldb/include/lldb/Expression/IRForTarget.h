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
    class CallInst;
    class Constant;
    class Function;
    class Instruction;
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
    // pass to find the result variable created in the result synthesizer and
    // make a result variable out of it (or a void variable if there is no
    // result)
    bool createResultVariable(llvm::Module &M,
                              llvm::Function &F);
    
    // pass to rewrite Objective-C method calls to use the runtime function
    // sel_registerName
    bool RewriteObjCSelector(llvm::Instruction* selector_load,
                             llvm::Module &M);
    bool rewriteObjCSelectors(llvm::Module &M, 
                              llvm::BasicBlock &BB);
    
    // pass to find declarations of, and references to, persistent variables and
    // register them for (de)materialization
    bool RewritePersistentAlloc(llvm::Instruction *persistent_alloc,
                                llvm::Module &M);
    bool rewritePersistentAllocs(llvm::Module &M,
                                 llvm::BasicBlock &BB);
    
    // pass to register referenced variables and redirect functions at their
    // targets in the debugged process
    bool MaybeHandleVariable(llvm::Module &M, 
                             llvm::Value *V,
                             bool Store);
    bool MaybeHandleCall(llvm::Module &M,
                         llvm::CallInst *C);
    bool resolveExternals(llvm::Module &M,
                          llvm::BasicBlock &BB);
    
    // pass to find references to guard variables and excise them
    bool removeGuards(llvm::Module &M,
                      llvm::BasicBlock &BB);
    
    // pass to replace all identified variables with references to members of
    // the argument struct
    bool replaceVariables(llvm::Module &M,
                          llvm::Function &F);
    
    lldb_private::ClangExpressionDeclMap *m_decl_map;
    const llvm::TargetData *m_target_data;
    
    llvm::Constant *m_sel_registerName;
};

#endif
