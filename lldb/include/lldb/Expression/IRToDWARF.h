//===-- IRToDWARF.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IRToDWARF_h_
#define liblldb_IRToDWARF_h_

#include "llvm/Pass.h"
#include "llvm/PassManager.h"

namespace llvm {
    class BasicBlock;
    class Module;
}

namespace lldb_private {
    class ClangExpressionVariableList;
    class ClangExpressionDeclMap;
    class StreamString;
}

class Relocator;

class IRToDWARF : public llvm::ModulePass
{
public:
    IRToDWARF(const void *pid,
              lldb_private::ClangExpressionVariableList &variable_list, 
              lldb_private::ClangExpressionDeclMap *decl_map,
              lldb_private::StreamString &strm);
    ~IRToDWARF();
    bool runOnModule(llvm::Module &M);
    void assignPassManager(llvm::PMStack &PMS,
                           llvm::PassManagerType T = llvm::PMT_ModulePassManager);
    llvm::PassManagerType getPotentialPassManagerType() const;
private:
    bool runOnBasicBlock(llvm::BasicBlock &BB, Relocator &Relocator);
    
    lldb_private::ClangExpressionVariableList &m_variable_list;
    lldb_private::ClangExpressionDeclMap *m_decl_map;
    lldb_private::StreamString &m_strm;
};

#endif