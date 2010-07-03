//===-- IRForTarget.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/IRForTarget.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/InstrTypes.h"
#include "llvm/Module.h"

#include "lldb/Core/dwarf.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"

#include <map>

using namespace llvm;

IRForTarget::IRForTarget(const void *pid,
                         lldb_private::ClangExpressionDeclMap *decl_map) :
    ModulePass(pid),
    m_decl_map(decl_map)
{
}

IRForTarget::~IRForTarget()
{
}

bool
IRForTarget::runOnBasicBlock(BasicBlock &BB)
{
    lldb_private::Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
        
    /////////////////////////////////////////////////////////////////////////
    // Prepare the current basic block for execution in the remote process
    //
    
    if (log)
    {
        log->Printf("Preparing basic block %s:",
                    BB.hasName() ? BB.getNameStr().c_str() : "[anonymous]");
        
        llvm::BasicBlock::iterator ii;
        
        for (ii = BB.begin();
             ii != BB.end();
             ++ii)
        {
            llvm::Instruction &inst = *ii;
            
            std::string s;
            raw_string_ostream os(s);
            
            inst.print(os);
            
            if (log)
                log->Printf("  %s", s.c_str());
        }
    }
    
    return true;
}

bool
IRForTarget::runOnModule(Module &M)
{
    lldb_private::Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    llvm::Function* function = M.getFunction(StringRef("___clang_expr"));
    
    if (!function)
    {
        if (log)
            log->Printf("Couldn't find ___clang_expr() in the module");
        
        return false;
    }
        
    llvm::Function::iterator bbi;
    
    for (bbi = function->begin();
         bbi != function->end();
         ++bbi)
    {
        runOnBasicBlock(*bbi);
    }
    
    return true;    
}

void
IRForTarget::assignPassManager(PMStack &PMS,
                             PassManagerType T)
{
}

PassManagerType
IRForTarget::getPotentialPassManagerType() const
{
    return PMT_ModulePassManager;
}
