//===-- IRToDWARF.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/IRToDWARF.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/InstrTypes.h"
#include "llvm/Module.h"

#include "lldb/Core/dwarf.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"
#include "lldb/Expression/ClangExpressionVariable.h"

#include <map>

using namespace llvm;

static char ID;

IRToDWARF::IRToDWARF(lldb_private::ClangExpressionVariableList &variable_list, 
                     lldb_private::ClangExpressionDeclMap *decl_map,
                     lldb_private::StreamString &strm) :
    ModulePass(&ID),
    m_variable_list(variable_list),
    m_decl_map(decl_map),
    m_strm(strm)
{
}

IRToDWARF::~IRToDWARF()
{
}

class Relocator
{
public:
    Relocator()
    {
    }
    
    ~Relocator()
    {
    }
    
    void MarkBasicBlock(BasicBlock *bb, uint16_t offset)
    {
        m_basic_blocks[bb] = offset;
    }
    
    bool BasicBlockIsMarked(BasicBlock *bb)
    {
        return m_basic_blocks.find(bb) != m_basic_blocks.end();
    }
    
    void MarkRelocation(BasicBlock *bb, uint16_t offset)
    {
        m_relocations[offset] = bb;
    }
    
    bool ResolveRelocations(lldb_private::StreamString &strm)
    {
        std::map<uint16_t, BasicBlock*>::const_iterator iter;
        
        lldb_private::StreamString swapper(0, 32, strm.GetByteOrder());
        
        // This array must be delete [] d at every exit
        size_t temporary_bufsize = strm.GetSize();
        uint8_t *temporary_buffer(new uint8_t[temporary_bufsize]);
        
        memcpy(temporary_buffer, strm.GetData(), temporary_bufsize);
                
        for (iter = m_relocations.begin();
             iter != m_relocations.end();
             ++iter)
        {
            const std::pair<uint16_t, BasicBlock*> &pair = *iter;
            
            uint16_t off = pair.first;
            BasicBlock *bb = pair.second;
            
            if (m_basic_blocks.find(bb) == m_basic_blocks.end())
            {
                delete [] temporary_buffer;
                return false;
            }
                
            uint16_t target_off = m_basic_blocks[bb];
            
            int16_t relative = (int16_t)target_off - (int16_t)off;
            
            swapper.Clear();
            swapper << relative;
            
            // off is intended to be the offset of the branch opcode (which is 
            // what the relative location is added to) so 
            // (temporary_buffer + off + 1) skips the opcode and writes to the
            // relative location
            memcpy(temporary_buffer + off + 1, swapper.GetData(), sizeof(uint16_t));
        }
        
        strm.Clear();
        strm.Write(temporary_buffer, temporary_bufsize);
        
        delete [] temporary_buffer;
        return true;
    }
private:
    std::map<BasicBlock*, uint16_t> m_basic_blocks;
    std::map<uint16_t, BasicBlock*> m_relocations;
};

bool
IRToDWARF::runOnBasicBlock(BasicBlock &BB, Relocator &R)
{    
    ///////////////////////////////////////
    // Mark the current block as visited
    //
    
    size_t stream_size = m_strm.GetSize();
    
    if (stream_size > 0xffff)
        return false;
    
    uint16_t offset = stream_size & 0xffff;
    
    R.MarkBasicBlock(&BB, offset);
    
    ////////////////////////////////////////////////
    // Translate the current basic block to DWARF
    //
    
    /////////////////////////////////////////////////
    // Visit all successors we haven't visited yet
    //
    
    TerminatorInst *arnold = BB.getTerminator();
    
    if (!arnold)
        return false;
    
    unsigned successor_index;
    unsigned num_successors = arnold->getNumSuccessors();
    
    for (successor_index = 0;
         successor_index < num_successors;
         ++successor_index)
    {
        BasicBlock *successor = arnold->getSuccessor(successor_index);
        
        if (!R.BasicBlockIsMarked(successor))
        {
            if (!runOnBasicBlock(*successor, R))
                return false;
        }
    }
    
    return true;
}

bool
IRToDWARF::runOnModule(Module &M)
{
    lldb_private::Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    llvm::Function* function = M.getFunction(StringRef("___clang_expr"));
    
    if (!function)
    {
        if (log)
            log->Printf("Couldn't find ___clang_expr() in the module");
        
        return 1;
    }
    
    Relocator relocator;
    
    if (!runOnBasicBlock(function->getEntryBlock(), relocator))
        return false;
    
    if (log)
    {
        std::string s;
        raw_string_ostream oss(s);
        
        M.print(oss, NULL);
        
        oss.flush();
        
        log->Printf("Module being translated to DWARF: \n%s", s.c_str());
    }
    
    // TEMPORARY: Fail in order to force execution in the target.
    return false;
    
    return relocator.ResolveRelocations(m_strm);    
}

void
IRToDWARF::assignPassManager(PMStack &PMS,
                                 PassManagerType T)
{
}

PassManagerType
IRToDWARF::getPotentialPassManagerType() const
{
    return PMT_ModulePassManager;
}
