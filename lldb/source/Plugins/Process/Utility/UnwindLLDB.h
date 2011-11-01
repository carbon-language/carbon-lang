//===-- UnwindLLDB.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_UnwindLLDB_h_
#define lldb_UnwindLLDB_h_

#include <vector>

#include "lldb/lldb-public.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Unwind.h"

namespace lldb_private {

class RegisterContextLLDB;

class UnwindLLDB : public lldb_private::Unwind
{
public: 
    UnwindLLDB (lldb_private::Thread &thread);
    
    virtual
    ~UnwindLLDB() { }

protected:
    friend class lldb_private::RegisterContextLLDB;

    struct RegisterLocation {
        enum RegisterLocationTypes
        {
            eRegisterNotSaved = 0,              // register was not preserved by callee.  If volatile reg, is unavailable
            eRegisterSavedAtMemoryLocation,     // register is saved at a specific word of target mem (target_memory_location)
            eRegisterInRegister,                // register is available in a (possible other) register (register_number)
            eRegisterSavedAtHostMemoryLocation, // register is saved at a word in lldb's address space
            eRegisterValueInferred              // register val was computed (and is in inferred_value)
        };
        int type;
        union
        {
            lldb::addr_t target_memory_location;
            uint32_t     register_number;       // in eRegisterKindLLDB register numbering system
            void*        host_memory_location;
            uint64_t     inferred_value;        // eRegisterValueInferred - e.g. stack pointer == cfa + offset
        } location;
    };

    void
    DoClear()
    {
        m_frames.clear();
    }

    virtual uint32_t
    DoGetFrameCount();

    bool
    DoGetFrameInfoAtIndex (uint32_t frame_idx,
                         lldb::addr_t& cfa, 
                         lldb::addr_t& start_pc);
    
    lldb::RegisterContextSP
    DoCreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    typedef lldb::SharedPtr<lldb_private::RegisterContextLLDB>::Type RegisterContextLLDBSharedPtr;

    // Needed to retrieve the "next" frame (e.g. frame 2 needs to retrieve frame 1's RegisterContextLLDB)
    // The RegisterContext for frame_num must already exist or this returns an empty shared pointer.
    RegisterContextLLDBSharedPtr
    GetRegisterContextForFrameNum (uint32_t frame_num);

    // Iterate over the RegisterContextLLDB's in our m_frames vector, look for the first one that
    // has a saved location for this reg.
    bool
    SearchForSavedLocationForRegister (uint32_t lldb_regnum, lldb_private::UnwindLLDB::RegisterLocation &regloc, uint32_t starting_frame_num);


private:

    struct Cursor
    {
        lldb::addr_t start_pc;  // The start address of the function/symbol for this frame - current pc if unknown
        lldb::addr_t cfa;       // The canonical frame address for this stack frame
        lldb_private::SymbolContext sctx;  // A symbol context we'll contribute to & provide to the StackFrame creation
        RegisterContextLLDBSharedPtr reg_ctx; // These are all RegisterContextLLDB's

        Cursor () : start_pc (LLDB_INVALID_ADDRESS), cfa (LLDB_INVALID_ADDRESS), sctx(), reg_ctx() { }
    private:
        DISALLOW_COPY_AND_ASSIGN (Cursor);
    };

    typedef lldb::SharedPtr<Cursor>::Type CursorSP;
    std::vector<CursorSP> m_frames;

    bool AddOneMoreFrame (ABI *abi);
    bool AddFirstFrame ();

    //------------------------------------------------------------------
    // For UnwindLLDB only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (UnwindLLDB);
};

}   // namespace lldb_private

#endif  // lldb_UnwindLLDB_h_
