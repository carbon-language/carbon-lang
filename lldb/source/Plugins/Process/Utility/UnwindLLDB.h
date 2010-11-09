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

#include "lldb/lldb-private.h"
#include "lldb/lldb-types.h"
#include "lldb/Target/Unwind.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "RegisterContextLLDB.h"
#include "lldb/Target/RegisterContext.h"
#include <vector>


namespace lldb_private {

class UnwindLLDB : public lldb_private::Unwind
{
public: 
    UnwindLLDB (lldb_private::Thread &thread);
    
    virtual
    ~UnwindLLDB() { }
    
    void
    Clear()
    {
        m_frames.clear();
    }

    virtual uint32_t
    GetFrameCount();

    bool
    GetFrameInfoAtIndex (uint32_t frame_idx,
                         lldb::addr_t& cfa, 
                         lldb::addr_t& start_pc);
    
    lldb_private::RegisterContext *
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

private:
    struct Cursor
    {
        lldb::addr_t start_pc;  // The start address of the function/symbol for this frame - current pc if unknown
        lldb::addr_t cfa;       // The canonical frame address for this stack frame
        lldb_private::SymbolContext sctx;  // A symbol context we'll contribute to & provide to the StackFrame creation
        lldb::RegisterContextSP reg_ctx; // These are all RegisterContextLLDB's

        Cursor () : start_pc (LLDB_INVALID_ADDRESS), cfa (LLDB_INVALID_ADDRESS), sctx(), reg_ctx() { }
    private:
        DISALLOW_COPY_AND_ASSIGN (Cursor);
    };

    typedef lldb::SharedPtr<Cursor>::Type CursorSP;
    std::vector<CursorSP> m_frames;

    bool AddOneMoreFrame ();
    bool AddFirstFrame ();

    //------------------------------------------------------------------
    // For UnwindLLDB only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (UnwindLLDB);
};

}

#endif  // lldb_UnwindLLDB_h_
