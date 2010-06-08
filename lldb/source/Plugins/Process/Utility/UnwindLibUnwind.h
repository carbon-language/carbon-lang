//===-- UnwindLibUnwind.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_UnwindLibUnwind_h_
#define lldb_UnwindLibUnwind_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
#include "libunwind.h"

// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/Unwind.h"

class UnwindLibUnwind : public lldb_private::Unwind
{
public: 
    UnwindLibUnwind (lldb_private::Thread &thread, 
                     lldb_private::unw_addr_space_t addr_space);
    
    virtual
    ~UnwindLibUnwind()
    {
    }
    
    virtual void
    Clear()
    {
        m_cursors.clear();
    }

    virtual uint32_t
    GetFrameCount();

    bool
    GetFrameInfoAtIndex (uint32_t frame_idx,
                         lldb::addr_t& cfa, 
                         lldb::addr_t& pc);
    
    lldb_private::RegisterContext *
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    lldb_private::Thread &
    GetThread();

private:
    lldb_private::unw_addr_space_t m_addr_space;
    std::vector<lldb_private::unw_cursor_t> m_cursors;
    uint32_t m_pc_regnum;
    uint32_t m_sp_regnum;
    //------------------------------------------------------------------
    // For UnwindLibUnwind only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (UnwindLibUnwind);
};

#endif  // lldb_UnwindLibUnwind_h_
