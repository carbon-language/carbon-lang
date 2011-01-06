//===-- Unwind.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Unwind_h_
#define liblldb_Unwind_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"

namespace lldb_private {

class Unwind 
{
protected:
    //------------------------------------------------------------------
    // Classes that inherit from Unwind can see and modify these
    //------------------------------------------------------------------
    Unwind(Thread &thread) :
        m_thread (thread)
    {
    }

public:
    virtual
    ~Unwind()
    {
    }

    virtual void
    Clear() = 0;

    virtual uint32_t
    GetFrameCount() = 0;

    virtual bool
    GetFrameInfoAtIndex (uint32_t frame_idx,
                         lldb::addr_t& cfa, 
                         lldb::addr_t& pc) = 0;
    
    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (StackFrame *frame) = 0;

    Thread &
    GetThread()
    {
        return m_thread;
    }

protected:
    //------------------------------------------------------------------
    // Classes that inherit from Unwind can see and modify these
    //------------------------------------------------------------------
    Thread &m_thread;
private:
    DISALLOW_COPY_AND_ASSIGN (Unwind);
};

} // namespace lldb_private

#endif  // liblldb_Unwind_h_
