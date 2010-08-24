//===-- UnwindLibUnwind.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Thread.h"
#include "UnwindLibUnwind.h"
#include "LibUnwindRegisterContext.h"

using namespace lldb;
using namespace lldb_private;

UnwindLibUnwind::UnwindLibUnwind (Thread &thread, unw_addr_space_t addr_space) :
    Unwind (thread),
    m_addr_space (addr_space),
    m_cursors()
{
    m_pc_regnum = thread.GetRegisterContext()->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    m_sp_regnum = thread.GetRegisterContext()->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
}

uint32_t
UnwindLibUnwind::GetFrameCount()
{
    if (m_cursors.empty())
    {
        unw_cursor_t cursor;
        unw_init_remote (&cursor, m_addr_space, &m_thread);

        m_cursors.push_back (cursor);

        while (1) 
        {
            int stepresult = unw_step (&cursor);
            if (stepresult > 0)
                m_cursors.push_back (cursor);
            else
                break;
        }
    }
    return m_cursors.size();
}

bool
UnwindLibUnwind::GetFrameInfoAtIndex (uint32_t idx, addr_t& cfa, addr_t& pc)
{
    const uint32_t frame_count = GetFrameCount();
    if (idx < frame_count)
    {
        int pc_err = unw_get_reg (&m_cursors[idx], m_pc_regnum, &pc);
        int sp_err = unw_get_reg (&m_cursors[idx], m_sp_regnum, &cfa);
        return pc_err == UNW_ESUCCESS && sp_err == UNW_ESUCCESS;
    }
    return false;
}
    
RegisterContext *
UnwindLibUnwind::CreateRegisterContextForFrame (StackFrame *frame)
{
    uint32_t idx = frame->GetConcreteFrameIndex ();
    const uint32_t frame_count = GetFrameCount();
    if (idx < frame_count)
        return new LibUnwindRegisterContext (m_thread, frame, m_cursors[idx]);
    return NULL;
}
