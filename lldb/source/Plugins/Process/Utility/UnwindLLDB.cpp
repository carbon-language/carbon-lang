//===-- UnwindLLDB.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Utility/ArchDefaultUnwindPlan.h"
#include "UnwindLLDB.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Core/Log.h"

using namespace lldb;
using namespace lldb_private;

UnwindLLDB::UnwindLLDB (Thread &thread) :
    Unwind (thread),
    m_frames()
{
}

uint32_t
UnwindLLDB::GetFrameCount()
{
    Log *log = GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND);
    if (m_frames.empty())
    {
        // First, set up the 0th (initial) frame
        CursorSP first_cursor_sp(new Cursor);
        RegisterContextSP no_frame; // an empty shared pointer
        RegisterContextLLDB *first_register_ctx = new RegisterContextLLDB(m_thread, no_frame, first_cursor_sp->sctx, 0);
        if (!first_register_ctx->IsValid())
        {
            delete first_register_ctx;
            return 0;
        }
        if (!first_register_ctx->GetCFA (first_cursor_sp->cfa))
        {
            delete first_register_ctx;
            return 0;
        }
        if (!first_register_ctx->GetPC (first_cursor_sp->start_pc))
        {
            delete first_register_ctx;
            return 0;
        }
        // Reuse the StackFrame provided by the processor native machine context for the first frame
        first_register_ctx->SetStackFrame (m_thread.GetStackFrameAtIndex(0).get());
        RegisterContextSP temp_rcs(first_register_ctx);
        first_cursor_sp->reg_ctx = temp_rcs;
        m_frames.push_back (first_cursor_sp);

        // Now walk up the rest of the stack
        while (1)
        {
            CursorSP cursor_sp(new Cursor);
            RegisterContextLLDB *register_ctx;
            int cur_idx = m_frames.size ();
            register_ctx = new RegisterContextLLDB (m_thread, m_frames[cur_idx - 1]->reg_ctx, cursor_sp->sctx, cur_idx);
            if (!register_ctx->IsValid())
            {
                delete register_ctx;
                if (log)
                {
                    log->Printf("%*sFrame %d invalid RegisterContext for this frame, stopping stack walk", 
                                cur_idx < 100 ? cur_idx : 100, "", cur_idx);
                }
                break;
            }
            if (!register_ctx->GetCFA (cursor_sp->cfa))
            {
                delete register_ctx;
                if (log)
                {
                    log->Printf("%*sFrame %d did not get CFA for this frame, stopping stack walk",
                                cur_idx < 100 ? cur_idx : 100, "", cur_idx);
                }
                break;
            }
            if (cursor_sp->cfa == (addr_t) -1 || cursor_sp->cfa == 1 || cursor_sp->cfa == 0)
            {
                delete register_ctx;
                if (log)
                {
                    log->Printf("%*sFrame %d did not get a valid CFA for this frame, stopping stack walk",
                                cur_idx < 100 ? cur_idx : 100, "", cur_idx);
                }
                break;
            }
            if (!register_ctx->GetPC (cursor_sp->start_pc))
            {
                delete register_ctx;
                if (log)
                {
                    log->Printf("%*sFrame %d did not get PC for this frame, stopping stack walk",
                                cur_idx < 100 ? cur_idx : 100, "", cur_idx);
                }
                break;
            }
            RegisterContextSP temp_rcs(register_ctx);
            StackFrame *frame = new StackFrame(cur_idx, cur_idx, m_thread, temp_rcs, cursor_sp->cfa, cursor_sp->start_pc, &(cursor_sp->sctx));
            register_ctx->SetStackFrame (frame);
            cursor_sp->reg_ctx = temp_rcs;
            m_frames.push_back (cursor_sp);
        }
    }
    return m_frames.size ();
}

bool
UnwindLLDB::GetFrameInfoAtIndex (uint32_t idx, addr_t& cfa, addr_t& pc)
{
    // FIXME don't get the entire stack if it isn't needed.
    if (m_frames.size() == 0)
        GetFrameCount();

    if (idx < m_frames.size ())
    {
        cfa = m_frames[idx]->cfa;
        pc = m_frames[idx]->start_pc;
        return true;
    }
    return false;
}

RegisterContext *
UnwindLLDB::CreateRegisterContextForFrame (StackFrame *frame)
{
    uint32_t idx = frame->GetFrameIndex ();
    
    // FIXME don't get the entire stack if it isn't needed.
    if (m_frames.size() == 0)
        GetFrameCount();
    
    if (idx == 0)
    {
        return m_thread.GetRegisterContext();
    }
    if (idx < m_frames.size ())
        return m_frames[idx]->reg_ctx.get();
    return NULL;
}
