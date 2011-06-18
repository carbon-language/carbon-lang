//===-- UnwindLLDB.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "lldb/Core/Log.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"

#include "UnwindLLDB.h"
#include "RegisterContextLLDB.h"

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
    if (m_frames.empty())
    {
//#define DEBUG_FRAME_SPEED 1
#if DEBUG_FRAME_SPEED
#define FRAME_COUNT 10000
        TimeValue time_value (TimeValue::Now());
#endif
        if (!AddFirstFrame ())
            return 0;

        ABI *abi = m_thread.GetProcess().GetABI().get();

        while (AddOneMoreFrame (abi))
        {
#if DEBUG_FRAME_SPEED
            if ((m_frames.size() % FRAME_COUNT) == 0)
            {
                TimeValue now(TimeValue::Now());
                uint64_t delta_t = now - time_value;
                printf ("%u frames in %llu.%09llu ms (%g frames/sec)\n", 
                        FRAME_COUNT,
                        delta_t / TimeValue::NanoSecPerSec, 
                        delta_t % TimeValue::NanoSecPerSec,
                        (float)FRAME_COUNT / ((float)delta_t / (float)TimeValue::NanoSecPerSec));
                time_value = now;
            }
#endif
        }
    }
    return m_frames.size ();
}

bool
UnwindLLDB::AddFirstFrame ()
{
    // First, set up the 0th (initial) frame
    CursorSP first_cursor_sp(new Cursor ());
    RegisterContextLLDB::SharedPtr reg_ctx_sp (new RegisterContextLLDB (m_thread, 
                                                                        RegisterContextLLDB::SharedPtr(), 
                                                                        first_cursor_sp->sctx, 
                                                                        0));
    if (reg_ctx_sp.get() == NULL)
        return false;
    
    if (!reg_ctx_sp->IsValid())
        return false;

    if (!reg_ctx_sp->GetCFA (first_cursor_sp->cfa))
        return false;

    if (!reg_ctx_sp->ReadPC (first_cursor_sp->start_pc))
        return false;

    // Everything checks out, so release the auto pointer value and let the
    // cursor own it in its shared pointer
    first_cursor_sp->reg_ctx = reg_ctx_sp;
    m_frames.push_back (first_cursor_sp);
    return true;
}

// For adding a non-zero stack frame to m_frames.
bool
UnwindLLDB::AddOneMoreFrame (ABI *abi)
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
    CursorSP cursor_sp(new Cursor ());

    // Frame zero is a little different
    if (m_frames.size() == 0)
        return false;

    uint32_t cur_idx = m_frames.size ();
    RegisterContextLLDB::SharedPtr reg_ctx_sp(new RegisterContextLLDB (m_thread, 
                                                                       m_frames[cur_idx - 1]->reg_ctx, 
                                                                       cursor_sp->sctx, 
                                                                       cur_idx));
    if (reg_ctx_sp.get() == NULL)
        return false;

    if (!reg_ctx_sp->IsValid())
    {
        if (log)
        {
            log->Printf("%*sFrame %d invalid RegisterContext for this frame, stopping stack walk", 
                        cur_idx < 100 ? cur_idx : 100, "", cur_idx);
        }
        return false;
    }
    if (!reg_ctx_sp->GetCFA (cursor_sp->cfa))
    {
        if (log)
        {
            log->Printf("%*sFrame %d did not get CFA for this frame, stopping stack walk",
                        cur_idx < 100 ? cur_idx : 100, "", cur_idx);
        }
        return false;
    }
    if (abi && !abi->CallFrameAddressIsValid(cursor_sp->cfa))
    {
        if (log)
        {
            log->Printf("%*sFrame %d did not get a valid CFA for this frame, stopping stack walk",
                        cur_idx < 100 ? cur_idx : 100, "", cur_idx);
        }
        return false;
    }
    if (!reg_ctx_sp->ReadPC (cursor_sp->start_pc))
    {
        if (log)
        {
            log->Printf("%*sFrame %d did not get PC for this frame, stopping stack walk",
                        cur_idx < 100 ? cur_idx : 100, "", cur_idx);
        }
        return false;
    }
    if (abi && !abi->CodeAddressIsValid (cursor_sp->start_pc))
    {
        if (log)
        {
            log->Printf("%*sFrame %d did not get a valid PC, stopping stack walk",
                        cur_idx < 100 ? cur_idx : 100, "", cur_idx);
        }
        return false;
    }
    if (!m_frames.empty())
    {
        if (m_frames.back()->start_pc == cursor_sp->start_pc)
        {
            if (m_frames.back()->cfa == cursor_sp->cfa)
                return false; // Infinite loop where the current cursor is the same as the previous one...
            else if (abi->StackUsesFrames())
            {
                // We might have a CFA that is not using the frame pointer and
                // we want to validate that the frame pointer is valid.
                if (reg_ctx_sp->GetFP() == 0)
                    return false;
            }
        }
    }
    cursor_sp->reg_ctx = reg_ctx_sp;
    m_frames.push_back (cursor_sp);
    return true;
}

bool
UnwindLLDB::GetFrameInfoAtIndex (uint32_t idx, addr_t& cfa, addr_t& pc)
{
    if (m_frames.size() == 0)
    {
        if (!AddFirstFrame())
            return false;
    }

    ABI *abi = m_thread.GetProcess().GetABI().get();

    while (idx >= m_frames.size() && AddOneMoreFrame (abi))
        ;

    if (idx < m_frames.size ())
    {
        cfa = m_frames[idx]->cfa;
        pc = m_frames[idx]->start_pc;
        return true;
    }
    return false;
}

lldb::RegisterContextSP
UnwindLLDB::CreateRegisterContextForFrame (StackFrame *frame)
{
    lldb::RegisterContextSP reg_ctx_sp;
    uint32_t idx = frame->GetConcreteFrameIndex ();

    if (idx == 0)
    {
        return m_thread.GetRegisterContext();
    }

    if (m_frames.size() == 0)
    {
        if (!AddFirstFrame())
            return reg_ctx_sp;
    }

    ABI *abi = m_thread.GetProcess().GetABI().get();

    while (idx >= m_frames.size() && AddOneMoreFrame (abi))
        ;

    if (idx < m_frames.size ())
        reg_ctx_sp = m_frames[idx]->reg_ctx;
    return reg_ctx_sp;
}
