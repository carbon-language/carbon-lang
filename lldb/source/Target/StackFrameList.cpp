//===-- StackFrameList.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StackFrameList.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Unwind.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// StackFrameList constructor
//----------------------------------------------------------------------
StackFrameList::StackFrameList(Thread &thread, bool show_inline_frames) :
    m_thread (thread),
    m_show_inlined_frames (show_inline_frames),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_actual_frames (),
    m_inline_frames (),
    m_current_frame_idx (0)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
StackFrameList::~StackFrameList()
{
}


uint32_t
StackFrameList::GetNumFrames()
{
    Mutex::Locker locker (m_mutex);

    if (m_show_inlined_frames)
    {
        if (m_inlined_frame_info.empty())
        {
            Unwind *unwinder = m_thread.GetUnwinder ();
            // If we are going to show inlined stack frames as actual frames,
            // we need to calculate all concrete frames first, then iterate
            // through all of them and count up how many inlined functions are
            // in each frame. We can then fill in m_inlined_frame_info with
            // the concrete frame index and inlined depth
            const uint32_t concrete_frame_count = unwinder->GetFrameCount();
            
            addr_t pc, cfa;
            InlinedFrameInfo inlined_frame_info;

            StackFrameSP frame_sp;
            for (uint32_t idx=0; idx<concrete_frame_count; ++idx)
            {
                if (idx == 0)
                {
                    m_thread.GetRegisterContext();
                    frame_sp.reset (new StackFrame (0, 
                                                    0, 
                                                    m_thread, 
                                                    m_thread.m_reg_context_sp, 
                                                    m_thread.m_reg_context_sp->GetSP(), 
                                                    0, 
                                                    m_thread.m_reg_context_sp->GetPC(), 
                                                    NULL));
                }
                else
                {
                    const bool success = unwinder->GetFrameInfoAtIndex(idx, cfa, pc);
                    assert (success);
                    frame_sp.reset (new StackFrame (m_inlined_frame_info.size(), idx, m_thread, cfa, 0, pc, NULL));
                }
                SetActualFrameAtIndex (idx, frame_sp);
                Block *block = frame_sp->GetSymbolContext (eSymbolContextBlock).block;

                inlined_frame_info.concrete_frame_index = idx;
                inlined_frame_info.inline_height = 0;
                inlined_frame_info.block = block;
                m_inlined_frame_info.push_back (inlined_frame_info);

                if (block)
                {
                    Block *inlined_block;
                    if (block->InlinedFunctionInfo())
                        inlined_block = block;
                    else
                        inlined_block = block->GetInlinedParent ();
                        
                    while (inlined_block)
                    {
                        inlined_frame_info.block = inlined_block;
                        inlined_frame_info.inline_height++;
                        m_inlined_frame_info.push_back (inlined_frame_info);
                        inlined_block = inlined_block->GetInlinedParent ();
                    }
                }
            }
        }
        return m_inlined_frame_info.size();
    }
    else
    {
        if (m_actual_frames.empty())
            m_actual_frames.resize(m_thread.GetUnwinder()->GetFrameCount());
            
        return m_actual_frames.size();
    }
    return 0;
}

lldb::StackFrameSP
StackFrameList::GetActualFrameAtIndex (uint32_t idx) const
{
    StackFrameSP frame_sp;
        if (idx < m_actual_frames.size())
            frame_sp = m_actual_frames[idx];
    return frame_sp;
}

lldb::StackFrameSP
StackFrameList::GetInlineFrameAtIndex (uint32_t idx) const
{
    StackFrameSP frame_sp;
    if (idx < m_inline_frames.size())
        frame_sp = m_inline_frames[idx];
    return frame_sp;
}


StackFrameSP
StackFrameList::GetFrameAtIndex (uint32_t idx)
{
    StackFrameSP frame_sp;
    {
        Mutex::Locker locker (m_mutex);
    
        if (m_show_inlined_frames)
        {
            frame_sp = GetInlineFrameAtIndex (idx);
        }
        else
        {
            frame_sp = GetActualFrameAtIndex (idx);
        }

        if (frame_sp.get())
        return frame_sp;

    // Special case the first frame (idx == 0) so that we don't need to
    // know how many stack frames there are to get it. If we need any other
    // frames, then we do need to know if "idx" is a valid index.
    if (idx == 0)
    {
        // If this is the first frame, we want to share the thread register
        // context with the stack frame at index zero.
        m_thread.GetRegisterContext();
        assert (m_thread.m_reg_context_sp.get());
        frame_sp.reset (new StackFrame (0, 
                                        0, 
                                        m_thread, 
                                        m_thread.m_reg_context_sp, 
                                        m_thread.m_reg_context_sp->GetSP(), 
                                        0, 
                                        m_thread.m_reg_context_sp->GetPC(), 
                                        NULL));
    }
    else if (idx < GetNumFrames())
    {
        if (m_show_inlined_frames)
        {
            if (m_inlined_frame_info[idx].inline_height == 0)
            {
                // Same as the concrete stack frame if block is NULL
                assert (m_inlined_frame_info[idx].concrete_frame_index < m_actual_frames.size());
                frame_sp = GetActualFrameAtIndex (m_inlined_frame_info[idx].concrete_frame_index);
            }
            else 
            {
                // We have blocks that were above an inlined function. Inlined
                // functions are represented as blocks with non-NULL inline
                // function info. Here we must reconstruct a frame by looking
                // at the block
                StackFrameSP previous_frame_sp (m_thread.GetStackFrameAtIndex (idx-1));

                SymbolContext inline_sc;
                
                Block *inlined_parent_block = m_inlined_frame_info[idx].block->GetInlinedParent();
                
                if (inlined_parent_block)
                    inlined_parent_block->CalculateSymbolContext (&inline_sc);
                else
                {
                    Block *parent_block = m_inlined_frame_info[idx].block->GetParent();
                    parent_block->CalculateSymbolContext(&inline_sc);
                }
                
                Address previous_frame_lookup_addr (previous_frame_sp->GetFrameCodeAddress());
                if (previous_frame_sp->IsConcrete () && previous_frame_sp->GetFrameIndex() > 0)
                    previous_frame_lookup_addr.Slide (-1);

                AddressRange range;
                m_inlined_frame_info[idx].block->GetRangeContainingAddress (previous_frame_lookup_addr, range);
                    
                const InlineFunctionInfo* inline_info = m_inlined_frame_info[idx].block->InlinedFunctionInfo();
                assert (inline_info);
                inline_sc.line_entry.range.GetBaseAddress() = previous_frame_sp->GetFrameCodeAddress();
                inline_sc.line_entry.file = inline_info->GetCallSite().GetFile();
                inline_sc.line_entry.line = inline_info->GetCallSite().GetLine();
                inline_sc.line_entry.column = inline_info->GetCallSite().GetColumn();

                StackFrameSP concrete_frame_sp (GetActualFrameAtIndex (m_inlined_frame_info[idx].concrete_frame_index));
                assert (previous_frame_sp.get());
                
                frame_sp.reset (new StackFrame (idx, 
                                                m_inlined_frame_info[idx].concrete_frame_index,
                                                m_thread, 
                                                concrete_frame_sp->GetRegisterContextSP (),
                                                concrete_frame_sp->GetStackID().GetCallFrameAddress(),  // CFA
                                                m_inlined_frame_info[idx].inline_height,                // Inline height
                                                range.GetBaseAddress(),
                                                &inline_sc));                                           // The symbol context for this inline frame
                
            }
        }
        else
        {
            Unwind *unwinder = m_thread.GetUnwinder ();
            if (unwinder)
            {
                addr_t pc, cfa;
                if (unwinder->GetFrameInfoAtIndex(idx, cfa, pc))
                    frame_sp.reset (new StackFrame (idx, idx, m_thread, cfa, 0, pc, NULL));
            }
        }
    }
    if (m_show_inlined_frames)
        SetInlineFrameAtIndex(idx, frame_sp);
    else
        SetActualFrameAtIndex(idx, frame_sp);
    return frame_sp;
            
    }
    return frame_sp;
}

bool
StackFrameList::SetActualFrameAtIndex (uint32_t idx, StackFrameSP &frame_sp)
{
    if (idx >= m_actual_frames.size())
        m_actual_frames.resize(idx + 1);
    // Make sure allocation succeeded by checking bounds again
    if (idx < m_actual_frames.size())
    {
        m_actual_frames[idx] = frame_sp;
        return true;
    }
    return false;   // resize failed, out of memory?
}

bool
StackFrameList::SetInlineFrameAtIndex (uint32_t idx, StackFrameSP &frame_sp)
{
    if (idx >= m_inline_frames.size())
        m_inline_frames.resize(idx + 1);
    // Make sure allocation succeeded by checking bounds again
    if (idx < m_inline_frames.size())
    {
        m_inline_frames[idx] = frame_sp;
        return true;
    }
    return false;   // resize failed, out of memory?
}

uint32_t
StackFrameList::GetCurrentFrameIndex () const
{
    Mutex::Locker locker (m_mutex);
    return m_current_frame_idx;
}


uint32_t
StackFrameList::SetCurrentFrame (lldb_private::StackFrame *frame)
{
    Mutex::Locker locker (m_mutex);
    const_iterator pos;
    const_iterator begin = m_show_inlined_frames ? m_inline_frames.begin() : m_actual_frames.begin();
    const_iterator end   = m_show_inlined_frames ? m_inline_frames.end()   : m_actual_frames.end();
    for (pos = begin; pos != end; ++pos)
    {
        if (pos->get() == frame)
        {
            m_current_frame_idx = std::distance (begin, pos);
            return m_current_frame_idx;
        }
    }
    m_current_frame_idx = 0;
    return m_current_frame_idx;
}

// Mark a stack frame as the current frame using the frame index
void
StackFrameList::SetCurrentFrameByIndex (uint32_t idx)
{
    Mutex::Locker locker (m_mutex);
    m_current_frame_idx = idx;
}

// The thread has been run, reset the number stack frames to zero so we can
// determine how many frames we have lazily.
void
StackFrameList::Clear ()
{
    Mutex::Locker locker (m_mutex);
    m_actual_frames.clear();
    m_inline_frames.clear();
    m_inlined_frame_info.clear();
}

void
StackFrameList::InvalidateFrames (uint32_t start_idx)
{
    Mutex::Locker locker (m_mutex);
    if (m_show_inlined_frames)
    {
        Clear();
    }
    else
    {
        const size_t num_frames = m_actual_frames.size();
        while (start_idx < num_frames)
        {
            m_actual_frames[start_idx].reset();
            ++start_idx;
        }
    }
}
