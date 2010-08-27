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
#include "lldb/Core/StreamFile.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Unwind.h"

//#define DEBUG_STACK_FRAMES 1

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// StackFrameList constructor
//----------------------------------------------------------------------
StackFrameList::StackFrameList(Thread &thread, StackFrameList *prev_frames, bool show_inline_frames) :
    m_thread (thread),
    m_prev_frames_ap (prev_frames),
    m_show_inlined_frames (show_inline_frames),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_frames (),
    m_selected_frame_idx (0)
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

    if (m_frames.size() <= 1)
    {
        if (m_show_inlined_frames)
        {
#if defined (DEBUG_STACK_FRAMES)
            StreamFile s(stdout);
#endif
            Unwind *unwinder = m_thread.GetUnwinder ();
            addr_t pc, cfa;

            // If we are going to show inlined stack frames as actual frames,
            // we need to calculate all concrete frames first, then iterate
            // through all of them and count up how many inlined functions are
            // in each frame. 
            const uint32_t unwind_frame_count = unwinder->GetFrameCount();
            
            StackFrameSP unwind_frame_sp;
            for (uint32_t idx=0; idx<unwind_frame_count; ++idx)
            {
                if (idx == 0)
                {
                    // We might have already created frame zero, only create it
                    // if we need to
                    if (m_frames.empty())
                    {
                        m_thread.GetRegisterContext();
                        unwind_frame_sp.reset (new StackFrame (m_frames.size(), 
                                                               idx, 
                                                               m_thread, 
                                                               m_thread.m_reg_context_sp, 
                                                               m_thread.m_reg_context_sp->GetSP(), 
                                                               m_thread.m_reg_context_sp->GetPC(), 
                                                               NULL));
                        m_frames.push_back (unwind_frame_sp);
                    }
                    else
                    {
                        unwind_frame_sp = m_frames.front();
                    }
                }
                else
                {
                    const bool success = unwinder->GetFrameInfoAtIndex(idx, cfa, pc);
                    assert (success);
                    unwind_frame_sp.reset (new StackFrame (m_frames.size(), idx, m_thread, cfa, pc, NULL));
                    m_frames.push_back (unwind_frame_sp);
                }

                Block *block = unwind_frame_sp->GetSymbolContext (eSymbolContextBlock).block;
                
                if (block)
                {
                    for (block = block->GetContainingInlinedBlock(); block != NULL; block = block->GetInlinedParent ())
                    {
                        SymbolContext inline_sc;
                        Block *parent_block = block->GetInlinedParent();

                        const bool is_inlined_frame = parent_block != NULL;
                    
                        if (parent_block == NULL)
                            parent_block = block->GetParent();
                        
                        parent_block->CalculateSymbolContext (&inline_sc);
                    
                        Address previous_frame_lookup_addr (m_frames.back()->GetFrameCodeAddress());
                        if (unwind_frame_sp->GetFrameIndex() > 0 && m_frames.back().get() == unwind_frame_sp.get())
                            previous_frame_lookup_addr.Slide (-1);
                    
                        AddressRange range;
                        block->GetRangeContainingAddress (previous_frame_lookup_addr, range);
                    
                        const InlineFunctionInfo* inline_info = block->InlinedFunctionInfo();
                        assert (inline_info);
                        inline_sc.line_entry.range.GetBaseAddress() = m_frames.back()->GetFrameCodeAddress();
                        inline_sc.line_entry.file = inline_info->GetCallSite().GetFile();
                        inline_sc.line_entry.line = inline_info->GetCallSite().GetLine();
                        inline_sc.line_entry.column = inline_info->GetCallSite().GetColumn();
                                        
                        StackFrameSP frame_sp(new StackFrame (m_frames.size(),
                                                              idx,
                                                              m_thread,
                                                              unwind_frame_sp->GetRegisterContextSP (),
                                                              unwind_frame_sp->GetStackID().GetCallFrameAddress(),  // CFA
                                                              range.GetBaseAddress(),
                                                              &inline_sc));                                           // The symbol context for this inline frame
                        
                        if (is_inlined_frame)
                            frame_sp->SetInlineBlockID (block->GetID());
                        
                        m_frames.push_back (frame_sp);
                    }
                }
            }
            StackFrameList *prev_frames = m_prev_frames_ap.get();
            if (prev_frames)
            {
                StackFrameList *curr_frames = this;

#if defined (DEBUG_STACK_FRAMES)
                s.PutCString("prev_frames:\n");
                prev_frames->Dump (&s);
                s.PutCString("curr_frames:\n");
                curr_frames->Dump (&s);
                s.EOL();
#endif
                size_t curr_frame_num, prev_frame_num;
                
                for (curr_frame_num = curr_frames->m_frames.size(), prev_frame_num = prev_frames->m_frames.size();
                     curr_frame_num > 0 && prev_frame_num > 0;
                     --curr_frame_num, --prev_frame_num)
                {
                    const size_t curr_frame_idx = curr_frame_num-1;
                    const size_t prev_frame_idx = prev_frame_num-1;
                    StackFrameSP curr_frame_sp (curr_frames->m_frames[curr_frame_idx]);
                    StackFrameSP prev_frame_sp (prev_frames->m_frames[prev_frame_idx]);

#if defined (DEBUG_STACK_FRAMES)
                    s.Printf("\nCurrent frame #%u ", curr_frame_idx);
                    if (curr_frame_sp)
                        curr_frame_sp->Dump (&s, true);
                    else
                        s.PutCString("NULL");
                    s.Printf("\nPrevious frame #%u ", prev_frame_idx);
                    if (prev_frame_sp)
                        prev_frame_sp->Dump (&s, true);
                    else
                        s.PutCString("NULL");
                    s.EOL();
#endif

                    StackFrame *curr_frame = curr_frame_sp.get();
                    StackFrame *prev_frame = prev_frame_sp.get();
                    
                    if (curr_frame == NULL || prev_frame == NULL)
                        break;

                    // Do a quick sanity check to see if the CFA values are the same.
                    if (curr_frame->m_id.GetCallFrameAddress() != prev_frame->m_id.GetCallFrameAddress())
                        break;

                    // Now check our function or symbol
                    SymbolContext curr_sc (curr_frame->GetSymbolContext (eSymbolContextFunction | eSymbolContextBlock | eSymbolContextSymbol));
                    SymbolContext prev_sc (prev_frame->GetSymbolContext (eSymbolContextFunction | eSymbolContextBlock | eSymbolContextSymbol));
                    if (curr_sc.function && curr_sc.function == prev_sc.function)
                    {
                        // Same function
                        if (curr_sc.block != prev_sc.block)
                        {
                            // Same function different block
                            if (m_show_inlined_frames)
                                break;
                            else
                                prev_frame->SetSymbolContext (curr_frame->m_sc);
                        }
                    }
                    else if (curr_sc.symbol && curr_sc.symbol == prev_sc.symbol)
                    {
                        // Same symbol
                    }
                    else if (curr_frame->GetFrameCodeAddress() != prev_frame->GetFrameCodeAddress())
                    {
                        // No symbols for this frame and the PC was different
                        break;
                    }

                    if (curr_frame->GetFrameCodeAddress() != prev_frame->GetFrameCodeAddress())
                    {
#if defined (DEBUG_STACK_FRAMES)
                        s.Printf("\nUpdating frame code address and symbol context in previous frame #%u to current frame #%u", prev_frame_idx, curr_frame_idx);
#endif
                        // We have a different code frame address, we might need to copy
                        // some stuff in prev_frame, yet update the code address...
                        prev_frame->SetFrameCodeAddress (curr_frame->GetFrameCodeAddress());
                        prev_frame->SetSymbolContext (curr_frame->m_sc);
                    }

                    curr_frames->m_frames[curr_frame_idx] = prev_frames->m_frames[prev_frame_idx];
                    
#if defined (DEBUG_STACK_FRAMES)
                    s.Printf("\nCopying previous frame #%u to current frame #%u", prev_frame_idx, curr_frame_idx);
#endif
                }
                // We are done with the old stack frame list, we can release it now
                m_prev_frames_ap.release();
                prev_frames = NULL;
            }
        }
        else
        {
            m_frames.resize(m_thread.GetUnwinder()->GetFrameCount());
        }
    }
    return m_frames.size();
}

void
StackFrameList::Dump (Stream *s)
{
    if (s == NULL)
        return;
    Mutex::Locker locker (m_mutex);

    const_iterator pos, begin = m_frames.begin(), end = m_frames.end();
    for (pos = begin; pos != end; ++pos)
    {
        StackFrame *frame = (*pos).get();
        s->Printf("%p: ", frame);
        if (frame)
            frame->Dump(s, true);
        else
            s->Printf("frame #%u", std::distance (begin, pos));
        s->EOL();
    }
    s->EOL();
}

StackFrameSP
StackFrameList::GetFrameAtIndex (uint32_t idx)
{
    StackFrameSP frame_sp;
    Mutex::Locker locker (m_mutex);
    if (idx < m_frames.size())
        frame_sp = m_frames[idx];

    if (frame_sp)
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
                                        m_thread.m_reg_context_sp->GetPC(), 
                                        NULL));
        
        if (m_show_inlined_frames)
        {
            Block *block = frame_sp->GetSymbolContext (eSymbolContextBlock).block;
            
            if (block)
            {
                Block *inline_block = block->GetContainingInlinedBlock();
                if (inline_block)
                    frame_sp->SetInlineBlockID (inline_block->GetID());
            }
        }
        SetFrameAtIndex(idx, frame_sp);
    }
    else if (idx < GetNumFrames())
    {
        if (m_show_inlined_frames)
        {
            // When inline frames are enabled we cache up all frames in GetNumFrames()
            frame_sp = m_frames[idx];
        }
        else
        {
            Unwind *unwinder = m_thread.GetUnwinder ();
            if (unwinder)
            {
                addr_t pc, cfa;
                if (unwinder->GetFrameInfoAtIndex(idx, cfa, pc))
                {
                    frame_sp.reset (new StackFrame (idx, idx, m_thread, cfa, pc, NULL));
                    SetFrameAtIndex(idx, frame_sp);
                }
            }
        }
        
    }
    return frame_sp;
}


bool
StackFrameList::SetFrameAtIndex (uint32_t idx, StackFrameSP &frame_sp)
{
    if (idx >= m_frames.size())
        m_frames.resize(idx + 1);
    // Make sure allocation succeeded by checking bounds again
    if (idx < m_frames.size())
    {
        m_frames[idx] = frame_sp;
        return true;
    }
    return false;   // resize failed, out of memory?
}

uint32_t
StackFrameList::GetSelectedFrameIndex () const
{
    Mutex::Locker locker (m_mutex);
    return m_selected_frame_idx;
}


uint32_t
StackFrameList::SetSelectedFrame (lldb_private::StackFrame *frame)
{
    Mutex::Locker locker (m_mutex);
    const_iterator pos;
    const_iterator begin = m_frames.begin();
    const_iterator end = m_frames.end();
    for (pos = begin; pos != end; ++pos)
    {
        if (pos->get() == frame)
        {
            m_selected_frame_idx = std::distance (begin, pos);
            return m_selected_frame_idx;
        }
    }
    m_selected_frame_idx = 0;
    return m_selected_frame_idx;
}

// Mark a stack frame as the current frame using the frame index
void
StackFrameList::SetSelectedFrameByIndex (uint32_t idx)
{
    Mutex::Locker locker (m_mutex);
    m_selected_frame_idx = idx;
}

// The thread has been run, reset the number stack frames to zero so we can
// determine how many frames we have lazily.
void
StackFrameList::Clear ()
{
    Mutex::Locker locker (m_mutex);
    m_frames.clear();
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
        const size_t num_frames = m_frames.size();
        while (start_idx < num_frames)
        {
            m_frames[start_idx].reset();
            ++start_idx;
        }
    }
}
