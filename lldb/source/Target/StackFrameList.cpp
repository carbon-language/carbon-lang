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
#include "lldb/Symbol/Symbol.h"
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
StackFrameList::StackFrameList
(
    Thread &thread, 
    const lldb::StackFrameListSP &prev_frames_sp, 
    bool show_inline_frames
) :
    m_thread (thread),
    m_prev_frames_sp (prev_frames_sp),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_frames (),
    m_selected_frame_idx (0),
    m_show_inlined_frames (show_inline_frames)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
StackFrameList::~StackFrameList()
{
}


uint32_t
StackFrameList::GetNumFrames (bool can_create)
{
    Mutex::Locker locker (m_mutex);

    if (can_create && m_frames.size() <= 1)
    {
        if (m_show_inlined_frames)
        {
#if defined (DEBUG_STACK_FRAMES)
            StreamFile s(stdout, false);
#endif
            Unwind *unwinder = m_thread.GetUnwinder ();
            addr_t pc = LLDB_INVALID_ADDRESS;
            addr_t cfa = LLDB_INVALID_ADDRESS;

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
                        cfa = m_thread.m_reg_context_sp->GetSP();
                        m_thread.GetRegisterContext();
                        unwind_frame_sp.reset (new StackFrame (m_frames.size(), 
                                                               idx, 
                                                               m_thread, 
                                                               m_thread.m_reg_context_sp, 
                                                               cfa, 
                                                               m_thread.m_reg_context_sp->GetPC(), 
                                                               NULL));
                        m_frames.push_back (unwind_frame_sp);
                    }
                    else
                    {
                        unwind_frame_sp = m_frames.front();
                        cfa = unwind_frame_sp->m_id.GetCallFrameAddress();
                    }
                }
                else
                {
                    const bool success = unwinder->GetFrameInfoAtIndex(idx, cfa, pc);
                    assert (success);
                    unwind_frame_sp.reset (new StackFrame (m_frames.size(), idx, m_thread, cfa, pc, NULL));
                    m_frames.push_back (unwind_frame_sp);
                }

                Block *unwind_block = unwind_frame_sp->GetSymbolContext (eSymbolContextBlock).block;
                
                if (unwind_block)
                {
                    Block *inlined_block = unwind_block->GetContainingInlinedBlock();
                    if (inlined_block)
                    {
                        for (; inlined_block != NULL; inlined_block = inlined_block->GetInlinedParent ())
                        {
                            SymbolContext inline_sc;
                            Block *parent_block = inlined_block->GetInlinedParent();

                            const bool is_inlined_frame = parent_block != NULL;
                        
                            if (parent_block == NULL)
                                parent_block = inlined_block->GetParent();
                            
                            parent_block->CalculateSymbolContext (&inline_sc);
                        
                            Address previous_frame_lookup_addr (m_frames.back()->GetFrameCodeAddress());
                            if (unwind_frame_sp->GetFrameIndex() > 0 && m_frames.back().get() == unwind_frame_sp.get())
                                previous_frame_lookup_addr.Slide (-1);
                        
                            AddressRange range;
                            inlined_block->GetRangeContainingAddress (previous_frame_lookup_addr, range);
                        
                            const InlineFunctionInfo* inline_info = inlined_block->GetInlinedFunctionInfo();
                            assert (inline_info);
                            inline_sc.line_entry.range.GetBaseAddress() = m_frames.back()->GetFrameCodeAddress();
                            inline_sc.line_entry.file = inline_info->GetCallSite().GetFile();
                            inline_sc.line_entry.line = inline_info->GetCallSite().GetLine();
                            inline_sc.line_entry.column = inline_info->GetCallSite().GetColumn();
                                            
                            StackFrameSP frame_sp(new StackFrame (m_frames.size(),
                                                                  idx,
                                                                  m_thread,
                                                                  unwind_frame_sp->GetRegisterContextSP (),
                                                                  cfa,
                                                                  range.GetBaseAddress(),
                                                                  &inline_sc));                                           // The symbol context for this inline frame
                            
                            if (is_inlined_frame)
                            {
                                // Use the block with the inlined function info
                                // as the symbol context since we want this frame
                                // to have only the variables for the inlined function
                                frame_sp->SetSymbolContextScope (parent_block);
                            }
                            else
                            {
                                // This block is not inlined with means it has no
                                // inlined parents either, so we want to use the top
                                // most function block.
                                frame_sp->SetSymbolContextScope (&unwind_frame_sp->GetSymbolContext (eSymbolContextFunction).function->GetBlock(false));
                            }
                            
                            m_frames.push_back (frame_sp);
                        }
                    }
                }
            }

            if (m_prev_frames_sp)
            {
                StackFrameList *prev_frames = m_prev_frames_sp.get();
                StackFrameList *curr_frames = this;

#if defined (DEBUG_STACK_FRAMES)
                s.PutCString("\nprev_frames:\n");
                prev_frames->Dump (&s);
                s.PutCString("\ncurr_frames:\n");
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
                    s.Printf("\n\nCurr frame #%u ", curr_frame_idx);
                    if (curr_frame_sp)
                        curr_frame_sp->Dump (&s, true, false);
                    else
                        s.PutCString("NULL");
                    s.Printf("\nPrev frame #%u ", prev_frame_idx);
                    if (prev_frame_sp)
                        prev_frame_sp->Dump (&s, true, false);
                    else
                        s.PutCString("NULL");
#endif

                    StackFrame *curr_frame = curr_frame_sp.get();
                    StackFrame *prev_frame = prev_frame_sp.get();
                    
                    if (curr_frame == NULL || prev_frame == NULL)
                        break;

                    // Check the stack ID to make sure they are equal
                    if (curr_frame->GetStackID() != prev_frame->GetStackID())
                        break;

                    prev_frame->UpdatePreviousFrameFromCurrentFrame (*curr_frame);
                    // Now copy the fixed up previous frame into the current frames
                    // so the pointer doesn't change
                    m_frames[curr_frame_idx] = prev_frame_sp;
                    //curr_frame->UpdateCurrentFrameFromPreviousFrame (*prev_frame);
                    
#if defined (DEBUG_STACK_FRAMES)
                    s.Printf("\n    Copying previous frame to current frame");
#endif
                }
                // We are done with the old stack frame list, we can release it now
                m_prev_frames_sp.reset();
            }
            
#if defined (DEBUG_STACK_FRAMES)
                s.PutCString("\n\nNew frames:\n");
                Dump (&s);
                s.EOL();
#endif
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
        {
            frame->GetStackID().Dump (s);
            frame->DumpUsingSettingsFormat (s);
        }
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
                    
                    Function *function = frame_sp->GetSymbolContext (eSymbolContextFunction).function;
                    if (function)
                    {
                        // When we aren't showing inline functions we always use
                        // the top most function block as the scope.
                        frame_sp->SetSymbolContextScope (&function->GetBlock(false));
                    }
                    else 
                    {
                        // Set the symbol scope from the symbol regardless if it is NULL or valid.
                        frame_sp->SetSymbolContextScope (frame_sp->GetSymbolContext (eSymbolContextSymbol).symbol);
                    }
                    SetFrameAtIndex(idx, frame_sp);
                }
            }
        }
    }
    return frame_sp;
}

StackFrameSP
StackFrameList::GetFrameWithConcreteFrameIndex (uint32_t unwind_idx)
{
    // First try assuming the unwind index is the same as the frame index. The 
    // unwind index is always greater than or equal to the frame index, so it
    // is a good place to start. If we have inlined frames we might have 5
    // concrete frames (frame unwind indexes go from 0-4), but we might have 15
    // frames after we make all the inlined frames. Most of the time the unwind
    // frame index (or the concrete frame index) is the same as the frame index.
    uint32_t frame_idx = unwind_idx;
    StackFrameSP frame_sp (GetFrameAtIndex (frame_idx));
    while (frame_sp)
    {
        if (frame_sp->GetFrameIndex() == unwind_idx)
            break;
        frame_sp = GetFrameAtIndex (++frame_idx);
    }
    return frame_sp;
}

StackFrameSP
StackFrameList::GetFrameWithStackID (StackID &stack_id)
{
    uint32_t frame_idx = 0;
    StackFrameSP frame_sp;
    do
    {
        frame_sp = GetFrameAtIndex (frame_idx);
        if (frame_sp && frame_sp->GetStackID() == stack_id)
            break;
        frame_idx++;
    }
    while (frame_sp);
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

void
StackFrameList::Merge (std::auto_ptr<StackFrameList>& curr_ap, lldb::StackFrameListSP& prev_sp)
{
    Mutex::Locker curr_locker (curr_ap.get() ? curr_ap->m_mutex.GetMutex() : NULL);
    Mutex::Locker prev_locker (prev_sp.get() ? prev_sp->m_mutex.GetMutex() : NULL);

#if defined (DEBUG_STACK_FRAMES)
    StreamFile s(stdout, false);
    s.PutCString("\n\nStackFrameList::Merge():\nPrev:\n");
    if (prev_sp.get())
        prev_sp->Dump (&s);
    else
        s.PutCString ("NULL");
    s.PutCString("\nCurr:\n");
    if (curr_ap.get())
        curr_ap->Dump (&s);
    else
        s.PutCString ("NULL");
    s.EOL();
#endif

    if (curr_ap.get() == NULL || curr_ap->GetNumFrames (false) == 0)
    {
#if defined (DEBUG_STACK_FRAMES)
        s.PutCString("No current frames, leave previous frames alone...\n");
#endif
        curr_ap.release();
        return;
    }

    if (prev_sp.get() == NULL || prev_sp->GetNumFrames (false) == 0)
    {
#if defined (DEBUG_STACK_FRAMES)
        s.PutCString("No previous frames, so use current frames...\n");
#endif
        // We either don't have any previous frames, or since we have more than
        // one current frames it means we have all the frames and can safely
        // replace our previous frames.
        prev_sp.reset (curr_ap.release());
        return;
    }

    const uint32_t num_curr_frames = curr_ap->GetNumFrames (false);
    
    if (num_curr_frames > 1)
    {
#if defined (DEBUG_STACK_FRAMES)
        s.PutCString("We have more than one current frame, so use current frames...\n");
#endif
        // We have more than one current frames it means we have all the frames 
        // and can safely replace our previous frames.
        prev_sp.reset (curr_ap.release());

#if defined (DEBUG_STACK_FRAMES)
        s.PutCString("\nMerged:\n");
        prev_sp->Dump (&s);
#endif
        return;
    }

    StackFrameSP prev_frame_zero_sp(prev_sp->GetFrameAtIndex (0));
    StackFrameSP curr_frame_zero_sp(curr_ap->GetFrameAtIndex (0));
    StackID curr_stack_id (curr_frame_zero_sp->GetStackID());
    StackID prev_stack_id (prev_frame_zero_sp->GetStackID());

#if defined (DEBUG_STACK_FRAMES)
    const uint32_t num_prev_frames = prev_sp->GetNumFrames (false);
    s.Printf("\n%u previous frames with one current frame\n", num_prev_frames);
#endif

    // We have only a single current frame
    // Our previous stack frames only had a single frame as well...
    if (curr_stack_id == prev_stack_id)
    {
#if defined (DEBUG_STACK_FRAMES)
        s.Printf("\nPrevious frame #0 is same as current frame #0, merge the cached data\n");
#endif

        curr_frame_zero_sp->UpdateCurrentFrameFromPreviousFrame (*prev_frame_zero_sp);
//        prev_frame_zero_sp->UpdatePreviousFrameFromCurrentFrame (*curr_frame_zero_sp);
//        prev_sp->SetFrameAtIndex (0, prev_frame_zero_sp);
    }
    else if (curr_stack_id < prev_stack_id)
    {
#if defined (DEBUG_STACK_FRAMES)
        s.Printf("\nCurrent frame #0 has a stack ID that is less than the previous frame #0, insert current frame zero in front of previous\n");
#endif
        prev_sp->m_frames.insert (prev_sp->m_frames.begin(), curr_frame_zero_sp);
    }
    
    curr_ap.release();

#if defined (DEBUG_STACK_FRAMES)
    s.PutCString("\nMerged:\n");
    prev_sp->Dump (&s);
#endif


}

lldb::StackFrameSP
StackFrameList::GetStackFrameSPForStackFramePtr (StackFrame *stack_frame_ptr)
{
    const_iterator pos;
    const_iterator begin = m_frames.begin();
    const_iterator end = m_frames.end();
    lldb::StackFrameSP ret_sp;
    
    for (pos = begin; pos != end; ++pos)
    {
        if (pos->get() == stack_frame_ptr)
        {
            ret_sp = (*pos);
            break;
        }
    }
    return ret_sp;
}

size_t
StackFrameList::GetStatus (Stream& strm,
                           uint32_t first_frame,
                           uint32_t num_frames,
                           bool show_frame_info,
                           uint32_t num_frames_with_source,
                           uint32_t source_lines_before,
                           uint32_t source_lines_after)
{
    size_t num_frames_displayed = 0;
    
    if (num_frames == 0)
        return 0;
    
    StackFrameSP frame_sp;
    uint32_t frame_idx = 0;
    uint32_t last_frame;
    
    // Don't let the last frame wrap around...
    if (num_frames == UINT32_MAX)
        last_frame = UINT32_MAX;
    else
        last_frame = first_frame + num_frames;
    
    for (frame_idx = first_frame; frame_idx < last_frame; ++frame_idx)
    {
        frame_sp = GetFrameAtIndex(frame_idx);
        if (frame_sp.get() == NULL)
            break;
        
        if (!frame_sp->GetStatus (strm,
                                  show_frame_info,
                                  num_frames_with_source > first_frame - frame_idx,
                                  source_lines_before,
                                  source_lines_after))
            break;
        ++num_frames_displayed;
    }
    
    strm.IndentLess();
    return num_frames_displayed;
}

