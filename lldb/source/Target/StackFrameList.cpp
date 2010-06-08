//===-- StackFrameList.cpp --------------------------------------*- C++ -*-===//
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
#include "lldb/Target/StackFrameList.h"
#include "lldb/Target/StackFrame.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// StackFrameList constructor
//----------------------------------------------------------------------
StackFrameList::StackFrameList() :
    m_mutex (Mutex::eMutexTypeRecursive),
    m_frames (),
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
StackFrameList::GetNumFrames() const
{
    Mutex::Locker locker (m_mutex);
    return m_frames.size();
}

// After we have determined the number of frames, we can set the count here
// and have the frame info be generated on demand.
void
StackFrameList::SetNumFrames(uint32_t count)
{
    Mutex::Locker locker (m_mutex);
    return m_frames.resize(count);
}

StackFrameSP
StackFrameList::GetFrameAtIndex (uint32_t idx) const
{
    StackFrameSP frame_sp;
    {
        Mutex::Locker locker (m_mutex);
        if (idx < m_frames.size())
            frame_sp = m_frames[idx];
    }
    return frame_sp;
}

bool
StackFrameList::SetFrameAtIndex (uint32_t idx, StackFrameSP &frame_sp)
{
    Mutex::Locker locker (m_mutex);
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
StackFrameList::GetCurrentFrameIndex () const
{
    Mutex::Locker locker (m_mutex);
    return m_current_frame_idx;
}


uint32_t
StackFrameList::SetCurrentFrame (lldb_private::StackFrame *frame)
{
    Mutex::Locker locker (m_mutex);
    const_iterator pos,
                   begin = m_frames.begin(),
                   end = m_frames.end();
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
    m_frames.clear();
}

void
StackFrameList::InvalidateFrames (uint32_t start_idx)
{
    Mutex::Locker locker (m_mutex);
    size_t num_frames = m_frames.size();
    while (start_idx < num_frames)
    {
        m_frames[start_idx].reset();
        ++start_idx;
    }
}
