//===-- StackFrameList.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StackFrameList_h_
#define liblldb_StackFrameList_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Host/Mutex.h"
#include "lldb/Target/StackFrame.h"

namespace lldb_private {

class StackFrameList
{
public:
    friend class Thread;
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    StackFrameList();

    virtual
    ~StackFrameList();

    uint32_t
    GetNumFrames() const;

    lldb::StackFrameSP
    GetFrameAtIndex (uint32_t idx) const;

    bool
    SetFrameAtIndex (uint32_t idx, lldb::StackFrameSP &frame_sp);

    // Mark a stack frame as the current frame
    uint32_t
    SetCurrentFrame (lldb_private::StackFrame *frame);

    uint32_t
    GetCurrentFrameIndex () const;

    // Mark a stack frame as the current frame using the frame index
    void
    SetCurrentFrameByIndex (uint32_t idx);

    void
    Clear ();

    // After we have determined the number of frames, we can set the count here
    // and have the frame info be generated on demand.
    void
    SetNumFrames(uint32_t count);

    void
    InvalidateFrames (uint32_t start_idx);
protected:

    //------------------------------------------------------------------
    // Classes that inherit from StackFrameList can see and modify these
    //------------------------------------------------------------------
    typedef std::vector<lldb::StackFrameSP> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;

    mutable Mutex m_mutex;
    collection m_frames;
    uint32_t m_current_frame_idx;

private:
    //------------------------------------------------------------------
    // For StackFrameList only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (StackFrameList);
};

} // namespace lldb_private

#endif  // liblldb_StackFrameList_h_
