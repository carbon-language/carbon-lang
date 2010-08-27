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
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    StackFrameList (Thread &thread, 
                    StackFrameList *prev_frames, 
                    bool show_inline_frames);

    ~StackFrameList();

    uint32_t
    GetNumFrames();

    lldb::StackFrameSP
    GetFrameAtIndex (uint32_t idx);

    // Mark a stack frame as the current frame
    uint32_t
    SetSelectedFrame (lldb_private::StackFrame *frame);

    uint32_t
    GetSelectedFrameIndex () const;

    // Mark a stack frame as the current frame using the frame index
    void
    SetSelectedFrameByIndex (uint32_t idx);

    void
    Clear ();

    void
    InvalidateFrames (uint32_t start_idx);
    
    void
    Dump (Stream *s);

protected:

    bool
    SetFrameAtIndex (uint32_t idx, lldb::StackFrameSP &frame_sp);

    //------------------------------------------------------------------
    // Classes that inherit from StackFrameList can see and modify these
    //------------------------------------------------------------------
    typedef std::vector<lldb::StackFrameSP> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;

    Thread &m_thread;
    std::auto_ptr<StackFrameList> m_prev_frames_ap;
    mutable Mutex m_mutex;
    collection m_frames;
    uint32_t m_selected_frame_idx;
    bool m_show_inlined_frames;

private:
    //------------------------------------------------------------------
    // For StackFrameList only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (StackFrameList);
};

} // namespace lldb_private

#endif  // liblldb_StackFrameList_h_
