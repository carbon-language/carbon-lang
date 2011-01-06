//===-- MachThreadContext.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MachThreadContext_h_
#define liblldb_MachThreadContext_h_

#include <vector>

#include "MachException.h"

class ThreadMacOSX;

class MachThreadContext
{
public:
    MachThreadContext (ThreadMacOSX &thread) :
        m_thread (thread)
    {
    }

    virtual ~MachThreadContext()
    {
    }

    virtual lldb::RegisterContextSP
    CreateRegisterContext (lldb_private::StackFrame *frame) const = 0;

    virtual void            InitializeInstance() = 0;
    virtual void            ThreadWillResume () = 0;
    virtual bool            ShouldStop () = 0;
    virtual void            RefreshStateAfterStop() = 0;
    virtual bool            NotifyException (MachException::Data& exc) { return false; }
    virtual bool            StepNotComplete () { return false; }
    virtual size_t          GetStackFrameData(lldb_private::StackFrame *frame, std::vector<std::pair<lldb::addr_t, lldb::addr_t> >& fp_pc_pairs) { return 0; }
//  virtual const uint8_t * SoftwareBreakpointOpcode (size_t byte_size) = 0;

protected:
    ThreadMacOSX &m_thread;

};

#endif  // #ifndef liblldb_MachThreadContext_h_
