//===-- MachThreadContext_x86_64.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MachThreadContext_x86_64_h_
#define liblldb_MachThreadContext_x86_64_h_

#if defined (__i386__) || defined (__x86_64__)

#include "MachThreadContext.h"
#include "RegisterContextMach_x86_64.h"

class ThreadMacOSX;

class MachThreadContext_x86_64 : public MachThreadContext
{
public:
    static MachThreadContext*
    Create(const lldb_private::ArchSpec &arch_spec, ThreadMacOSX &thread);

    // Class init function
    static void
    Initialize();

    // Instance init function
    void
    InitializeInstance();

    MachThreadContext_x86_64 (ThreadMacOSX &thread);

    virtual
    ~MachThreadContext_x86_64();

    virtual lldb::RegisterContextSP
    CreateRegisterContext (lldb_private::StackFrame *frame) const;

    virtual void
    ThreadWillResume ();

    virtual bool
    ShouldStop ();

    virtual void
    RefreshStateAfterStop ();

    virtual bool
    NotifyException (MachException::Data& exc);

    virtual size_t
    GetStackFrameData (lldb_private::StackFrame *first_frame, std::vector<std::pair<lldb::addr_t, lldb::addr_t> >& fp_pc_pairs);

protected:
//    kern_return_t EnableHardwareSingleStep (bool enable);
    uint32_t m_flags_reg;
private:
    DISALLOW_COPY_AND_ASSIGN (MachThreadContext_x86_64);
};

//#if defined (__x86_64__)
//typedef MachThreadContext_x86_64    DNBArch;
//#endif

#endif    // defined (__i386__) || defined (__x86_64__)
#endif    // #ifndef liblldb_MachThreadContext_x86_64_h_
