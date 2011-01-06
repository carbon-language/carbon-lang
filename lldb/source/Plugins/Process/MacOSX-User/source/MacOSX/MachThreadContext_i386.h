//===-- MachThreadContext_i386.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MachThreadContext_i386_h_
#define liblldb_MachThreadContext_i386_h_

#if defined (__i386__) || defined (__x86_64__)

#include "MachThreadContext.h"
#include "RegisterContextMach_i386.h"

class ThreadMacOSX;

class MachThreadContext_i386 : public MachThreadContext
{
public:
    static MachThreadContext* Create(const lldb_private::ArchSpec &arch_spec, ThreadMacOSX &thread);

    // Class init function
    static void Initialize();

    MachThreadContext_i386(ThreadMacOSX &thread);

    virtual
    ~MachThreadContext_i386();

    virtual lldb::RegisterContextSP
    CreateRegisterContext (lldb_private::StackFrame *frame) const;

    virtual void            InitializeInstance();
    virtual void            ThreadWillResume();
    virtual bool            ShouldStop ();
    virtual void            RefreshStateAfterStop();

    virtual bool            NotifyException(MachException::Data& exc);
    virtual size_t          GetStackFrameData(lldb_private::StackFrame *first_frame, std::vector<std::pair<lldb::addr_t, lldb::addr_t> >& fp_pc_pairs);

protected:
//    kern_return_t EnableHardwareSingleStep (bool enable);
    uint32_t m_flags_reg;
private:
    DISALLOW_COPY_AND_ASSIGN (MachThreadContext_i386);
};

//#if defined (__i386__)
//typedef MachThreadContext_i386    DNBArch;
//#endif

#endif    // defined (__i386__) || defined (__x86_64__)
#endif    // #ifndef liblldb_MachThreadContext_i386_h_
