//===-- MachThreadContext_arm.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MachThreadContext_arm_h_
#define liblldb_MachThreadContext_arm_h_

#include "MachThreadContext.h"
#include "RegisterContextMach_arm.h"

class ThreadMacOSX;

class MachThreadContext_arm : public MachThreadContext
{
public:
    enum { kMaxNumThumbITBreakpoints = 4 };

    static MachThreadContext*
    Create (const lldb_private::ArchSpec &arch_spec, ThreadMacOSX &thread);

    static void
    Initialize();

    MachThreadContext_arm(ThreadMacOSX &thread);

    virtual
    ~MachThreadContext_arm();

    virtual lldb::RegisterContextSP
    CreateRegisterContext (lldb_private::StackFrame *frame) const;

    virtual void
    InitializeInstance();

    virtual void
    ThreadWillResume ();

    virtual bool
    ShouldStop ();

    virtual void
    RefreshStateAfterStop ();

protected:
    kern_return_t
    EnableHardwareSingleStep (bool enable);

protected:
    lldb::addr_t m_hw_single_chained_step_addr;
    uint32_t m_bvr0_reg;
    uint32_t m_bcr0_reg;
    uint32_t m_bvr0_save;
    uint32_t m_bcr0_save;
};
#endif  // #ifndef liblldb_MachThreadContext_arm_h_
