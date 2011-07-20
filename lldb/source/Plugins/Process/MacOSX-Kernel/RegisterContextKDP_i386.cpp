//===-- RegisterContextKDP_i386.cpp -----------------------------*- C++ -*-===//
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
#include "RegisterContextKDP_i386.h"
#include "ProcessKDP.h"
#include "ThreadKDP.h"

using namespace lldb;
using namespace lldb_private;


RegisterContextKDP_i386::RegisterContextKDP_i386 (ThreadKDP &thread, uint32_t concrete_frame_idx) :
    RegisterContextDarwin_i386 (thread, concrete_frame_idx),
    m_kdp_thread (thread)
{
}

RegisterContextKDP_i386::~RegisterContextKDP_i386()
{
}

int
RegisterContextKDP_i386::DoReadGPR (lldb::tid_t tid, int flavor, GPR &gpr)
{
    Error error;
    if (m_kdp_thread.GetKDPProcess().GetCommunication().SendRequestReadRegisters (tid, GPRRegSet, &gpr, sizeof(gpr), error))
    {
        if (error.Success())
            return 0;
    }
    return -1;
}

int
RegisterContextKDP_i386::DoReadFPU (lldb::tid_t tid, int flavor, FPU &fpu)
{
    Error error;
    if (m_kdp_thread.GetKDPProcess().GetCommunication().SendRequestReadRegisters (tid, FPURegSet, &fpu, sizeof(fpu), error))
    {
        if (error.Success())
            return 0;
    }
    return -1;
}

int
RegisterContextKDP_i386::DoReadEXC (lldb::tid_t tid, int flavor, EXC &exc)
{
    Error error;
    if (m_kdp_thread.GetKDPProcess().GetCommunication().SendRequestReadRegisters (tid, EXCRegSet, &exc, sizeof(exc), error))
    {
        if (error.Success())
            return 0;
    }
    return -1;
}

int
RegisterContextKDP_i386::DoWriteGPR (lldb::tid_t tid, int flavor, const GPR &gpr)
{
    return ::thread_set_state(tid, flavor, (thread_state_t)&gpr, GPRWordCount);
}

int
RegisterContextKDP_i386::DoWriteFPU (lldb::tid_t tid, int flavor, const FPU &fpu)
{
    return ::thread_set_state(tid, flavor, (thread_state_t)&fpu, FPUWordCount);
}

int
RegisterContextKDP_i386::DoWriteEXC (lldb::tid_t tid, int flavor, const EXC &exc)
{
    return ::thread_set_state(tid, flavor, (thread_state_t)&exc, EXCWordCount);
}


