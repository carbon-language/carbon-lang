//===-- RegisterContextKDP_arm.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RegisterContextKDP_arm.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "ProcessKDP.h"
#include "ThreadKDP.h"

using namespace lldb;
using namespace lldb_private;


RegisterContextKDP_arm::RegisterContextKDP_arm(ThreadKDP &thread, uint32_t concrete_frame_idx) :
    RegisterContextDarwin_arm (thread, concrete_frame_idx),
    m_kdp_thread (thread)
{
}

RegisterContextKDP_arm::~RegisterContextKDP_arm()
{
}

int
RegisterContextKDP_arm::DoReadGPR (lldb::tid_t tid, int flavor, GPR &gpr)
{
    Error error;
    if (m_kdp_thread.GetKDPProcess().GetCommunication().SendRequestReadRegisters (tid, 
                                                                                  GPRRegSet, 
                                                                                  &gpr, sizeof(gpr), 
                                                                                  error))
    {
        if (error.Success())
            return 0;
    }
    return -1;
}

int
RegisterContextKDP_arm::DoReadFPU (lldb::tid_t tid, int flavor, FPU &fpu)
{
    return -1;
}

int
RegisterContextKDP_arm::DoReadEXC (lldb::tid_t tid, int flavor, EXC &exc)
{
    return -1;
}

int
RegisterContextKDP_arm::DoReadDBG (lldb::tid_t tid, int flavor, DBG &dbg)
{
    return -1;
}

int
RegisterContextKDP_arm::DoWriteGPR (lldb::tid_t tid, int flavor, const GPR &gpr)
{
    return -1;
}

int
RegisterContextKDP_arm::DoWriteFPU (lldb::tid_t tid, int flavor, const FPU &fpu)
{
    return -1;
}

int
RegisterContextKDP_arm::DoWriteEXC (lldb::tid_t tid, int flavor, const EXC &exc)
{
    return -1;
}

int
RegisterContextKDP_arm::DoWriteDBG (lldb::tid_t tid, int flavor, const DBG &dbg)
{
    return -1;
}


