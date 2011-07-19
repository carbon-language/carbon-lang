//===-- RegisterContextKDP_x86_64.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// C Includes
#include <mach/thread_act.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "RegisterContextKDP_x86_64.h"

using namespace lldb;
using namespace lldb_private;


RegisterContextKDP_x86_64::RegisterContextKDP_x86_64(Thread &thread, uint32_t concrete_frame_idx) :
    RegisterContextDarwin_x86_64 (thread, concrete_frame_idx)
{
}

RegisterContextKDP_x86_64::~RegisterContextKDP_x86_64()
{
}

int
RegisterContextKDP_x86_64::DoReadGPR (lldb::tid_t tid, int flavor, GPR &gpr)
{
    return -1;
}

int
RegisterContextKDP_x86_64::DoReadFPU (lldb::tid_t tid, int flavor, FPU &fpu)
{
    return -1;
}

int
RegisterContextKDP_x86_64::DoReadEXC (lldb::tid_t tid, int flavor, EXC &exc)
{
    return -1;
}

int
RegisterContextKDP_x86_64::DoWriteGPR (lldb::tid_t tid, int flavor, const GPR &gpr)
{
    return -1;
}

int
RegisterContextKDP_x86_64::DoWriteFPU (lldb::tid_t tid, int flavor, const FPU &fpu)
{
    return -1;
}

int
RegisterContextKDP_x86_64::DoWriteEXC (lldb::tid_t tid, int flavor, const EXC &exc)
{
    return -1;
}
