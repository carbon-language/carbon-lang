//===-- RegisterContextLinux_i386.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "RegisterContextPOSIX_x86.h"
#include "RegisterContextLinux_i386.h"

using namespace lldb_private;
using namespace lldb;

struct GPR
{
    uint32_t ebx;
    uint32_t ecx;
    uint32_t edx;
    uint32_t esi;
    uint32_t edi;
    uint32_t ebp;
    uint32_t eax;
    uint32_t ds;
    uint32_t es;
    uint32_t fs;
    uint32_t gs;
    uint32_t orig_ax;
    uint32_t eip;
    uint32_t cs;
    uint32_t eflags;
    uint32_t esp;
    uint32_t ss;
};

struct UserArea
{
    GPR      regs;          // General purpose registers.
    int32_t  fpvalid;       // True if FPU is being used.
    FXSAVE   i387;          // FPU registers.
    uint32_t tsize;         // Text segment size.
    uint32_t dsize;         // Data segment size.
    uint32_t ssize;         // Stack segment size.
    uint32_t start_code;    // VM address of text.
    uint32_t start_stack;   // VM address of stack bottom (top in rsp).
    int32_t  signal;        // Signal causing core dump.
    int32_t  reserved;      // Unused.
    uint32_t ar0;           // Location of GPR's.
    uint32_t fpstate;       // Location of FPR's. Should be a FXSTATE *, but this
	                        //  has to be 32-bits even on 64-bit systems.
    uint32_t magic;         // Identifier for core dumps.
    char     u_comm[32];    // Command causing core dump.
    uint32_t u_debugreg[8]; // Debug registers (DR0 - DR7).
};

#define DR_SIZE sizeof(UserArea::u_debugreg[0])
#define DR_OFFSET(reg_index) \
    (LLVM_EXTENSION offsetof(UserArea, u_debugreg[reg_index]))

//---------------------------------------------------------------------------
// Include RegisterInfos_i386 to declare our g_register_infos_i386 structure.
//---------------------------------------------------------------------------
#define DECLARE_REGISTER_INFOS_I386_STRUCT
#include "RegisterInfos_i386.h"
#undef DECLARE_REGISTER_INFOS_I386_STRUCT

RegisterContextLinux_i386::RegisterContextLinux_i386(const ArchSpec &target_arch) :
    RegisterInfoInterface(target_arch)
{
}

RegisterContextLinux_i386::~RegisterContextLinux_i386()
{
}

size_t
RegisterContextLinux_i386::GetGPRSize()
{
    return sizeof(GPR);
}

const RegisterInfo *
RegisterContextLinux_i386::GetRegisterInfo()
{
    switch (m_target_arch.GetCore())
    {
        case ArchSpec::eCore_x86_32_i386:
        case ArchSpec::eCore_x86_32_i486:
        case ArchSpec::eCore_x86_32_i486sx:
            return g_register_infos_i386;
        default:
            assert(false && "Unhandled target architecture.");
            return NULL;
    }
}
