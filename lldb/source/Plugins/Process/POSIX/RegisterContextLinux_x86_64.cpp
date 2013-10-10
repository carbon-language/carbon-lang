//===-- RegisterContextLinux_x86_64.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include <vector>
#include "RegisterContextPOSIX_x86.h"
#include "RegisterContextLinux_i386.h"
#include "RegisterContextLinux_x86_64.h"

using namespace lldb_private;
using namespace lldb;

typedef struct _GPR
{
    uint64_t r15;
    uint64_t r14;
    uint64_t r13;
    uint64_t r12;
    uint64_t rbp;
    uint64_t rbx;
    uint64_t r11;
    uint64_t r10;
    uint64_t r9;
    uint64_t r8;
    uint64_t rax;
    uint64_t rcx;
    uint64_t rdx;
    uint64_t rsi;
    uint64_t rdi;
    uint64_t orig_ax;
    uint64_t rip;
    uint64_t cs;
    uint64_t rflags;
    uint64_t rsp;
    uint64_t ss;
    uint64_t fs_base;
    uint64_t gs_base;
    uint64_t ds;
    uint64_t es;
    uint64_t fs;
    uint64_t gs;
} GPR;

struct UserArea
{
    GPR      gpr;           // General purpose registers.
    int32_t  fpvalid;       // True if FPU is being used.
    int32_t  pad0;
    FXSAVE   i387;          // General purpose floating point registers (see FPR for extended register sets).
    uint64_t tsize;         // Text segment size.
    uint64_t dsize;         // Data segment size.
    uint64_t ssize;         // Stack segment size.
    uint64_t start_code;    // VM address of text.
    uint64_t start_stack;   // VM address of stack bottom (top in rsp).
    int64_t  signal;        // Signal causing core dump.
    int32_t  reserved;      // Unused.
    int32_t  pad1;
    uint64_t ar0;           // Location of GPR's.
    FXSAVE*  fpstate;       // Location of FPR's.
    uint64_t magic;         // Identifier for core dumps.
    char     u_comm[32];    // Command causing core dump.
    uint64_t u_debugreg[8]; // Debug registers (DR0 - DR7).
    uint64_t error_code;    // CPU error code.
    uint64_t fault_address; // Control register CR3.
};

#define DR_SIZE sizeof(UserArea::u_debugreg[0])
#define DR_OFFSET(reg_index) \
    (LLVM_EXTENSION offsetof(UserArea, u_debugreg[reg_index]))

//---------------------------------------------------------------------------
// Include RegisterInfos_x86_64 to declare our g_register_infos_x86_64 structure.
//---------------------------------------------------------------------------
#define DECLARE_REGISTER_INFOS_X86_64_STRUCT
#include "RegisterInfos_x86_64.h"
#undef DECLARE_REGISTER_INFOS_X86_64_STRUCT

static const RegisterInfo *
GetRegisterInfo_i386(const lldb_private::ArchSpec &arch)
{
    static std::vector<lldb_private::RegisterInfo> g_register_infos;

    // Allocate RegisterInfo only once
    if (g_register_infos.empty())
    {
        // Copy the register information from base class
        std::unique_ptr<RegisterContextLinux_i386> reg_interface(new RegisterContextLinux_i386 (arch));
        const RegisterInfo *base_info = reg_interface->GetRegisterInfo();
        g_register_infos.insert(g_register_infos.end(), &base_info[0], &base_info[k_num_registers_i386]);

        //---------------------------------------------------------------------------
        // Include RegisterInfos_x86_64 to update the g_register_infos structure
        //  with x86_64 offsets.
        //---------------------------------------------------------------------------
        #define UPDATE_REGISTER_INFOS_I386_STRUCT_WITH_X86_64_OFFSETS
        #include "RegisterInfos_x86_64.h"
        #undef UPDATE_REGISTER_INFOS_I386_STRUCT_WITH_X86_64_OFFSETS
    }

    return &g_register_infos[0];
}

RegisterContextLinux_x86_64::RegisterContextLinux_x86_64(const ArchSpec &target_arch) :
    RegisterInfoInterface(target_arch)
{
}

RegisterContextLinux_x86_64::~RegisterContextLinux_x86_64()
{
}

size_t
RegisterContextLinux_x86_64::GetGPRSize()
{
    return sizeof(GPR);
}

const RegisterInfo *
RegisterContextLinux_x86_64::GetRegisterInfo()
{
    switch (m_target_arch.GetCore())
    {
        case ArchSpec::eCore_x86_32_i386:
        case ArchSpec::eCore_x86_32_i486:
        case ArchSpec::eCore_x86_32_i486sx:
            return GetRegisterInfo_i386 (m_target_arch);
        case ArchSpec::eCore_x86_64_x86_64:
            return g_register_infos_x86_64;
        default:
            assert(false && "Unhandled target architecture.");
            return NULL;
    }
}

