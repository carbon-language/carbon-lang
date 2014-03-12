//===-- RegisterContextFreeBSD_i386.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "RegisterContextPOSIX_x86.h"
#include "RegisterContextFreeBSD_i386.h"

using namespace lldb_private;
using namespace lldb;

// http://svnweb.freebsd.org/base/head/sys/x86/include/reg.h
struct GPR
{
    uint32_t fs;
    uint32_t es;
    uint32_t ds;
    uint32_t edi;
    uint32_t esi;
    uint32_t ebp;
    uint32_t isp;
    uint32_t ebx;
    uint32_t edx;
    uint32_t ecx;
    uint32_t eax;
    uint32_t trapno;
    uint32_t err;
    uint32_t eip;
    uint32_t cs;
    uint32_t eflags;
    uint32_t esp;
    uint32_t ss;
    uint32_t gs;
};

struct dbreg {
    uint32_t  dr[8];     /* debug registers */
                         /* Index 0-3: debug address registers */
                         /* Index 4-5: reserved */
                         /* Index 6: debug status */
                         /* Index 7: debug control */
};


#define DR_SIZE sizeof(uint32_t)
#define DR_OFFSET(reg_index) \
    (LLVM_EXTENSION offsetof(dbreg, dr[reg_index]))

//---------------------------------------------------------------------------
// Include RegisterInfos_i386 to declare our g_register_infos_i386 structure.
//---------------------------------------------------------------------------
#define DECLARE_REGISTER_INFOS_I386_STRUCT
#include "RegisterInfos_i386.h"
#undef DECLARE_REGISTER_INFOS_I386_STRUCT

RegisterContextFreeBSD_i386::RegisterContextFreeBSD_i386(const ArchSpec &target_arch) :
    RegisterInfoInterface(target_arch)
{
}

RegisterContextFreeBSD_i386::~RegisterContextFreeBSD_i386()
{
}

size_t
RegisterContextFreeBSD_i386::GetGPRSize()
{
    return sizeof(GPR);
}

const RegisterInfo *
RegisterContextFreeBSD_i386::GetRegisterInfo()
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
