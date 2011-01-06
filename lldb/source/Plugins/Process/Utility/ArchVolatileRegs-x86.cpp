//===-- ArchVolatileRegs-x86.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ArchVolatileRegs-x86.h"
#include "llvm/Support/MachO.h"
#include "lldb/lldb-private.h"
#include "lldb/Utility/ArchVolatileRegs.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/RegisterContext.h"
#include <set>

using namespace lldb;
using namespace lldb_private;

bool
ArchVolatileRegs_x86::RegisterIsVolatile (Thread& thread, uint32_t regnum)
{
    initialize_regset (thread);
    if (m_non_volatile_regs.find (regnum) == m_non_volatile_regs.end())
        return true;
    else
        return false;
}

lldb_private::ArchVolatileRegs *
ArchVolatileRegs_x86::CreateInstance (const lldb_private::ArchSpec &arch)
{
   uint32_t cpu = arch.GetCPUType ();
   if (cpu != llvm::MachO::CPUTypeX86_64 && cpu != llvm::MachO::CPUTypeI386)
       return NULL;

   return new ArchVolatileRegs_x86 (cpu);
}

ArchVolatileRegs_x86::ArchVolatileRegs_x86(int cpu) :
                lldb_private::ArchVolatileRegs(), 
                m_cpu(cpu), 
                m_non_volatile_regs()
{
}

void

ArchVolatileRegs_x86::initialize_regset(Thread& thread)
{
    if (m_non_volatile_regs.size() > 0)
        return;

   
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    const RegisterInfo *ri;

    const char *x86_64_regnames[] = { "rbx", 
                                      "rsp", 
                                      "rbp",
                                      "r12", 
                                      "r13",
                                      "r14",
                                      "r15",
                                      "rip" };

    const char *i386_regnames[] = { "ebx",
                                    "ebp",
                                    "esi",
                                    "edi",
                                    "esp",
                                    "eip" };

    
    const char **names;
    int namecount;
    if (m_cpu == llvm::MachO::CPUTypeX86_64)
    {
        names = x86_64_regnames;
        namecount = sizeof (x86_64_regnames) / sizeof (char *);
    }
    else
    {
        names = i386_regnames;
        namecount = sizeof (i386_regnames) / sizeof (char *);
    }

    for (int i = 0; i < namecount; i++)
    {
        ri = reg_ctx->GetRegisterInfoByName (names[i]);
        if (ri)
            m_non_volatile_regs.insert (ri->kinds[eRegisterKindLLDB]);
     }
}


//------------------------------------------------------------------
// PluginInterface protocol in ArchVolatileRegs_x86
//------------------------------------------------------------------

const char *
ArchVolatileRegs_x86::GetPluginName()
{
    return "ArchVolatileRegs_x86";
}

const char *
ArchVolatileRegs_x86::GetShortPluginName()
{
    return "archvolatileregs.x86";
}


uint32_t
ArchVolatileRegs_x86::GetPluginVersion()
{
    return 1;
}

void
ArchVolatileRegs_x86::GetPluginCommandHelp (const char *command, Stream *strm)
{
}

Error
ArchVolatileRegs_x86::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in command are currently supported.");
    return error;
}

Log *
ArchVolatileRegs_x86::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}

void
ArchVolatileRegs_x86::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
ArchVolatileRegs_x86::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
ArchVolatileRegs_x86::GetPluginNameStatic()
{
    return "ArchVolatileRegs_x86";
}

const char *
ArchVolatileRegs_x86::GetPluginDescriptionStatic()
{
    return "i386 and x86_64 architecture volatile register information.";
}
