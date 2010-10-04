//===-- ArchDefaultUnwindPlan-x86.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ArchDefaultUnwindPlan-x86.h"
#include "llvm/Support/MachO.h"
#include "lldb/lldb-private.h"
#include "lldb/Utility/ArchDefaultUnwindPlan.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/lldb-enumerations.h"

using namespace lldb;
using namespace lldb_private;

lldb_private::UnwindPlan*
ArchDefaultUnwindPlan_x86::GetArchDefaultUnwindPlan (Thread& thread, Address current_pc)
{
    if (m_cpu == llvm::MachO::CPUTypeX86_64)
    {
        return &m_64bit_default;
    }
    if (m_cpu == llvm::MachO::CPUTypeI386)
    {
        return &m_32bit_default;
    }
    return NULL;
}

lldb_private::ArchDefaultUnwindPlan *
ArchDefaultUnwindPlan_x86::CreateInstance (const lldb_private::ArchSpec &arch)
{
   uint32_t cpu = arch.GetCPUType ();
   if (cpu != llvm::MachO::CPUTypeX86_64 && cpu != llvm::MachO::CPUTypeI386)
       return NULL;

   return new ArchDefaultUnwindPlan_x86 (cpu);
}

ArchDefaultUnwindPlan_x86::ArchDefaultUnwindPlan_x86(int cpu) :
                lldb_private::ArchDefaultUnwindPlan(), 
                m_cpu(cpu), 
                m_32bit_default(), 
                m_64bit_default() 
{ 
    UnwindPlan::Row row;
    UnwindPlan::Row::RegisterLocation regloc;

    m_32bit_default.SetRegisterKind (eRegisterKindGeneric);
    row.SetCFARegister (LLDB_REGNUM_GENERIC_FP);
    row.SetCFAOffset (2 * 4);
    row.SetOffset (0);

    regloc.SetAtCFAPlusOffset (2 * -4);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_FP, regloc);
    regloc.SetAtCFAPlusOffset (1 * -4);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_PC, regloc);
    regloc.SetIsCFAPlusOffset (0);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_SP, regloc);

    m_32bit_default.AppendRow (row);

    row.Clear();

    m_64bit_default.SetRegisterKind (eRegisterKindGeneric);
    row.SetCFARegister (LLDB_REGNUM_GENERIC_FP);
    row.SetCFAOffset (2 * 8);
    row.SetOffset (0);

    regloc.SetAtCFAPlusOffset (2 * -8);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_FP, regloc);
    regloc.SetAtCFAPlusOffset (1 * -8);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_PC, regloc);
    regloc.SetIsCFAPlusOffset (0);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_SP, regloc);

    m_64bit_default.AppendRow (row);
}




//------------------------------------------------------------------
// PluginInterface protocol in UnwindAssemblyParser_x86
//------------------------------------------------------------------

const char *
ArchDefaultUnwindPlan_x86::GetPluginName()
{
    return "ArchDefaultUnwindPlan_x86";
}

const char *
ArchDefaultUnwindPlan_x86::GetShortPluginName()
{
    return "archdefaultunwindplan.x86";
}


uint32_t
ArchDefaultUnwindPlan_x86::GetPluginVersion()
{
    return 1;
}

void
ArchDefaultUnwindPlan_x86::GetPluginCommandHelp (const char *command, Stream *strm)
{
}

Error
ArchDefaultUnwindPlan_x86::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in command are currently supported.");
    return error;
}

Log *
ArchDefaultUnwindPlan_x86::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}

void
ArchDefaultUnwindPlan_x86::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
ArchDefaultUnwindPlan_x86::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
ArchDefaultUnwindPlan_x86::GetPluginNameStatic()
{
    return "ArchDefaultUnwindPlan_x86";
}

const char *
ArchDefaultUnwindPlan_x86::GetPluginDescriptionStatic()
{
    return "i386 and x86_64 architecture default unwind plan assembly plugin.";
}
