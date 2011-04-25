//===-- UnwindAssemblyInstEmulation.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnwindAssemblyInstEmulation.h"

#include "llvm-c/EnhancedDisassembly.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnwindAssemblyProfiler.h"

using namespace lldb;
using namespace lldb_private;



//-----------------------------------------------------------------------------------------------
//  UnwindAssemblyParser_x86 method definitions 
//-----------------------------------------------------------------------------------------------

bool
UnwindAssemblyInstEmulation::GetNonCallSiteUnwindPlanFromAssembly (AddressRange& func, Thread& thread, UnwindPlan& unwind_plan)
{
    return false;
}

bool
UnwindAssemblyInstEmulation::GetFastUnwindPlan (AddressRange& func, Thread& thread, UnwindPlan &unwind_plan)
{
    return false;
}

bool
UnwindAssemblyInstEmulation::FirstNonPrologueInsn (AddressRange& func, Target& target, Thread* thread, Address& first_non_prologue_insn)
{
    return false;
}

UnwindAssemblyProfiler *
UnwindAssemblyInstEmulation::CreateInstance (const ArchSpec &arch)
{
    return NULL;
}


//------------------------------------------------------------------
// PluginInterface protocol in UnwindAssemblyParser_x86
//------------------------------------------------------------------

const char *
UnwindAssemblyInstEmulation::GetPluginName()
{
    return "UnwindAssemblyInstEmulation";
}

const char *
UnwindAssemblyInstEmulation::GetShortPluginName()
{
    return "unwindassembly.inst-emulation";
}


uint32_t
UnwindAssemblyInstEmulation::GetPluginVersion()
{
    return 1;
}

void
UnwindAssemblyInstEmulation::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
UnwindAssemblyInstEmulation::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
UnwindAssemblyInstEmulation::GetPluginNameStatic()
{
    return "UnwindAssemblyInstEmulation";
}

const char *
UnwindAssemblyInstEmulation::GetPluginDescriptionStatic()
{
    return "Instruction emulation based unwind information.";
}
