//===-- DynamicLoaderWindows.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DynamicLoaderWindows.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/Triple.h"

using namespace lldb;
using namespace lldb_private;

DynamicLoaderWindows::DynamicLoaderWindows(Process *process)
    : DynamicLoader(process)
{

}

DynamicLoaderWindows::~DynamicLoaderWindows()
{
}

void DynamicLoaderWindows::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  CreateInstance);
}

void DynamicLoaderWindows::Terminate()
{

}

ConstString DynamicLoaderWindows::GetPluginNameStatic()
{
    static ConstString g_plugin_name("windows-dyld");
    return g_plugin_name;
}

const char *DynamicLoaderWindows::GetPluginDescriptionStatic()
{
    return "Dynamic loader plug-in that watches for shared library "
           "loads/unloads in Windows processes.";
}


DynamicLoader *DynamicLoaderWindows::CreateInstance(Process *process, bool force)
{
    bool should_create = force;
    if (!should_create)
    {
        const llvm::Triple &triple_ref = process->GetTarget().GetArchitecture().GetTriple();
        if (triple_ref.getOS() == llvm::Triple::Win32)
            should_create = true;
    }

    if (should_create)
        return new DynamicLoaderWindows (process);
    return nullptr;
}

void DynamicLoaderWindows::DidAttach()
{

}

void DynamicLoaderWindows::DidLaunch()
{

}

Error DynamicLoaderWindows::CanLoadImage()
{
    return Error();
}

ConstString DynamicLoaderWindows::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t DynamicLoaderWindows::GetPluginVersion()
{
    return 1;
}

ThreadPlanSP
DynamicLoaderWindows::GetStepThroughTrampolinePlan(Thread &thread, bool stop)
{
    return ThreadPlanSP();
}