//===-- DynamicLoaderWindowsDYLD.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DynamicLoaderWindowsDYLD.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/Triple.h"

using namespace lldb;
using namespace lldb_private;

DynamicLoaderWindowsDYLD::DynamicLoaderWindowsDYLD(Process *process)
    : DynamicLoader(process)
{
}

DynamicLoaderWindowsDYLD::~DynamicLoaderWindowsDYLD()
{
}

void
DynamicLoaderWindowsDYLD::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance);
}

void
DynamicLoaderWindowsDYLD::Terminate()
{
}

ConstString
DynamicLoaderWindowsDYLD::GetPluginNameStatic()
{
    static ConstString g_plugin_name("windows-dyld");
    return g_plugin_name;
}

const char *
DynamicLoaderWindowsDYLD::GetPluginDescriptionStatic()
{
    return "Dynamic loader plug-in that watches for shared library "
           "loads/unloads in Windows processes.";
}

DynamicLoader *
DynamicLoaderWindowsDYLD::CreateInstance(Process *process, bool force)
{
    bool should_create = force;
    if (!should_create)
    {
        const llvm::Triple &triple_ref = process->GetTarget().GetArchitecture().GetTriple();
        if (triple_ref.getOS() == llvm::Triple::Win32)
            should_create = true;
    }

    if (should_create)
        return new DynamicLoaderWindowsDYLD(process);

    return nullptr;
}

void
DynamicLoaderWindowsDYLD::DidAttach()
{
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
    if (log)
        log->Printf("DynamicLoaderWindowsDYLD::%s()", __FUNCTION__);

    DidLaunch();

    m_process->LoadModules();
}

void
DynamicLoaderWindowsDYLD::DidLaunch()
{
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
    if (log)
        log->Printf("DynamicLoaderWindowsDYLD::%s()", __FUNCTION__);

    ModuleSP executable = GetTargetExecutable();

    if (!executable.get())
        return;

    ModuleList module_list;
    module_list.Append(executable);
    // FIXME: We probably should not always use 0 as the load address
    // here. Testing showed that when debugging a process that we start
    // ourselves, there's no randomization of the load address of the
    // main module, therefore an offset of 0 will be valid.
    // If we attach to an already running process, this is probably
    // going to be wrong and we'll have to get the load address somehow.
    UpdateLoadedSections(executable, LLDB_INVALID_ADDRESS, 0);

    m_process->GetTarget().ModulesDidLoad(module_list);
}

Error
DynamicLoaderWindowsDYLD::CanLoadImage()
{
    return Error();
}

ConstString
DynamicLoaderWindowsDYLD::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
DynamicLoaderWindowsDYLD::GetPluginVersion()
{
    return 1;
}

ThreadPlanSP
DynamicLoaderWindowsDYLD::GetStepThroughTrampolinePlan(Thread &thread, bool stop)
{
    return ThreadPlanSP();
}
