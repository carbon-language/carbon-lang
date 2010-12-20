//===-- UnwindAssemblyProfiler.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Utility/UnwindAssemblyProfiler.h"

using namespace lldb;
using namespace lldb_private;

UnwindAssemblyProfiler*
UnwindAssemblyProfiler::FindPlugin (const ArchSpec &arch)
{
    UnwindAssemblyProfilerCreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetUnwindAssemblyProfilerCreateCallbackAtIndex(idx)) != NULL;
         ++idx)
    {
        std::auto_ptr<UnwindAssemblyProfiler> assembly_profiler_ap (create_callback (arch));
        if (assembly_profiler_ap.get ())
            return assembly_profiler_ap.release ();
    }
    return NULL;
}

UnwindAssemblyProfiler::UnwindAssemblyProfiler ()
{
}

UnwindAssemblyProfiler::~UnwindAssemblyProfiler ()
{
}
