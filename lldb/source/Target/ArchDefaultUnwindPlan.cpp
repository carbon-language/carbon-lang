//===-- ArchDefaultUnwindPlan.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"

#include <map>

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/ArchDefaultUnwindPlan.h"

using namespace lldb;
using namespace lldb_private;

ArchDefaultUnwindPlanSP
ArchDefaultUnwindPlan::FindPlugin (const ArchSpec &arch)
{
    ArchDefaultUnwindPlanCreateInstance create_callback;
    typedef std::map <const ArchSpec, ArchDefaultUnwindPlanSP> ArchDefaultUnwindPlanMap;
    static ArchDefaultUnwindPlanMap g_plugin_map;
    static Mutex g_plugin_map_mutex (Mutex::eMutexTypeRecursive);
    Mutex::Locker locker (g_plugin_map_mutex);
    ArchDefaultUnwindPlanMap::iterator pos = g_plugin_map.find (arch);
    if (pos != g_plugin_map.end())
        return pos->second;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetArchDefaultUnwindPlanCreateCallbackAtIndex(idx)) != NULL;
         ++idx)
    {
        ArchDefaultUnwindPlanSP default_unwind_plan_sp (create_callback (arch));
        if (default_unwind_plan_sp)
        {
            g_plugin_map[arch] = default_unwind_plan_sp;
            return default_unwind_plan_sp;
        }
    }
    return ArchDefaultUnwindPlanSP();
}

ArchDefaultUnwindPlan::ArchDefaultUnwindPlan ()
{
}

ArchDefaultUnwindPlan::~ArchDefaultUnwindPlan ()
{
}
