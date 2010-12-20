//===-- ArchDefaultUnwindPlan.cpp -------------------------------*- C++ -*-===//
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
#include "lldb/Utility/ArchDefaultUnwindPlan.h"

using namespace lldb;
using namespace lldb_private;

ArchDefaultUnwindPlan*
ArchDefaultUnwindPlan::FindPlugin (const ArchSpec &arch)
{
    ArchDefaultUnwindPlanCreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetArchDefaultUnwindPlanCreateCallbackAtIndex(idx)) != NULL;
         ++idx)
    {
        std::auto_ptr<ArchDefaultUnwindPlan> default_unwind_plan_ap (create_callback (arch));
        if (default_unwind_plan_ap.get ())
            return default_unwind_plan_ap.release ();
    }
    return NULL;
}

ArchDefaultUnwindPlan::ArchDefaultUnwindPlan ()
{
}

ArchDefaultUnwindPlan::~ArchDefaultUnwindPlan ()
{
}
