//===-- ArchVolatileRegs.cpp ------------------------------------*- C++ -*-===//
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
#include "lldb/Target/ArchVolatileRegs.h"

using namespace lldb;
using namespace lldb_private;

ArchVolatileRegs*
ArchVolatileRegs::FindPlugin (const ArchSpec &arch)
{
    ArchVolatileRegsCreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetArchVolatileRegsCreateCallbackAtIndex(idx)) != NULL;
         ++idx)
    {
        std::auto_ptr<ArchVolatileRegs> default_volatile_regs_ap (create_callback (arch));
        if (default_volatile_regs_ap.get ())
            return default_volatile_regs_ap.release ();
    }
    return NULL;
}

ArchVolatileRegs::ArchVolatileRegs ()
{
}

ArchVolatileRegs::~ArchVolatileRegs ()
{
}
