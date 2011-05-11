//===-- ABI.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ABI.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;

ABISP
ABI::FindPlugin (const ArchSpec &arch)
{
    ABISP abi_sp;
    ABICreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetABICreateCallbackAtIndex(idx)) != NULL;
         ++idx)
    {
        abi_sp = create_callback(arch);

        if (abi_sp)
            return abi_sp;
    }
    abi_sp.reset();
    return abi_sp;
}

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
ABI::ABI()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ABI::~ABI()
{
}
