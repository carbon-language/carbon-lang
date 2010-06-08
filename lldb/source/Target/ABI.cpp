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

ABI*
ABI::FindPlugin (const ConstString &triple)
{
    std::auto_ptr<ABI> abi_ap;
    ABICreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetABICreateCallbackAtIndex(idx)) != NULL;
         ++idx)
    {
        abi_ap.reset (create_callback(triple));

        if (abi_ap.get())
            return abi_ap.release();
    }

    return NULL;
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
