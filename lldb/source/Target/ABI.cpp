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


bool
ABI::GetRegisterInfoByName (const ConstString &name, RegisterInfo &info)
{
    uint32_t count = 0;
    const RegisterInfo *register_info_array = GetRegisterInfoArray (count);
    if (register_info_array)
    {
        const char *unique_name_cstr = name.GetCString();
        uint32_t i;
        for (i=0; i<count; ++i)
        {
            if (register_info_array[i].name == unique_name_cstr)
            {
                info = register_info_array[i];
                return true;
            }
        }
        for (i=0; i<count; ++i)
        {
            if (register_info_array[i].alt_name == unique_name_cstr)
            {
                info = register_info_array[i];
                return true;
            }
        }
    }
    return false;
}

bool
ABI::GetRegisterInfoByKind (RegisterKind reg_kind, uint32_t reg_num, RegisterInfo &info)
{
    if (reg_kind < eRegisterKindGCC || reg_kind >= kNumRegisterKinds)
        return false;
        
    uint32_t count = 0;
    const RegisterInfo *register_info_array = GetRegisterInfoArray (count);
    if (register_info_array)
    {
        for (uint32_t i=0; i<count; ++i)
        {
            if (register_info_array[i].kinds[reg_kind] == reg_num)
            {
                info = register_info_array[i];
                return true;
            }
        }
    }
    return false;
}
