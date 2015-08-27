//===-- CPlusPlusLanguage.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CPlusPlusLanguage.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;

void
CPlusPlusLanguage::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "C++ Language",
                                   CreateInstance);
}

void
CPlusPlusLanguage::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

lldb_private::ConstString
CPlusPlusLanguage::GetPluginNameStatic()
{
    static ConstString g_name("cplusplus");
    return g_name;
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
CPlusPlusLanguage::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
CPlusPlusLanguage::GetPluginVersion()
{
    return 1;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
Language *
CPlusPlusLanguage::CreateInstance (lldb::LanguageType language)
{
    switch (language)
    {
        case lldb::eLanguageTypeC_plus_plus:
        case lldb::eLanguageTypeC_plus_plus_03:
        case lldb::eLanguageTypeC_plus_plus_11:
        case lldb::eLanguageTypeC_plus_plus_14:
            return new CPlusPlusLanguage();
        default:
            return nullptr;
    }
}
