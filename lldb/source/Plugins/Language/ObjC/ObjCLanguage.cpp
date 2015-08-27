//===-- ObjCLanguage.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ObjCLanguage.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;

void
ObjCLanguage::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "Objective-C Language",
                                   CreateInstance);
}

void
ObjCLanguage::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

lldb_private::ConstString
ObjCLanguage::GetPluginNameStatic()
{
    static ConstString g_name("objc");
    return g_name;
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
ObjCLanguage::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ObjCLanguage::GetPluginVersion()
{
    return 1;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
Language *
ObjCLanguage::CreateInstance (lldb::LanguageType language)
{
    switch (language)
    {
        case lldb::eLanguageTypeObjC:
            return new ObjCLanguage();
        default:
            return nullptr;
    }
}
