//===-- ItaniumABILanguageRuntime.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ItaniumABILanguageRuntime.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

static const char *pluginName = "ItaniumABILanguageRuntime";
static const char *pluginDesc = "Itanium ABI for the C++ language";
static const char *pluginShort = "language.itanium";

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
lldb_private::LanguageRuntime *
ItaniumABILanguageRuntime::CreateInstance (Process *process, lldb::LanguageType language)
{
    // FIXME: We have to check the process and make sure we actually know that this process supports
    // the Itanium ABI.
    if (language == eLanguageTypeC_plus_plus)
        return new ItaniumABILanguageRuntime (process);
    else
        return NULL;
}

void
ItaniumABILanguageRuntime::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);    
}

void
ItaniumABILanguageRuntime::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ItaniumABILanguageRuntime::GetPluginName()
{
    return pluginName;
}

const char *
ItaniumABILanguageRuntime::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
ItaniumABILanguageRuntime::GetPluginVersion()
{
    return 1;
}

void
ItaniumABILanguageRuntime::GetPluginCommandHelp (const char *command, Stream *strm)
{
}

Error
ItaniumABILanguageRuntime::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in command are currently supported.");
    return error;
}

Log *
ItaniumABILanguageRuntime::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}

bool
ItaniumABILanguageRuntime::IsVTableName (const char *name)
{
    if (name == NULL)
        return false;
        
    // Can we maybe ask Clang about this?
    if (strstr (name, "_vptr$") == name)
        return true;
    else
        return false;
}
