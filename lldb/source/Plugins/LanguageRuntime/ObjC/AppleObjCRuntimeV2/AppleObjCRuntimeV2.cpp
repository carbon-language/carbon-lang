//===-- AppleObjCRuntimeV2.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AppleObjCRuntimeV2.h"

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

static const char *pluginName = "AppleObjCRuntimeV2";
static const char *pluginDesc = "Apple Objective C Language Runtime - Version 2";
static const char *pluginShort = "language.apple.objc.v2";

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
lldb_private::LanguageRuntime *
AppleObjCRuntimeV2::CreateInstance (Process *process, lldb::LanguageType language)
{
    // FIXME: This should be a MacOS or iOS process, and we need to look for the OBJC section to make
    // sure we aren't using the V1 runtime.
    if (language == eLanguageTypeObjC)
        return new AppleObjCRuntimeV2 (process);
    else
        return NULL;
}

void
AppleObjCRuntimeV2::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);    
}

void
AppleObjCRuntimeV2::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
AppleObjCRuntimeV2::GetPluginName()
{
    return pluginName;
}

const char *
AppleObjCRuntimeV2::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
AppleObjCRuntimeV2::GetPluginVersion()
{
    return 1;
}

void
AppleObjCRuntimeV2::GetPluginCommandHelp (const char *command, Stream *strm)
{
}

Error
AppleObjCRuntimeV2::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in command are currently supported.");
    return error;
}

Log *
AppleObjCRuntimeV2::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}
