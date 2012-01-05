//===-- ProcessFreeBSD.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Target.h"

#include "ProcessFreeBSD.h"
#include "ProcessPOSIXLog.h"
#include "Plugins/Process/Utility/InferiorCallPOSIX.h"
#include "ProcessMonitor.h"
#include "POSIXThread.h"

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------------------
// Static functions.

Process*
ProcessFreeBSD::CreateInstance(Target& target, Listener &listener)
{
    return new ProcessFreeBSD(target, listener);
}

void
ProcessFreeBSD::Initialize()
{
    static bool g_initialized = false;

    if (!g_initialized)
    {
        PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                      GetPluginDescriptionStatic(),
                                      CreateInstance);
        Log::Callbacks log_callbacks = {
            ProcessPOSIXLog::DisableLog,
            ProcessPOSIXLog::EnableLog,
            ProcessPOSIXLog::ListLogCategories
        };

        Log::RegisterLogChannel (ProcessFreeBSD::GetPluginNameStatic(), log_callbacks);
        ProcessPOSIXLog::RegisterPluginName(GetPluginNameStatic());
        g_initialized = true;
    }
}

const char *
ProcessFreeBSD::GetPluginNameStatic()
{
    return "freebsd";
}

const char *
ProcessFreeBSD::GetPluginDescriptionStatic()
{
    return "Process plugin for FreeBSD";
}

//------------------------------------------------------------------------------
// ProcessInterface protocol.

const char *
ProcessFreeBSD::GetPluginName()
{
    return "process.freebsd";
}

const char *
ProcessFreeBSD::GetShortPluginName()
{
    return "process.freebsd";
}

uint32_t
ProcessFreeBSD::GetPluginVersion()
{
    return 1;
}

void
ProcessFreeBSD::GetPluginCommandHelp(const char *command, Stream *strm)
{
}

Error
ProcessFreeBSD::ExecutePluginCommand(Args &command, Stream *strm)
{
    return Error(1, eErrorTypeGeneric);
}

Log *
ProcessFreeBSD::EnablePluginLogging(Stream *strm, Args &command)
{
    return NULL;
}

//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessFreeBSD::ProcessFreeBSD(Target& target, Listener &listener)
    : ProcessPOSIX(target, listener)
{
    // FIXME: Putting this code in the ctor and saving the byte order in a
    // member variable is a hack to avoid const qual issues in GetByteOrder.
    ObjectFile *obj_file = GetTarget().GetExecutableModule()->GetObjectFile();
    m_byte_order = obj_file->GetByteOrder();
}

void
ProcessFreeBSD::Terminate()
{
}

uint32_t
ProcessFreeBSD::UpdateThreadList(ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    // XXX haxx
    new_thread_list = old_thread_list;
  
    return 0;
}
