//===-- LanguageRuntime.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;

LanguageRuntime*
LanguageRuntime::FindPlugin (Process *process, lldb::LanguageType language)
{
    std::auto_ptr<LanguageRuntime> language_runtime_ap;
    LanguageRuntimeCreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetLanguageRuntimeCreateCallbackAtIndex(idx)) != NULL;
         ++idx)
    {
        language_runtime_ap.reset (create_callback(process, language));

        if (language_runtime_ap.get())
            return language_runtime_ap.release();
    }

    return NULL;
}

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
LanguageRuntime::LanguageRuntime(Process *process) :
    m_process (process)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
LanguageRuntime::~LanguageRuntime()
{
}
