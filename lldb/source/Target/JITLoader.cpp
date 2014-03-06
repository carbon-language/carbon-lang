//===-- JITLoader.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/Target/JITLoader.h"
#include "lldb/Target/JITLoaderList.h"
#include "lldb/Target/Process.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;

void
JITLoader::LoadPlugins (Process *process, JITLoaderList &list)
{
    JITLoaderCreateInstance create_callback = NULL;
    for (uint32_t idx = 0; (create_callback = PluginManager::GetJITLoaderCreateCallbackAtIndex(idx)) != NULL; ++idx)
    {
        JITLoaderSP instance_sp(create_callback(process, false));
        if (instance_sp)
            list.Append(std::move(instance_sp));
    }
}

JITLoader::JITLoader(Process *process) :
    m_process (process)
{
}

JITLoader::~JITLoader()
{
}
