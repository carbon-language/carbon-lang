//===-- FormatManager.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/FormatManager.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

bool FormatManager::GetFormatForType (const ConstString &type, lldb::Format& format, bool& cascade)
{
    Mutex::Locker locker (m_format_map_mutex);
    FormatMap& fmtmap = m_format_map;
    FormatMap::iterator iter = fmtmap.find(type.GetCString());
    if(iter == fmtmap.end())
        return false;
    else {
        format = iter->second.FormatStyle;
        cascade = iter->second.Cascades;
        return true;
    }
}

void FormatManager::AddFormatForType (const ConstString &type, lldb::Format format, bool cascade)
{
    format_entry_t entry(format, cascade);
    Mutex::Locker locker (m_format_map_mutex);
    FormatMap& fmtmap = m_format_map;
    fmtmap[type.GetCString()] = entry;
}

bool FormatManager::DeleteFormatForType (const ConstString &type)
{
    Mutex::Locker locker (m_format_map_mutex);
    FormatMap& fmtmap = m_format_map;
    const char* typeCS = type.GetCString();
    FormatMap::iterator iter = fmtmap.find(typeCS);
    if (iter == fmtmap.end())
        return false;
    else {
        fmtmap.erase(typeCS);
        return true;
    }
}

void FormatManager::LoopThroughFormatList (FormatCallback cback, void* param)
{
    Mutex::Locker locker (m_format_map_mutex);
    FormatMap& fmtmap = m_format_map;
    FormatIterator iter = fmtmap.begin();
    while(iter != fmtmap.end()) {
        const char* type = iter->first;
        lldb::Format format = iter->second.FormatStyle;
        bool cascade = iter->second.Cascades;
        if(!cback(param, type,format, cascade)) break;
        iter++;
    }
}

