//===-- SBSourceManager.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/API/SBSourceManager.h"

#include "lldb/API/SBFileSpec.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"


using namespace lldb;
using namespace lldb_private;


SBSourceManager::SBSourceManager (SourceManager& source_manager) :
    m_source_manager (source_manager)
{
}

SBSourceManager::~SBSourceManager()
{
}

size_t
SBSourceManager::DisplaySourceLinesWithLineNumbers
(
    const SBFileSpec &file,
    uint32_t line,
    uint32_t context_before,
    uint32_t context_after,
    const char* current_line_cstr,
    FILE *f
)
{
    if (f == NULL)
        return 0;

    if (file.IsValid())
    {
        StreamFile str (f);


        return m_source_manager.DisplaySourceLinesWithLineNumbers (*file,
                                                                   line,
                                                                   context_before,
                                                                   context_after,
                                                                   current_line_cstr,
                                                                   &str);
    }
    return 0;
}

SourceManager &
SBSourceManager::GetLLDBManager ()
{
    return m_source_manager;
}
