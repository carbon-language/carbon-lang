//===-- SBSourceManager.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBStream.h"

#include "lldb/API/SBFileSpec.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"


using namespace lldb;
using namespace lldb_private;


SBSourceManager::SBSourceManager (SourceManager* source_manager) :
    m_opaque_ptr (source_manager)
{
}

SBSourceManager::~SBSourceManager()
{
}

SBSourceManager::SBSourceManager(const SBSourceManager &rhs) :
    m_opaque_ptr (rhs.m_opaque_ptr)
{
}

const SBSourceManager &
SBSourceManager::operator = (const SBSourceManager &rhs)
{
    m_opaque_ptr = rhs.m_opaque_ptr;
    return *this;
}

size_t
SBSourceManager::DisplaySourceLinesWithLineNumbers
(
    const SBFileSpec &file,
    uint32_t line,
    uint32_t context_before,
    uint32_t context_after,
    const char* current_line_cstr,
    SBStream &s
)
{
    if (m_opaque_ptr == NULL)
        return 0;

    if (s.m_opaque_ap.get() == NULL)
        return 0;

    if (file.IsValid())
    {
        return m_opaque_ptr->DisplaySourceLinesWithLineNumbers (*file,
                                                                line,
                                                                context_before,
                                                                context_after,
                                                                current_line_cstr,
                                                                s.m_opaque_ap.get());
    }
    return 0;
}
