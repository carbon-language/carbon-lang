//===-- ClangExternalASTSourceCommon.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/ClangExternalASTSourceCommon.h"

using namespace lldb_private;

#define ClangExternalASTSourceCommon_MAGIC  (0x00112233aabbccddull)

ClangExternalASTSourceCommon::ClangExternalASTSourceCommon() : clang::ExternalASTSource()
{
    m_magic = ClangExternalASTSourceCommon_MAGIC;
}

uint64_t ClangExternalASTSourceCommon::GetMetadata (uintptr_t object)
{
    assert (m_magic == ClangExternalASTSourceCommon_MAGIC);
    
    return m_metadata[object];
}

void ClangExternalASTSourceCommon::SetMetadata (uintptr_t object, uint64_t metadata)
{
    assert (m_magic == ClangExternalASTSourceCommon_MAGIC);
    
    m_metadata[object] = metadata;
}

bool ClangExternalASTSourceCommon::HasMetadata (uintptr_t object)
{
    assert (m_magic == ClangExternalASTSourceCommon_MAGIC);

    return m_metadata.find(object) != m_metadata.end();
}