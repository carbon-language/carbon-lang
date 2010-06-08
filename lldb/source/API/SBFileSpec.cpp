//===-- SBFileSpec.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBFileSpec.h"
#include "lldb/Core/FileSpec.h"

using namespace lldb;
using namespace lldb_private;



SBFileSpec::SBFileSpec () :
    m_lldb_object_ap()
{
}

SBFileSpec::SBFileSpec (const SBFileSpec &rhs) :
    m_lldb_object_ap()
{
    if (rhs.m_lldb_object_ap.get())
        m_lldb_object_ap.reset (new FileSpec (*m_lldb_object_ap));
}

SBFileSpec::SBFileSpec (const char *path) :
    m_lldb_object_ap(new FileSpec (path))
{
}

SBFileSpec::~SBFileSpec ()
{
}

const SBFileSpec &
SBFileSpec::operator = (const SBFileSpec &rhs)
{
    if (this != &rhs)
    {
        if (rhs.IsValid())
            m_lldb_object_ap.reset (new lldb_private::FileSpec(*rhs.m_lldb_object_ap.get()));
    }
    return *this;
}

bool
SBFileSpec::IsValid() const
{
    return m_lldb_object_ap.get() != NULL;
}

bool
SBFileSpec::Exists () const
{
    if (m_lldb_object_ap.get())
        return m_lldb_object_ap->Exists();
    return false;
}


int
SBFileSpec::ResolvePath (const char *src_path, char *dst_path, size_t dst_len)
{
    return lldb_private::FileSpec::Resolve (src_path, dst_path, dst_len);
}

const char *
SBFileSpec::GetFileName() const
{
    if (m_lldb_object_ap.get())
        return m_lldb_object_ap->GetFilename().AsCString();
    return NULL;
}

const char *
SBFileSpec::GetDirectory() const
{
    if (m_lldb_object_ap.get())
        return m_lldb_object_ap->GetDirectory().AsCString();
    return NULL;
}

uint32_t
SBFileSpec::GetPath (char *dst_path, size_t dst_len) const
{
    if (m_lldb_object_ap.get())
        return m_lldb_object_ap->GetPath (dst_path, dst_len);

    if (dst_path && dst_len)
        *dst_path = '\0';
    return 0;
}


const lldb_private::FileSpec *
SBFileSpec::operator->() const
{
    return m_lldb_object_ap.get();
}

const lldb_private::FileSpec *
SBFileSpec::get() const
{
    return m_lldb_object_ap.get();
}


const lldb_private::FileSpec &
SBFileSpec::operator*() const
{
    return *m_lldb_object_ap.get();
}

const lldb_private::FileSpec &
SBFileSpec::ref() const
{
    return *m_lldb_object_ap.get();
}


void
SBFileSpec::SetFileSpec (const lldb_private::FileSpec& fs)
{
    if (m_lldb_object_ap.get())
        *m_lldb_object_ap = fs;
    else
        m_lldb_object_ap.reset (new FileSpec (fs));
}

