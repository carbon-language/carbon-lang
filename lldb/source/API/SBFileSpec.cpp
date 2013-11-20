//===-- SBFileSpec.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <limits.h>

#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBStream.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"

using namespace lldb;
using namespace lldb_private;



SBFileSpec::SBFileSpec () :
    m_opaque_ap(new lldb_private::FileSpec())
{
}

SBFileSpec::SBFileSpec (const SBFileSpec &rhs) :
    m_opaque_ap(new lldb_private::FileSpec(*rhs.m_opaque_ap))
{
}

SBFileSpec::SBFileSpec (const lldb_private::FileSpec& fspec) :
    m_opaque_ap(new lldb_private::FileSpec(fspec))
{
}

// Deprected!!!
SBFileSpec::SBFileSpec (const char *path) :
    m_opaque_ap(new FileSpec (path, true))
{
}

SBFileSpec::SBFileSpec (const char *path, bool resolve) :
    m_opaque_ap(new FileSpec (path, resolve))
{
}

SBFileSpec::~SBFileSpec ()
{
}

const SBFileSpec &
SBFileSpec::operator = (const SBFileSpec &rhs)
{
    if (this != &rhs)
        *m_opaque_ap = *rhs.m_opaque_ap;
    return *this;
}

bool
SBFileSpec::IsValid() const
{
    return m_opaque_ap->operator bool();
}

bool
SBFileSpec::Exists () const
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    bool result = m_opaque_ap->Exists();

    if (log)
        log->Printf ("SBFileSpec(%p)::Exists () => %s", m_opaque_ap.get(), (result ? "true" : "false"));

    return result;
}

bool
SBFileSpec::ResolveExecutableLocation ()
{
    return m_opaque_ap->ResolveExecutableLocation ();
}

int
SBFileSpec::ResolvePath (const char *src_path, char *dst_path, size_t dst_len)
{
    return lldb_private::FileSpec::Resolve (src_path, dst_path, dst_len);
}

const char *
SBFileSpec::GetFilename() const
{
    const char *s = m_opaque_ap->GetFilename().AsCString();

    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (s)
            log->Printf ("SBFileSpec(%p)::GetFilename () => \"%s\"", m_opaque_ap.get(), s);
        else
            log->Printf ("SBFileSpec(%p)::GetFilename () => NULL", m_opaque_ap.get());
    }

    return s;
}

const char *
SBFileSpec::GetDirectory() const
{
    const char *s = m_opaque_ap->GetDirectory().AsCString();
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (s)
            log->Printf ("SBFileSpec(%p)::GetDirectory () => \"%s\"", m_opaque_ap.get(), s);
        else
            log->Printf ("SBFileSpec(%p)::GetDirectory () => NULL", m_opaque_ap.get());
    }
    return s;
}

void
SBFileSpec::SetFilename(const char *filename)
{
    if (filename && filename[0])
        m_opaque_ap->GetFilename().SetCString(filename);
    else
        m_opaque_ap->GetFilename().Clear();
}

void
SBFileSpec::SetDirectory(const char *directory)
{
    if (directory && directory[0])
        m_opaque_ap->GetDirectory().SetCString(directory);
    else
        m_opaque_ap->GetDirectory().Clear();
}

uint32_t
SBFileSpec::GetPath (char *dst_path, size_t dst_len) const
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    uint32_t result = m_opaque_ap->GetPath (dst_path, dst_len);

    if (log)
        log->Printf ("SBFileSpec(%p)::GetPath (dst_path=\"%.*s\", dst_len=%" PRIu64 ") => %u",
                     m_opaque_ap.get(), result, dst_path, (uint64_t)dst_len, result);

    if (result == 0 && dst_path && dst_len > 0)
        *dst_path = '\0';
    return result;
}


const lldb_private::FileSpec *
SBFileSpec::operator->() const
{
    return m_opaque_ap.get();
}

const lldb_private::FileSpec *
SBFileSpec::get() const
{
    return m_opaque_ap.get();
}


const lldb_private::FileSpec &
SBFileSpec::operator*() const
{
    return *m_opaque_ap.get();
}

const lldb_private::FileSpec &
SBFileSpec::ref() const
{
    return *m_opaque_ap.get();
}


void
SBFileSpec::SetFileSpec (const lldb_private::FileSpec& fs)
{
    *m_opaque_ap = fs;
}

bool
SBFileSpec::GetDescription (SBStream &description) const
{
    Stream &strm = description.ref();
    char path[PATH_MAX];
    if (m_opaque_ap->GetPath(path, sizeof(path)))
        strm.PutCString (path);    
    return true;
}
