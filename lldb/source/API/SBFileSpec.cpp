//===-- SBFileSpec.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/Log.h"

using namespace lldb;
using namespace lldb_private;



SBFileSpec::SBFileSpec () :
    m_opaque_ap()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBFileSpec::SBFileSpec () ==> this = %p", this);
}

SBFileSpec::SBFileSpec (const SBFileSpec &rhs) :
    m_opaque_ap()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
    {
        SBStream sstr;
        rhs.GetDescription (sstr);
        log->Printf ("SBFileSpec::SBFileSpec (const SBFileSpec &rhs) rhs.m_opaque_ap.get() = %p (%s) ==> this = %p",
                     rhs.m_opaque_ap.get(), sstr.GetData(), this);
    }

    if (rhs.m_opaque_ap.get())
        m_opaque_ap.reset (new FileSpec (rhs.get()));
}

// Deprected!!!
SBFileSpec::SBFileSpec (const char *path) :
    m_opaque_ap(new FileSpec (path, true))
{
}

SBFileSpec::SBFileSpec (const char *path, bool resolve) :
    m_opaque_ap(new FileSpec (path, resolve))
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBFileSpec::SBFileSpec (const char *path, bool resolve) path = '%s', resolve = %s ==> "
                     "this = %p (m_opaque_ap.get() = %p)", path, (resolve ? "true" : "false"), this,
                     m_opaque_ap.get());
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
            m_opaque_ap.reset (new lldb_private::FileSpec(*rhs.m_opaque_ap.get()));
    }
    return *this;
}

bool
SBFileSpec::IsValid() const
{
    return m_opaque_ap.get() != NULL;
}

bool
SBFileSpec::Exists () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBFileSpec::Exists ()");

    bool result = false;
    if (m_opaque_ap.get())
        result = m_opaque_ap->Exists();

    if (log)
        log->Printf ("SBFileSpec::Exists ==> %s", (result ? "true" : "false"));

    return result;
}

bool
SBFileSpec::ResolveExecutableLocation ()
{
    if (m_opaque_ap.get())
        return m_opaque_ap->ResolveExecutableLocation ();
    return false;
}

int
SBFileSpec::ResolvePath (const char *src_path, char *dst_path, size_t dst_len)
{
    return lldb_private::FileSpec::Resolve (src_path, dst_path, dst_len);
}

const char *
SBFileSpec::GetFilename() const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBFileSpec::GetFilename ()");

    if (m_opaque_ap.get())
    {
        if (log)
            log->Printf ("SBFileSpec::GetFilename ==> %s", m_opaque_ap->GetFilename().AsCString());
        return m_opaque_ap->GetFilename().AsCString();
    }

    if (log)
        log->Printf ("SBFileSpec::GetFilename ==> NULL");

    return NULL;
}

const char *
SBFileSpec::GetDirectory() const
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetDirectory().AsCString();
    return NULL;
}

uint32_t
SBFileSpec::GetPath (char *dst_path, size_t dst_len) const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBFileSpec::GetPath (dst_path, dst_len)");

    uint32_t result;
    if (m_opaque_ap.get())
    {
        result = m_opaque_ap->GetPath (dst_path, dst_len);
        if (log)
            log->Printf ("SBFileSpec::GetPath ==> %s (%d)", dst_path, result);
        return result;
    }

    if (log)
        log->Printf ("SBFileSpec::GetPath ==> NULL (0)");

    if (dst_path && dst_len)
        *dst_path = '\0';
    return 0;
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
    if (m_opaque_ap.get())
        *m_opaque_ap = fs;
    else
        m_opaque_ap.reset (new FileSpec (fs));
}

bool
SBFileSpec::GetDescription (SBStream &description)
{
    if (m_opaque_ap.get())
    {
        const char *filename = GetFilename();
        const char *dir_name = GetDirectory();
        if (!filename && !dir_name)
            description.Printf ("No value");
        else if (!dir_name)
            description.Printf ("%s", filename);
        else
            description.Printf ("%s/%s", dir_name, filename);
    }
    else
        description.Printf ("No value");
    
    return true;
}

bool
SBFileSpec::GetDescription (SBStream &description) const
{
    if (m_opaque_ap.get())
    {
        const char *filename = GetFilename();
        const char *dir_name = GetDirectory();
        if (!filename && !dir_name)
            description.Printf ("No value");
        else if (!dir_name)
            description.Printf ("%s", filename);
        else
            description.Printf ("%s/%s", dir_name, filename);
    }
    else
        description.Printf ("No value");
    
    return true;
}
