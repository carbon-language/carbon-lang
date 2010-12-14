//===-- SectionLoadList.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/SectionLoadList.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"

using namespace lldb;
using namespace lldb_private;


bool
SectionLoadList::IsEmpty() const
{
    Mutex::Locker locker(m_mutex);
    return m_collection.empty();
}

void
SectionLoadList::Clear ()
{
    Mutex::Locker locker(m_mutex);
    return m_collection.clear();
}

addr_t
SectionLoadList::GetSectionLoadAddress (const Section *section) const
{
    // TODO: add support for the same section having multiple load addresses
    addr_t section_load_addr = LLDB_INVALID_ADDRESS;
    if (section)
    {
        Mutex::Locker locker(m_mutex);
        collection::const_iterator pos, end = m_collection.end();
        for (pos = m_collection.begin(); pos != end; ++pos)
        {
            const addr_t pos_load_addr = pos->first;
            const Section *pos_section = pos->second;
            if (pos_section == section)
            {
                section_load_addr = pos_load_addr;
                break;
            }
        }
    }
    return section_load_addr;
}

bool
SectionLoadList::SetSectionLoadAddress (const Section *section, addr_t load_addr)
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_DYNAMIC_LOADER | LIBLLDB_LOG_VERBOSE));

    if (log)
    {
        const FileSpec &module_file_spec (section->GetModule()->GetFileSpec());
        log->Printf ("SectionLoadList::%s (section = %p (%s%s%s.%s), load_addr = 0x%16.16llx)",
                     __FUNCTION__,
                     section,
                     module_file_spec.GetDirectory().AsCString(),
                     module_file_spec.GetDirectory() ? "/" : "",
                     module_file_spec.GetFilename().AsCString(),
                     section->GetName().AsCString(),
                     load_addr);
    }

    Mutex::Locker locker(m_mutex);
    collection::iterator pos = m_collection.find(load_addr);
    if (pos != m_collection.end())
    {
        if (section == pos->second)
            return false; // No change...
        else
            pos->second = section;
    }
    else
    {
        m_collection[load_addr] = section;
    }
    return true;    // Changed
}

size_t
SectionLoadList::SetSectionUnloaded (const Section *section)
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_DYNAMIC_LOADER | LIBLLDB_LOG_VERBOSE));

    if (log)
    {
        const FileSpec &module_file_spec (section->GetModule()->GetFileSpec());
        log->Printf ("SectionLoadList::%s (section = %p (%s%s%s.%s))",
                     __FUNCTION__,
                     section,
                     module_file_spec.GetDirectory().AsCString(),
                     module_file_spec.GetDirectory() ? "/" : "",
                     module_file_spec.GetFilename().AsCString(),
                     section->GetName().AsCString());
    }

    size_t unload_count = 0;
    Mutex::Locker locker(m_mutex);
    bool erased = false;
    do 
    {
        erased = false;
        for (collection::iterator pos = m_collection.begin(); pos != m_collection.end(); ++pos)
        {
            if (pos->second == section)
            {
                m_collection.erase(pos);
                erased = true;
            }
        }
    } while (erased);
    
    return unload_count;
}

bool
SectionLoadList::SetSectionUnloaded (const Section *section, addr_t load_addr)
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_DYNAMIC_LOADER | LIBLLDB_LOG_VERBOSE));

    if (log)
    {
        const FileSpec &module_file_spec (section->GetModule()->GetFileSpec());
        log->Printf ("SectionLoadList::%s (section = %p (%s%s%s.%s), load_addr = 0x%16.16llx)",
                     __FUNCTION__,
                     section,
                     module_file_spec.GetDirectory().AsCString(),
                     module_file_spec.GetDirectory() ? "/" : "",
                     module_file_spec.GetFilename().AsCString(),
                     section->GetName().AsCString(),
                     load_addr);
    }
    Mutex::Locker locker(m_mutex);
    return m_collection.erase (load_addr) != 0;
}


bool
SectionLoadList::ResolveLoadAddress (addr_t load_addr, Address &so_addr) const
{
    // First find the top level section that this load address exists in    
    Mutex::Locker locker(m_mutex);
    collection::const_iterator pos = m_collection.lower_bound (load_addr);
    if (pos != m_collection.end())
    {
        if (load_addr != pos->first && pos != m_collection.begin())
            --pos;
        assert (load_addr >= pos->first);
        addr_t offset = load_addr - pos->first;
        if (offset < pos->second->GetByteSize())
        {
            // We have found the top level section, now we need to find the
            // deepest child section.
            return pos->second->ResolveContainedAddress (offset, so_addr);
        }
    }
    so_addr.Clear();
    return false;
}

void
SectionLoadList::Dump (Stream &s, Target *target)
{
    Mutex::Locker locker(m_mutex);
    collection::const_iterator pos, end;
    for (pos = m_collection.begin(), end = m_collection.end(); pos != end; ++pos)
    {
        s.Printf("addr = 0x%16.16llx, section = %p: ", pos->first, pos->second);
        pos->second->Dump (&s, target, 0);
    }
}


