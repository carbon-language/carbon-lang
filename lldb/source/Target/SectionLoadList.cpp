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
    return m_addr_to_sect.empty();
}

void
SectionLoadList::Clear ()
{
    Mutex::Locker locker(m_mutex);
    m_addr_to_sect.clear();
    m_sect_to_addr.clear();
}

addr_t
SectionLoadList::GetSectionLoadAddress (const Section *section) const
{
    // TODO: add support for the same section having multiple load addresses
    addr_t section_load_addr = LLDB_INVALID_ADDRESS;
    if (section)
    {
        Mutex::Locker locker(m_mutex);
        sect_to_addr_collection::const_iterator pos = m_sect_to_addr.find (section);
        
        if (pos != m_sect_to_addr.end())
            section_load_addr = pos->second;
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
    sect_to_addr_collection::iterator sta_pos = m_sect_to_addr.find(section);
    if (sta_pos != m_sect_to_addr.end())
    {
        if (load_addr == sta_pos->second)
            return false; // No change...
        else
            sta_pos->second = load_addr;
    }
    else
        m_sect_to_addr[section] = load_addr;

    addr_to_sect_collection::iterator ats_pos = m_addr_to_sect.find(load_addr);
    if (ats_pos != m_addr_to_sect.end())
    {
        assert (section != ats_pos->second);
        ats_pos->second = section;
    }
    else
        m_addr_to_sect[load_addr] = section;

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
    
    sect_to_addr_collection::iterator sta_pos = m_sect_to_addr.find(section);
    if (sta_pos != m_sect_to_addr.end())
    {
        addr_t load_addr = sta_pos->second;
        m_sect_to_addr.erase (sta_pos);

        addr_to_sect_collection::iterator ats_pos = m_addr_to_sect.find(load_addr);
        if (ats_pos != m_addr_to_sect.end())
            m_addr_to_sect.erase (ats_pos);
    }
    
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
    bool erased = false;
    Mutex::Locker locker(m_mutex);
    sect_to_addr_collection::iterator sta_pos = m_sect_to_addr.find(section);
    if (sta_pos != m_sect_to_addr.end())
    {
        erased = true;
        m_sect_to_addr.erase (sta_pos);
    }
        
    addr_to_sect_collection::iterator ats_pos = m_addr_to_sect.find(load_addr);
    if (ats_pos != m_addr_to_sect.end())
    {
        erased = true;
        m_addr_to_sect.erase (ats_pos);
    }

    return erased;
}


bool
SectionLoadList::ResolveLoadAddress (addr_t load_addr, Address &so_addr) const
{
    // First find the top level section that this load address exists in    
    Mutex::Locker locker(m_mutex);
    if (!m_addr_to_sect.empty())
    {
        addr_to_sect_collection::const_iterator pos = m_addr_to_sect.lower_bound (load_addr);
        if (pos != m_addr_to_sect.end())
        {
            if (load_addr != pos->first && pos != m_addr_to_sect.begin())
                --pos;
            if (load_addr >= pos->first)
            {
                addr_t offset = load_addr - pos->first;
                if (offset < pos->second->GetByteSize())
                {
                    // We have found the top level section, now we need to find the
                    // deepest child section.
                    return pos->second->ResolveContainedAddress (offset, so_addr);
                }
            }
        }
        else
        {
            // There are no entries that have an address that is >= load_addr,
            // so we need to check the last entry on our collection.
            addr_to_sect_collection::const_reverse_iterator rpos = m_addr_to_sect.rbegin();
            if (load_addr >= rpos->first)
            {
                addr_t offset = load_addr - rpos->first;
                if (offset < rpos->second->GetByteSize())
                {
                    // We have found the top level section, now we need to find the
                    // deepest child section.
                    return rpos->second->ResolveContainedAddress (offset, so_addr);
                }
            }
        }
    }
    so_addr.Clear();
    return false;
}

void
SectionLoadList::Dump (Stream &s, Target *target)
{
    Mutex::Locker locker(m_mutex);
    addr_to_sect_collection::const_iterator pos, end;
    for (pos = m_addr_to_sect.begin(), end = m_addr_to_sect.end(); pos != end; ++pos)
    {
        s.Printf("addr = 0x%16.16llx, section = %p: ", pos->first, pos->second);
        pos->second->Dump (&s, target, 0);
    }
}


