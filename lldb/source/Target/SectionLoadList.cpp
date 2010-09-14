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
    return m_section_load_info.IsEmpty();
}

void
SectionLoadList::Clear ()
{
    m_section_load_info.Clear();
}

addr_t
SectionLoadList::GetSectionLoadAddress (const Section *section) const
{
    // TODO: add support for the same section having multiple load addresses
    addr_t section_load_addr = LLDB_INVALID_ADDRESS;
    if (m_section_load_info.GetFirstKeyForValue (section, section_load_addr))
        return section_load_addr;
    return LLDB_INVALID_ADDRESS;
}

bool
SectionLoadList::SetSectionLoadAddress (const Section *section, addr_t load_addr)
{
    Log *log = lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_SHLIB | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SectionLoadList::%s (section = %p (%s.%s), load_addr = 0x%16.16llx)",
                     __FUNCTION__,
                     section,
                     section->GetModule()->GetFileSpec().GetFilename().AsCString(),
                     section->GetName().AsCString(),
                     load_addr);


    const Section *existing_section = NULL;
    Mutex::Locker locker(m_section_load_info.GetMutex());

    if (m_section_load_info.GetValueForKeyNoLock (load_addr, existing_section))
    {
        if (existing_section == section)
            return false;   // No change
    }
    m_section_load_info.SetValueForKeyNoLock (load_addr, section);
    return true;    // Changed
}

size_t
SectionLoadList::SetSectionUnloaded (const Section *section)
{
    Log *log = lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_SHLIB | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SectionLoadList::%s (section = %p (%s.%s))",
                     __FUNCTION__,
                     section,
                     section->GetModule()->GetFileSpec().GetFilename().AsCString(),
                     section->GetName().AsCString());

    Mutex::Locker locker(m_section_load_info.GetMutex());

    size_t unload_count = 0;
    addr_t section_load_addr;
    while (m_section_load_info.GetFirstKeyForValueNoLock (section, section_load_addr))
    {
        unload_count += m_section_load_info.EraseNoLock (section_load_addr);
    }
    return unload_count;
}

bool
SectionLoadList::SetSectionUnloaded (const Section *section, addr_t load_addr)
{
    Log *log = lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_SHLIB | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SectionLoadList::%s (section = %p (%s.%s), load_addr = 0x%16.16llx)",
                     __FUNCTION__,
                     section,
                     section->GetModule()->GetFileSpec().GetFilename().AsCString(),
                     section->GetName().AsCString(),
                     load_addr);

    return m_section_load_info.Erase (load_addr) == 1;
}


bool
SectionLoadList::ResolveLoadAddress (addr_t load_addr, Address &so_addr) const
{
    addr_t section_load_addr = LLDB_INVALID_ADDRESS;
    const Section *section = NULL;

    // First find the top level section that this load address exists in
    if (m_section_load_info.LowerBound (load_addr, section_load_addr, section, true))
    {
        addr_t offset = load_addr - section_load_addr;
        if (offset < section->GetByteSize())
        {
            // We have found the top level section, now we need to find the
            // deepest child section.
            return section->ResolveContainedAddress (offset, so_addr);
        }
    }
    so_addr.Clear();
    return false;
}
