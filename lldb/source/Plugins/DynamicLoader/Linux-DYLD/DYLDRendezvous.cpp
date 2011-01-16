//===-- DYLDRendezvous.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "DYLDRendezvous.h"

using namespace lldb;
using namespace lldb_private;

/// Locates the address of the rendezvous structure.  Returns the address on
/// success and LLDB_INVALID_ADDRESS on failure.
static addr_t
ResolveRendezvousAddress(Process *process)
{
    addr_t info_location;
    addr_t info_addr;
    Error error;
    size_t size;

    info_location = process->GetImageInfoAddress();

    if (info_location == LLDB_INVALID_ADDRESS)
        return LLDB_INVALID_ADDRESS;

    info_addr = 0;
    size = process->DoReadMemory(info_location, &info_addr,
                                 process->GetAddressByteSize(), error);
    if (size != process->GetAddressByteSize() || error.Fail())
        return LLDB_INVALID_ADDRESS;

    if (info_addr == 0)
        return LLDB_INVALID_ADDRESS;

    return info_addr;
}

DYLDRendezvous::DYLDRendezvous(Process *process)
    : m_process(process),
      m_rendezvous_addr(LLDB_INVALID_ADDRESS),
      m_current(),
      m_previous(),
      m_soentries(),
      m_added_soentries(),
      m_removed_soentries()
{
}

bool
DYLDRendezvous::Resolve()
{
    const size_t word_size = 4;
    Rendezvous info;
    size_t address_size;
    size_t padding;
    addr_t info_addr;
    addr_t cursor;

    address_size = m_process->GetAddressByteSize();
    padding = address_size - word_size;

    if (m_rendezvous_addr == LLDB_INVALID_ADDRESS)
        cursor = info_addr = ResolveRendezvousAddress(m_process);
    else
        cursor = info_addr = m_rendezvous_addr;
    
    if (cursor == LLDB_INVALID_ADDRESS)
        return false;

    if (!(cursor = ReadMemory(cursor, &info.version, word_size)))
        return false;

    if (!(cursor = ReadMemory(cursor + padding, &info.map_addr, address_size)))
        return false;

    if (!(cursor = ReadMemory(cursor, &info.brk, address_size)))
        return false;

    if (!(cursor = ReadMemory(cursor, &info.state, word_size)))
        return false;

    if (!(cursor = ReadMemory(cursor + padding, &info.ldbase, address_size)))
        return false;

    // The rendezvous was successfully read.  Update our internal state.
    m_rendezvous_addr = info_addr;
    m_previous = m_current;
    m_current = info;

    return UpdateSOEntries();
}

bool
DYLDRendezvous::IsValid()
{
    return m_rendezvous_addr != LLDB_INVALID_ADDRESS;
}

bool
DYLDRendezvous::UpdateSOEntries()
{
    SOEntry entry;

    if (m_current.map_addr == 0)
        return false;

    // If we are about to add or remove a shared object clear out the current
    // state and take a snapshot of the currently loaded images.
    if (m_current.state == eAdd || m_current.state == eDelete)
    {
        assert(m_previous.state == eConsistent);
        m_soentries.clear();
        m_added_soentries.clear();
        m_removed_soentries.clear();
        return TakeSnapshot(m_soentries);
    }

    // Otherwise check the previous state to determine what to expect and update
    // accordingly.
    if (m_previous.state == eAdd)
        return UpdateSOEntriesForAddition();
    else if (m_previous.state == eDelete)
        return UpdateSOEntriesForDeletion();

    return false;
}
 
bool
DYLDRendezvous::UpdateSOEntriesForAddition()
{
    SOEntry entry;
    iterator pos;

    assert(m_previous.state == eAdd);

    if (m_current.map_addr == 0)
        return false;

    for (addr_t cursor = m_current.map_addr; cursor != 0; cursor = entry.next)
    {
        if (!ReadSOEntryFromMemory(cursor, entry))
            return false;

        if (entry.path.empty())
            continue;

        pos = std::find(m_soentries.begin(), m_soentries.end(), entry);
        if (pos == m_soentries.end())
        {
            m_soentries.push_back(entry);
            m_added_soentries.push_back(entry);
        }
    }

    return true;
}

bool
DYLDRendezvous::UpdateSOEntriesForDeletion()
{
    SOEntryList entry_list;
    iterator pos;

    assert(m_previous.state == eDelete);

    if (!TakeSnapshot(entry_list))
        return false;

    for (iterator I = begin(); I != end(); ++I)
    {
        pos = std::find(entry_list.begin(), entry_list.end(), *I);
        if (pos == entry_list.end())
            m_removed_soentries.push_back(*I);
    }

    m_soentries = entry_list;
    return true;
}

bool
DYLDRendezvous::TakeSnapshot(SOEntryList &entry_list)
{
    SOEntry entry;

    if (m_current.map_addr == 0)
        return false;

    for (addr_t cursor = m_current.map_addr; cursor != 0; cursor = entry.next)
    {
        if (!ReadSOEntryFromMemory(cursor, entry))
            return false;

        if (entry.path.empty())
            continue;

        entry_list.push_back(entry);
    }

    return true;
}

addr_t
DYLDRendezvous::ReadMemory(addr_t addr, void *dst, size_t size)
{
    size_t bytes_read;
    Error error;

    bytes_read = m_process->DoReadMemory(addr, dst, size, error);
    if (bytes_read != size || error.Fail())
        return 0;

    return addr + bytes_read;
}

std::string
DYLDRendezvous::ReadStringFromMemory(addr_t addr)
{
    std::string str;
    Error error;
    size_t size;
    char c;

    if (addr == LLDB_INVALID_ADDRESS)
        return std::string();

    for (;;) {
        size = m_process->DoReadMemory(addr, &c, 1, error);
        if (size != 1 || error.Fail())
            return std::string();
        if (c == 0)
            break;
        else {
            str.push_back(c);
            addr++;
        }
    }

    return str;
}

bool
DYLDRendezvous::ReadSOEntryFromMemory(lldb::addr_t addr, SOEntry &entry)
{
    size_t address_size = m_process->GetAddressByteSize();

    entry.clear();
    
    if (!(addr = ReadMemory(addr, &entry.base_addr, address_size)))
        return false;
    
    if (!(addr = ReadMemory(addr, &entry.path_addr, address_size)))
        return false;
    
    if (!(addr = ReadMemory(addr, &entry.dyn_addr, address_size)))
        return false;
    
    if (!(addr = ReadMemory(addr, &entry.next, address_size)))
        return false;
    
    if (!(addr = ReadMemory(addr, &entry.prev, address_size)))
        return false;
    
    entry.path = ReadStringFromMemory(entry.path_addr);
    
    return true;
}

void
DYLDRendezvous::DumpToLog(LogSP log) const
{
    int state = GetState();

    if (!log)
        return;

    log->PutCString("DYLDRendezvous:");
    log->Printf("   Address: %lx", GetRendezvousAddress());
    log->Printf("   Version: %d",  GetVersion());
    log->Printf("   Link   : %lx", GetLinkMapAddress());
    log->Printf("   Break  : %lx", GetBreakAddress());
    log->Printf("   LDBase : %lx", GetLDBase());
    log->Printf("   State  : %s", 
                (state == eConsistent) ? "consistent" :
                (state == eAdd)        ? "add"        :
                (state == eDelete)     ? "delete"     : "unknown");
    
    iterator I = begin();
    iterator E = end();

    if (I != E) 
        log->PutCString("DYLDRendezvous SOEntries:");
    
    for (int i = 1; I != E; ++I, ++i) 
    {
        log->Printf("\n   SOEntry [%d] %s", i, I->path.c_str());
        log->Printf("      Base : %lx", I->base_addr);
        log->Printf("      Path : %lx", I->path_addr);
        log->Printf("      Dyn  : %lx", I->dyn_addr);
        log->Printf("      Next : %lx", I->next);
        log->Printf("      Prev : %lx", I->prev);
    }
}
