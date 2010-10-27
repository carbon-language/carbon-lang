//===-- MachVMRegion.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <mach/mach_vm.h>
#include "MachVMRegion.h"
#include "ProcessMacOSXLog.h"

using namespace lldb_private;

MachVMRegion::MachVMRegion(task_t task) :
    m_task(task),
    m_addr(LLDB_INVALID_ADDRESS),
    m_err(),
    m_start(LLDB_INVALID_ADDRESS),
    m_size(0),
    m_depth(-1),
    m_data(),
    m_curr_protection(0),
    m_protection_addr(LLDB_INVALID_ADDRESS),
    m_protection_size(0)
{
    memset(&m_data, 0, sizeof(m_data));
}

MachVMRegion::~MachVMRegion()
{
    // Restore any original protections and clear our vars
    Clear();
}

void
MachVMRegion::Clear()
{
    RestoreProtections();
    m_addr = LLDB_INVALID_ADDRESS;
    m_err.Clear();
    m_start = LLDB_INVALID_ADDRESS;
    m_size = 0;
    m_depth = -1;
    memset(&m_data, 0, sizeof(m_data));
    m_curr_protection = 0;
    m_protection_addr = LLDB_INVALID_ADDRESS;
    m_protection_size = 0;
}

bool
MachVMRegion::SetProtections(mach_vm_address_t addr, mach_vm_size_t size, vm_prot_t prot)
{
    if (ContainsAddress(addr))
    {
        mach_vm_size_t prot_size = size;
        mach_vm_address_t end_addr = EndAddress();
        if (prot_size > (end_addr - addr))
            prot_size = end_addr - addr;


        Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_MEMORY_PROTECTIONS);
        if (prot_size > 0)
        {
            if (prot == (m_curr_protection & VM_PROT_ALL))
            {
                if (log)
                    log->Printf ("MachVMRegion::%s: protections (%u) already sufficient for task 0x%4.4x at address 0x%8.8llx) ", __FUNCTION__, prot, m_task, (uint64_t)addr);
                // Protections are already set as requested...
                return true;
            }
            else
            {
                m_err = ::mach_vm_protect (m_task, addr, prot_size, 0, prot);
                if (log || m_err.Fail())
                    m_err.PutToLog(log, "::mach_vm_protect ( task = 0x%4.4x, addr = 0x%8.8llx, size = %llu, set_max = %i, prot = %u )", m_task, (uint64_t)addr, (uint64_t)prot_size, 0, prot);
                if (m_err.Fail())
                {
                    // Try again with the ability to create a copy on write region
                    m_err = ::mach_vm_protect (m_task, addr, prot_size, 0, prot | VM_PROT_COPY);
                    if (log || m_err.Fail())
                        m_err.PutToLog(log, "::mach_vm_protect ( task = 0x%4.4x, addr = 0x%8.8llx, size = %llu, set_max = %i, prot = %u )", m_task, (uint64_t)addr, (uint64_t)prot_size, 0, prot | VM_PROT_COPY);
                }
                if (m_err.Success())
                {
                    m_curr_protection = prot;
                    m_protection_addr = addr;
                    m_protection_size = prot_size;
                    return true;
                }
            }
        }
        else
        {
            log->Printf("MachVMRegion::%s: Zero size for task 0x%4.4x at address 0x%8.8llx) ", __FUNCTION__, m_task, (uint64_t)addr);
        }
    }
    return false;
}

bool
MachVMRegion::RestoreProtections()
{
    if (m_curr_protection != m_data.protection && m_protection_size > 0)
    {
        m_err = ::mach_vm_protect (m_task, m_protection_addr, m_protection_size, 0, m_data.protection);
        Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_MEMORY_PROTECTIONS);
        if (log || m_err.Fail())
            m_err.PutToLog(log, "::mach_vm_protect ( task = 0x%4.4x, addr = 0x%8.8llx, size = %llu, set_max = %i, prot = %u )", m_task, (uint64_t)m_protection_addr, (uint64_t)m_protection_size, 0, m_data.protection);
        if (m_err.Success())
        {
            m_protection_size = 0;
            m_protection_addr = LLDB_INVALID_ADDRESS;
            m_curr_protection = m_data.protection;
            return true;
        }
    }
    else
    {
        m_err.Clear();
        return true;
    }

    return false;
}

bool
MachVMRegion::GetRegionForAddress(lldb::addr_t addr)
{
    // Restore any original protections and clear our vars
    Clear();
    m_addr = addr;
    m_start = addr;
    m_depth = 1024;
    mach_msg_type_number_t info_size = kRegionInfoSize;
    assert(sizeof(info_size) == 4);
    m_err = ::mach_vm_region_recurse (m_task, &m_start, &m_size, &m_depth, (vm_region_recurse_info_t)&m_data, &info_size);
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_MEMORY_PROTECTIONS);
    if (log || m_err.Fail())
        m_err.PutToLog(log, "::mach_vm_region_recurse ( task = 0x%4.4x, address => 0x%8.8llx, size => %llu, nesting_depth => %d, info => %p, infoCnt => %d) addr = 0x%8.8llx ", m_task, (uint64_t)m_start, (uint64_t)m_size, m_depth, &m_data, info_size, (uint64_t)addr);
    if (m_err.Fail())
    {
        return false;
    }
    else
    {
        if (log && log->GetMask().Test(PD_LOG_VERBOSE))
        {
            log->Printf("info = { prot = %u, "
                        "max_prot = %u, "
                        "inheritance = 0x%8.8x, "
                        "offset = 0x%8.8llx, "
                        "user_tag = 0x%8.8x, "
                        "ref_count = %u, "
                        "shadow_depth = %u, "
                        "ext_pager = %u, "
                        "share_mode = %u, "
                        "is_submap = %d, "
                        "behavior = %d, "
                        "object_id = 0x%8.8x, "
                        "user_wired_count = 0x%4.4x }",
                        m_data.protection,
                        m_data.max_protection,
                        m_data.inheritance,
                        (uint64_t)m_data.offset,
                        m_data.user_tag,
                        m_data.ref_count,
                        m_data.shadow_depth,
                        m_data.external_pager,
                        m_data.share_mode,
                        m_data.is_submap,
                        m_data.behavior,
                        m_data.object_id,
                        m_data.user_wired_count);
        }
    }

    m_curr_protection = m_data.protection;

    return true;
}
