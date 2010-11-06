//===-- MachVMMemory.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MachVMMemory.h"

#include <mach/mach_vm.h>

#include "MachVMRegion.h"
#include "ProcessMacOSXLog.h"

using namespace lldb;
using namespace lldb_private;

MachVMMemory::MachVMMemory() :
    m_page_size (kInvalidPageSize)
{
}

MachVMMemory::~MachVMMemory()
{
}

size_t
MachVMMemory::PageSize(lldb_private::Error &error)
{
    if (m_page_size == kInvalidPageSize)
    {
        error = ::host_page_size( ::mach_host_self(), &m_page_size);
        if (error.Fail())
            m_page_size = 0;
    }

    if (m_page_size != 0 && m_page_size != kInvalidPageSize)
    {
        if (error.Success())
            error.SetErrorString ("unable to determine page size");
    }
    return m_page_size;
}

size_t
MachVMMemory::MaxBytesLeftInPage (lldb::addr_t addr, size_t count)
{
    Error error;
    const size_t page_size = PageSize(error);
    if (page_size > 0)
    {
        size_t page_offset = (addr % page_size);
        size_t bytes_left_in_page = page_size - page_offset;
        if (count > bytes_left_in_page)
            count = bytes_left_in_page;
    }
    return count;
}

size_t
MachVMMemory::Read(task_t task, lldb::addr_t address, void *data, size_t data_count, Error &error)
{
    if (data == NULL || data_count == 0)
        return 0;

    size_t total_bytes_read = 0;
    lldb::addr_t curr_addr = address;
    uint8_t *curr_data = (uint8_t*)data;
    while (total_bytes_read < data_count)
    {
        mach_vm_size_t curr_size = MaxBytesLeftInPage(curr_addr, data_count - total_bytes_read);
        mach_msg_type_number_t curr_bytes_read = 0;
        vm_offset_t vm_memory = NULL;
        error = ::mach_vm_read (task, curr_addr, curr_size, &vm_memory, &curr_bytes_read);
        LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_MEMORY|PD_LOG_VERBOSE));

        if (log || error.Fail())
            error.PutToLog (log.get(), "::mach_vm_read (task = 0x%4.4x, addr = 0x%8.8llx, size = %llu, data => %8.8p, dataCnt => %i)", task, (uint64_t)curr_addr, (uint64_t)curr_size, vm_memory, curr_bytes_read);

        if (error.Success())
        {
            if (curr_bytes_read != curr_size)
            {
                if (log)
                    error.PutToLog (log.get(), "::mach_vm_read (task = 0x%4.4x, addr = 0x%8.8llx, size = %llu, data => %8.8p, dataCnt=>%i) only read %u of %llu bytes", task, (uint64_t)curr_addr, (uint64_t)curr_size, vm_memory, curr_bytes_read, curr_bytes_read, (uint64_t)curr_size);
            }
            ::memcpy (curr_data, (void *)vm_memory, curr_bytes_read);
            ::vm_deallocate (mach_task_self (), vm_memory, curr_bytes_read);
            total_bytes_read += curr_bytes_read;
            curr_addr += curr_bytes_read;
            curr_data += curr_bytes_read;
        }
        else
        {
            break;
        }
    }
    return total_bytes_read;
}


size_t
MachVMMemory::Write(task_t task, lldb::addr_t address, const void *data, size_t data_count, Error &error)
{
    MachVMRegion vmRegion(task);

    size_t total_bytes_written = 0;
    lldb::addr_t curr_addr = address;
    const uint8_t *curr_data = (const uint8_t*)data;


    while (total_bytes_written < data_count)
    {
        if (vmRegion.GetRegionForAddress(curr_addr))
        {
            mach_vm_size_t curr_data_count = data_count - total_bytes_written;
            mach_vm_size_t region_bytes_left = vmRegion.BytesRemaining(curr_addr);
            if (region_bytes_left == 0)
            {
                break;
            }
            if (curr_data_count > region_bytes_left)
                curr_data_count = region_bytes_left;

            if (vmRegion.SetProtections(curr_addr, curr_data_count, VM_PROT_READ | VM_PROT_WRITE))
            {
                size_t bytes_written = WriteRegion(task, curr_addr, curr_data, curr_data_count, error);
                if (bytes_written <= 0)
                {
                    // Error should have already be posted by WriteRegion...
                    break;
                }
                else
                {
                    total_bytes_written += bytes_written;
                    curr_addr += bytes_written;
                    curr_data += bytes_written;
                }
            }
            else
            {
                ProcessMacOSXLog::LogIf (PD_LOG_MEMORY_PROTECTIONS, "Failed to set read/write protections on region for address: [0x%8.8llx-0x%8.8llx)", (uint64_t)curr_addr, (uint64_t)(curr_addr + curr_data_count));
                break;
            }
        }
        else
        {
            ProcessMacOSXLog::LogIf (PD_LOG_MEMORY_PROTECTIONS, "Failed to get region for address: 0x%8.8llx", (uint64_t)address);
            break;
        }
    }

    return total_bytes_written;
}


size_t
MachVMMemory::WriteRegion(task_t task, const lldb::addr_t address, const void *data, const size_t data_count, Error &error)
{
    if (data == NULL || data_count == 0)
        return 0;

    size_t total_bytes_written = 0;
    lldb::addr_t curr_addr = address;
    const uint8_t *curr_data = (const uint8_t*)data;
    while (total_bytes_written < data_count)
    {
        mach_msg_type_number_t curr_data_count = MaxBytesLeftInPage(curr_addr, data_count - total_bytes_written);
        error = ::mach_vm_write (task, curr_addr, (pointer_t) curr_data, curr_data_count);
        LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_MEMORY));
        if (log || error.Fail())
            error.PutToLog (log.get(), "::mach_vm_write ( task = 0x%4.4x, addr = 0x%8.8llx, data = %8.8p, dataCnt = %u )", task, (uint64_t)curr_addr, curr_data, curr_data_count);

#if defined (__powerpc__) || defined (__ppc__)
        vm_machine_attribute_val_t mattr_value = MATTR_VAL_CACHE_FLUSH;

        error = ::vm_machine_attribute (task, curr_addr, curr_data_count, MATTR_CACHE, &mattr_value);
        if (log || error.Fail())
            error.Log(log.get(), "::vm_machine_attribute ( task = 0x%4.4x, addr = 0x%8.8llx, size = %u, attr = MATTR_CACHE, mattr_value => MATTR_VAL_CACHE_FLUSH )", task, (uint64_t)curr_addr, curr_data_count);
#endif

        if (error.Success())
        {
            total_bytes_written += curr_data_count;
            curr_addr += curr_data_count;
            curr_data += curr_data_count;
        }
        else
        {
            break;
        }
    }
    return total_bytes_written;
}
