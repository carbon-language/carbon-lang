//===-- MachVMMemory.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MachVMMemory_h_
#define liblldb_MachVMMemory_h_

#include <mach/mach.h>

#include "lldb/lldb-private.h"
#include "lldb/Core/Error.h"

class MachVMMemory
{
public:
    enum { kInvalidPageSize = ~0 };
    MachVMMemory();
    ~MachVMMemory();
    size_t Read(task_t task, lldb::addr_t address, void *data, size_t data_count, lldb_private::Error &error);
    size_t Write(task_t task, lldb::addr_t address, const void *data, size_t data_count, lldb_private::Error &error);
    size_t PageSize(lldb_private::Error &error);

protected:
    size_t MaxBytesLeftInPage(lldb::addr_t addr, size_t count);

    size_t WriteRegion(task_t task, const lldb::addr_t address, const void *data, const size_t data_count, lldb_private::Error &error);
    vm_size_t m_page_size;
};


#endif //    #ifndef liblldb_MachVMMemory_h_
