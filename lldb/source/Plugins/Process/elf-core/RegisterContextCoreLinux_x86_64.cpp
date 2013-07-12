//===-- RegisterContextCoreLinux_x86_64.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Target/Thread.h"
#include "RegisterContextCoreLinux_x86_64.h"

RegisterContextCoreLinux_x86_64::RegisterContextCoreLinux_x86_64(Thread &thread,
                                                const DataExtractor &gpregset,
                                                const DataExtractor &fpregset)
    : RegisterContextLinux_x86_64(thread, 0)
{
    size_t size, len;

    size = GetGPRSize();
    m_gpregset = new uint8_t[size];
    len = gpregset.ExtractBytes(0, size, lldb::eByteOrderLittle, m_gpregset);
    assert(len == size);
}

RegisterContextCoreLinux_x86_64::~RegisterContextCoreLinux_x86_64()
{
    delete [] m_gpregset;
}

bool
RegisterContextCoreLinux_x86_64::ReadRegister(const RegisterInfo *reg_info, RegisterValue &value)
{
    value = *(uint64_t *)(m_gpregset + reg_info->byte_offset);
    return true;
}

bool
RegisterContextCoreLinux_x86_64::ReadAllRegisterValues(lldb::DataBufferSP &data_sp)
{
    return false;
}

bool
RegisterContextCoreLinux_x86_64::WriteRegister(const RegisterInfo *reg_info, const RegisterValue &value)
{
    return false;
}

bool
RegisterContextCoreLinux_x86_64::WriteAllRegisterValues(const lldb::DataBufferSP &data_sp)
{
    return false;
}

bool
RegisterContextCoreLinux_x86_64::UpdateAfterBreakpoint()
{
    return false;
}

bool
RegisterContextCoreLinux_x86_64::HardwareSingleStep(bool enable)
{
    return false;
}

