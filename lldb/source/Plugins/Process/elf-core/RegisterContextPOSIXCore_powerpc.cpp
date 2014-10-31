//===-- RegisterContextCorePOSIX_powerpc.cpp ---------------------*- C++ -*-===//
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
#include "RegisterContextPOSIX.h"
#include "RegisterContextPOSIXCore_powerpc.h"

using namespace lldb_private;

RegisterContextCorePOSIX_powerpc::RegisterContextCorePOSIX_powerpc(Thread &thread,
                                                                 RegisterInfoInterface *register_info,
                                                                 const DataExtractor &gpregset,
                                                                 const DataExtractor &fpregset)
    : RegisterContextPOSIX_powerpc(thread, 0, register_info)
{
    m_gpr_buffer.reset(new DataBufferHeap(gpregset.GetDataStart(), gpregset.GetByteSize()));
    m_gpr.SetData(m_gpr_buffer);
    m_gpr.SetByteOrder(gpregset.GetByteOrder());
    m_fpr_buffer.reset(new DataBufferHeap(fpregset.GetDataStart(), fpregset.GetByteSize()));
    m_fpr.SetData(m_fpr_buffer);
    m_fpr.SetByteOrder(fpregset.GetByteOrder());
}

RegisterContextCorePOSIX_powerpc::~RegisterContextCorePOSIX_powerpc()
{
}

bool
RegisterContextCorePOSIX_powerpc::ReadGPR()
{
    return true;
}

bool
RegisterContextCorePOSIX_powerpc::ReadFPR()
{
    return true;
}

bool
RegisterContextCorePOSIX_powerpc::WriteGPR()
{
    assert(0);
    return false;
}

bool
RegisterContextCorePOSIX_powerpc::WriteFPR()
{
    assert(0);
    return false;
}

bool
RegisterContextCorePOSIX_powerpc::ReadRegister(const RegisterInfo *reg_info, RegisterValue &value)
{
    lldb::offset_t offset = reg_info->byte_offset;
    if (reg_info->name[0] == 'f') {
        uint64_t v = m_fpr.GetMaxU64(&offset, reg_info->byte_size);
        if (offset == reg_info->byte_offset + reg_info->byte_size)
        {
            value = v;
            return true;
        }
    } else {
        uint64_t v = m_gpr.GetMaxU64(&offset, reg_info->byte_size);
        if (offset == reg_info->byte_offset + reg_info->byte_size)
        {
            if (reg_info->byte_size < sizeof(v))
                value = (uint32_t)v;
            else
                value = v;
            return true;
        }
    }
    return false;
}

bool
RegisterContextCorePOSIX_powerpc::ReadAllRegisterValues(lldb::DataBufferSP &data_sp)
{
    return false;
}

bool
RegisterContextCorePOSIX_powerpc::WriteRegister(const RegisterInfo *reg_info, const RegisterValue &value)
{
    return false;
}

bool
RegisterContextCorePOSIX_powerpc::WriteAllRegisterValues(const lldb::DataBufferSP &data_sp)
{
    return false;
}

bool
RegisterContextCorePOSIX_powerpc::HardwareSingleStep(bool enable)
{
    return false;
}
