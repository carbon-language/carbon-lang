//===-- RegisterContextCorePOSIX_powerpc.h ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextCorePOSIX_powerpc_H_
#define liblldb_RegisterContextCorePOSIX_powerpc_H_

#include "lldb/Core/DataBufferHeap.h"
#include "Plugins/Process/Utility/RegisterContextPOSIX_powerpc.h"

class RegisterContextCorePOSIX_powerpc :
    public RegisterContextPOSIX_powerpc
{
public:
    RegisterContextCorePOSIX_powerpc (lldb_private::Thread &thread,
                                     lldb_private::RegisterInfoInterface *register_info,
                                     const lldb_private::DataExtractor &gpregset,
                                     const lldb_private::DataExtractor &fpregset,
                                     const lldb_private::DataExtractor &vregset);

    ~RegisterContextCorePOSIX_powerpc();

    virtual bool
    ReadRegister(const lldb_private::RegisterInfo *reg_info, lldb_private::RegisterValue &value);

    virtual bool
    WriteRegister(const lldb_private::RegisterInfo *reg_info, const lldb_private::RegisterValue &value);

    bool
    ReadAllRegisterValues(lldb::DataBufferSP &data_sp);

    bool
    WriteAllRegisterValues(const lldb::DataBufferSP &data_sp);

    bool
    HardwareSingleStep(bool enable);

protected:
    bool
    ReadGPR();

    bool
    ReadFPR();

    bool
    ReadVMX();

    bool
    WriteGPR();

    bool
    WriteFPR();

    bool
    WriteVMX();

private:
    lldb::DataBufferSP m_gpr_buffer;
    lldb::DataBufferSP m_fpr_buffer;
    lldb::DataBufferSP m_vec_buffer;
    lldb_private::DataExtractor m_gpr;
    lldb_private::DataExtractor m_fpr;
    lldb_private::DataExtractor m_vec;
};

#endif // #ifndef liblldb_RegisterContextCorePOSIX_powerpc_H_
