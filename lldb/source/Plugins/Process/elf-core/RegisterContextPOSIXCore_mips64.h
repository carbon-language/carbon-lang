//===-- RegisterContextCorePOSIX_mips64.h ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextCorePOSIX_mips64_H_
#define liblldb_RegisterContextCorePOSIX_mips64_H_

#include "Plugins/Process/POSIX/RegisterContextPOSIX_mips64.h"

class RegisterContextCorePOSIX_mips64 :
    public RegisterContextPOSIX_mips64
{
public:
    RegisterContextCorePOSIX_mips64 (lldb_private::Thread &thread,
                                     RegisterInfoInterface *register_info,
                                     const lldb_private::DataExtractor &gpregset,
                                     const lldb_private::DataExtractor &fpregset);

    ~RegisterContextCorePOSIX_mips64();

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
    WriteGPR();

    bool
    WriteFPR();

private:
    uint64_t m_reg[40];
};

#endif // #ifndef liblldb_RegisterContextCorePOSIX_mips64_H_
