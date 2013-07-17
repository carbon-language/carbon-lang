//===-- RegisterContextCoreFreeBSD_x86_64.h ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextCoreFreeBSD_x86_64_H_
#define liblldb_RegisterContextCoreFreeBSD_x86_64_H_

#include "Plugins/Process/POSIX/RegisterContextFreeBSD_x86_64.h"

using namespace lldb_private;

class RegisterContextCoreFreeBSD_x86_64: public RegisterContextFreeBSD_x86_64
{
public:
    RegisterContextCoreFreeBSD_x86_64 (Thread &thread, const DataExtractor &gpregset,
                                       const DataExtractor &fpregset);

    ~RegisterContextCoreFreeBSD_x86_64();

    virtual bool
    ReadRegister(const RegisterInfo *reg_info, RegisterValue &value);

    bool
    ReadAllRegisterValues(lldb::DataBufferSP &data_sp);

    virtual bool
    WriteRegister(const RegisterInfo *reg_info, const RegisterValue &value);

    bool
    WriteAllRegisterValues(const lldb::DataBufferSP &data_sp);

    bool
    HardwareSingleStep(bool enable);

    bool
    UpdateAfterBreakpoint();

private:
    uint8_t *m_gpregset;
};

#endif // #ifndef liblldb_RegisterContextCoreFreeBSD_x86_64_H_
