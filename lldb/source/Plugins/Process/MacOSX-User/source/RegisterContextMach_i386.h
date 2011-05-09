//===-- RegisterContextMach_i386.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextMach_i386_h_
#define liblldb_RegisterContextMach_i386_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/RegisterContext.h"

class RegisterContextMach_i386 : public lldb_private::RegisterContext
{
public:

    RegisterContextMach_i386(lldb_private::Thread &thread,
                             uint32_t concrete_frame_idx);

    virtual
    ~RegisterContextMach_i386();

    virtual void
    InvalidateAllRegisters ();

    virtual size_t
    GetRegisterCount ();

    virtual const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex (uint32_t reg);

    virtual size_t
    GetRegisterSetCount ();

    virtual const lldb_private::RegisterSet *
    GetRegisterSet (uint32_t set);

    virtual bool
    ReadRegister (const lldb_private::RegisterInfo *reg_info, lldb_private::RegisterValue &value);

    virtual bool
    WriteRegister (const lldb_private::RegisterInfo *reg_info, const lldb_private::RegisterValue &value);
    
    virtual bool
    ReadAllRegisterValues (lldb::DataBufferSP &data_sp);

    virtual bool
    WriteAllRegisterValues (const lldb::DataBufferSP &data_sp);

    virtual uint32_t
    ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num);

    virtual bool
    HardwareSingleStep (bool enable);

    struct GPR
    {
        uint32_t eax;
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
        uint32_t edi;
        uint32_t esi;
        uint32_t ebp;
        uint32_t esp;
        uint32_t ss;
        uint32_t eflags;
        uint32_t eip;
        uint32_t cs;
        uint32_t ds;
        uint32_t es;
        uint32_t fs;
        uint32_t gs;
    };

    struct MMSReg
    {
        uint8_t bytes[10];
        uint8_t pad[6];
    };

    struct XMMReg
    {
        uint8_t bytes[16];
    };

    struct FPU
    {
        uint32_t    pad[2];
        uint16_t    fcw;
        uint16_t    fsw;
        uint8_t     ftw;
        uint8_t     pad1;
        uint16_t    fop;
        uint32_t    ip;
        uint16_t    cs;
        uint16_t    pad2;
        uint32_t    dp;
        uint16_t    ds;
        uint16_t    pad3;
        uint32_t    mxcsr;
        uint32_t    mxcsrmask;
        MMSReg      stmm[8];
        XMMReg      xmm[8];
        uint8_t     pad4[14*16];
        int         pad5;
    };

    struct EXC
    {
        uint32_t trapno;
        uint32_t err;
        uint32_t faultvaddr;
    };

protected:

    enum
    {
        GPRRegSet = 1,
        FPURegSet = 2,
        EXCRegSet = 3
    };

    enum
    {
        GPRWordCount = sizeof(GPR)/sizeof(uint32_t),
        FPUWordCount = sizeof(FPU)/sizeof(uint32_t),
        EXCWordCount = sizeof(EXC)/sizeof(uint32_t)
    };

    enum
    {
        Read = 0,
        Write = 1,
        kNumErrors = 2
    };

    GPR gpr;
    FPU fpu;
    EXC exc;
    kern_return_t gpr_errs[2]; // Read/Write errors
    kern_return_t fpu_errs[2]; // Read/Write errors
    kern_return_t exc_errs[2]; // Read/Write errors

    void
    InvalidateAllRegisterStates()
    {
        SetError (GPRRegSet, Read, -1);
        SetError (FPURegSet, Read, -1);
        SetError (EXCRegSet, Read, -1);
    }

    kern_return_t
    GetError (int flavor, uint32_t err_idx) const
    {
        if (err_idx < kNumErrors)
        {
            switch (flavor)
            {
            // When getting all errors, just OR all values together to see if
            // we got any kind of error.
            case GPRRegSet:    return gpr_errs[err_idx];
            case FPURegSet:    return fpu_errs[err_idx];
            case EXCRegSet:    return exc_errs[err_idx];
            default: break;
            }
        }
        return -1;
    }

    bool
    SetError (int flavor, uint32_t err_idx, kern_return_t err)
    {
        if (err_idx < kNumErrors)
        {
            switch (flavor)
            {
            case GPRRegSet:
                gpr_errs[err_idx] = err;
                return true;

            case FPURegSet:
                fpu_errs[err_idx] = err;
                return true;

            case EXCRegSet:
                exc_errs[err_idx] = err;
                return true;

            default: break;
            }
        }
        return false;
    }

    bool
    RegisterSetIsCached (int set) const
    {
        return GetError(set, Read) == KERN_SUCCESS;
    }

    void
    LogGPR (lldb_private::Log *log, const char *title);

    kern_return_t
    ReadGPR (bool force);

    kern_return_t
    ReadFPU (bool force);

    kern_return_t
    ReadEXC (bool force);

    kern_return_t
    WriteGPR ();

    kern_return_t
    WriteFPU ();

    kern_return_t
    WriteEXC ();

    kern_return_t
    ReadRegisterSet (uint32_t set, bool force);

    kern_return_t
    WriteRegisterSet (uint32_t set);

    static uint32_t
    GetRegisterNumber (uint32_t reg_kind, uint32_t reg_num);

    static int
    GetSetForNativeRegNum (int reg_num);

    static size_t
    GetRegisterInfosCount ();

    static const lldb_private::RegisterInfo *
    GetRegisterInfos ();
};

#endif  // liblldb_RegisterContextMach_i386_h_
