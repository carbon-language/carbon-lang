//===-- RegisterContext_x86_64.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContext_x86_64_H_
#define liblldb_RegisterContext_x86_64_H_

#include "lldb/Core/Log.h"
#include "RegisterContextPOSIX.h"

#ifdef __FreeBSD__
#include "RegisterContextFreeBSD_x86_64.h"
#endif

#ifdef __linux__
#include "RegisterContextLinux_x86_64.h"
#endif

class ProcessMonitor;

class RegisterContext_x86_64
  : public RegisterContextPOSIX
{
public:
    RegisterContext_x86_64 (lldb_private::Thread &thread,
                                 uint32_t concrete_frame_idx);

    ~RegisterContext_x86_64();

    void
    Invalidate();

    void
    InvalidateAllRegisters();

    size_t
    GetRegisterCount();

    const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex(size_t reg);

    size_t
    GetRegisterSetCount();

    const lldb_private::RegisterSet *
    GetRegisterSet(size_t set);

    static unsigned
    GetRegisterIndexFromOffset(unsigned offset);

    static const char *
    GetRegisterName(unsigned reg);

    virtual bool
    ReadRegister(const lldb_private::RegisterInfo *reg_info,
                 lldb_private::RegisterValue &value);

    bool
    ReadAllRegisterValues(lldb::DataBufferSP &data_sp);

    virtual bool
    WriteRegister(const lldb_private::RegisterInfo *reg_info,
                  const lldb_private::RegisterValue &value);

    bool
    WriteAllRegisterValues(const lldb::DataBufferSP &data_sp);

    uint32_t
    ConvertRegisterKindToRegisterNumber(uint32_t kind, uint32_t num);

    bool
    HardwareSingleStep(bool enable);

    bool
    UpdateAfterBreakpoint();

    //---------------------------------------------------------------------------
    // Generic floating-point registers
    //---------------------------------------------------------------------------
    struct MMSReg
    {
        uint8_t bytes[10];
        uint8_t pad[6];
    };

    struct XMMReg
    {
        uint8_t bytes[16]; // 128-bits for each XMM register
    };

    struct FXSAVE
    {
        uint16_t fcw;
        uint16_t fsw;
        uint16_t ftw;
        uint16_t fop;
        uint64_t ip;
        uint64_t dp;
        uint32_t mxcsr;
        uint32_t mxcsrmask;
        MMSReg   stmm[8];
        XMMReg   xmm[16];
        uint32_t padding[24];
    };

    //---------------------------------------------------------------------------
    // Extended floating-point registers
    //---------------------------------------------------------------------------
    struct YMMHReg
    {
        uint8_t  bytes[16];     // 16 * 8 bits for the high bytes of each YMM register
    };

    struct YMMReg
    {
        uint8_t  bytes[32];     // 16 * 16 bits for each YMM register
    };

    struct YMM
    {
        YMMReg   ymm[16];       // assembled from ymmh and xmm registers
    };

    struct XSAVE_HDR
    {
        uint64_t  xstate_bv;    // OS enabled xstate mask to determine the extended states supported by the processor
        uint64_t  reserved1[2];
        uint64_t  reserved2[5];
    } __attribute__((packed));

    // x86 extensions to FXSAVE (i.e. for AVX processors) 
    struct XSAVE 
    {
        FXSAVE    i387;         // floating point registers typical in i387_fxsave_struct
        XSAVE_HDR header;       // The xsave_hdr_struct can be used to determine if the following extensions are usable
        YMMHReg   ymmh[16];     // High 16 bytes of each of 16 YMM registers (the low bytes are in FXSAVE.xmm for compatibility with SSE)
        // Slot any extensions to the register file here
    } __attribute__((packed, aligned (64)));

    struct IOVEC
    {
        void    *iov_base;      // pointer to XSAVE
        size_t   iov_len;       // sizeof(XSAVE)
    };

    //---------------------------------------------------------------------------
    // Note: prefer kernel definitions over user-land
    //---------------------------------------------------------------------------
    enum FPRType
    {
        eNotValid = 0,
        eFSAVE,  // TODO
        eFXSAVE,
        eSOFT,   // TODO
        eXSAVE
    };

    // Floating-point registers
    struct FPR
    {
        // Thread state for the floating-point unit of the processor read by ptrace.
        union XSTATE {
            FXSAVE   fxsave;    // Generic floating-point registers.
            XSAVE    xsave;     // x86 extended processor state.
        } xstate;

        YMM      ymm_set;       // Copy of ymmh and xmm register halves.
    };

    struct UserArea
    {
        GPR      regs;          // General purpose registers.
        FPRType  fpr_type;      // Determines the type of data stored by union FPR, if any.
        int32_t  pad0;
        FPR      i387;          // Floating point registers.
        uint64_t tsize;         // Text segment size.
        uint64_t dsize;         // Data segment size.
        uint64_t ssize;         // Stack segment size.
        uint64_t start_code;    // VM address of text.
        uint64_t start_stack;   // VM address of stack bottom (top in rsp).
        int64_t  signal;        // Signal causing core dump.
        int32_t  reserved;      // Unused.
        int32_t  pad1;
        uint64_t ar0;           // Location of GPR's.
        FPR*     fpstate;       // Location of FPR's.
        uint64_t magic;         // Identifier for core dumps.
        char     u_comm[32];    // Command causing core dump.
        uint64_t u_debugreg[8]; // Debug registers (DR0 - DR7).
        uint64_t error_code;    // CPU error code.
        uint64_t fault_address; // Control register CR3.
        IOVEC    iovec;         // wrapper for xsave
    };

protected:
    // Determines if an extended register set is supported on the processor running the inferior process.
    virtual bool
    IsRegisterSetAvailable(size_t set_index);

private:
    UserArea user;

    ProcessMonitor &GetMonitor();
    lldb::ByteOrder GetByteOrder();

    static bool IsFPR(unsigned reg, FPRType fpr_type);

    bool ReadGPR();
    bool ReadFPR();

    bool WriteGPR();
    bool WriteFPR();
};

#endif // #ifndef liblldb_RegisterContext_x86_64_H_
