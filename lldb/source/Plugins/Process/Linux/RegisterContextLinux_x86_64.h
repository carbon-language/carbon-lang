//===-- RegisterContextLinux_x86_64.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextLinux_x86_64_H_
#define liblldb_RegisterContextLinux_x86_64_H_

#include "RegisterContextLinux.h"

class ProcessMonitor;

class RegisterContextLinux_x86_64
    : public RegisterContextLinux
{
public:
    RegisterContextLinux_x86_64 (lldb_private::Thread &thread,
                                 uint32_t concrete_frame_idx);

    ~RegisterContextLinux_x86_64();

    void
    Invalidate();

    size_t
    GetRegisterCount();

    const lldb::RegisterInfo *
    GetRegisterInfoAtIndex(uint32_t reg);

    size_t
    GetRegisterSetCount();

    const lldb::RegisterSet *
    GetRegisterSet(uint32_t set);

    bool
    ReadRegisterValue(uint32_t reg, lldb_private::Scalar &value);

    bool
    ReadRegisterBytes(uint32_t reg, lldb_private::DataExtractor &data);

    bool
    ReadAllRegisterValues(lldb::DataBufferSP &data_sp);

    bool
    WriteRegisterValue(uint32_t reg, const lldb_private::Scalar &value);

    bool
    WriteRegisterBytes(uint32_t reg, lldb_private::DataExtractor &data,
                       uint32_t data_offset = 0);

    bool
    WriteAllRegisterValues(const lldb::DataBufferSP &data_sp);

    uint32_t
    ConvertRegisterKindToRegisterNumber(uint32_t kind, uint32_t num);

    bool
    HardwareSingleStep(bool enable);

    bool
    UpdateAfterBreakpoint();

    struct GPR
    {
        uint64_t r15;
        uint64_t r14;
        uint64_t r13;
        uint64_t r12;
        uint64_t rbp;
        uint64_t rbx;
        uint64_t r11;
        uint64_t r10;
        uint64_t r9;
        uint64_t r8;
        uint64_t rax;
        uint64_t rcx;
        uint64_t rdx;
        uint64_t rsi;
        uint64_t rdi;
        uint64_t orig_ax;
        uint64_t rip;
        uint64_t cs;
        uint64_t rflags;
        uint64_t rsp;
        uint64_t ss;
        uint64_t fs_base;
        uint64_t gs_base;
        uint64_t ds;
        uint64_t es;
        uint64_t fs;
        uint64_t gs;
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

    struct UserArea
    {
        GPR      regs;          // General purpose registers.
        int32_t  fpvalid;       // True if FPU is being used.
        int32_t  pad0;
        FPU      i387;          // FPU registers.
        uint64_t tsize;         // Text segment size.
        uint64_t dsize;         // Data segment size.
        uint64_t ssize;         // Stack segment size.
        uint64_t start_code;    // VM address of text.
        uint64_t start_stack;   // VM address of stack bottom (top in rsp).
        int64_t  signal;        // Signal causing core dump.
        int32_t  reserved;      // Unused.
        int32_t  pad1;
        uint64_t ar0;           // Location of GPR's.
        FPU*     fpstate;       // Location of FPR's.
        uint64_t magic;         // Identifier for core dumps.
        char     u_comm[32];    // Command causing core dump.
        uint64_t u_debugreg[8]; // Debug registers (DR0 - DR7).
        uint64_t error_code;    // CPU error code.
        uint64_t fault_address; // Control register CR3.
    };

private:
    UserArea user;

    ProcessMonitor &GetMonitor();
};

#endif // #ifndef liblldb_RegisterContextLinux_x86_64_H_
