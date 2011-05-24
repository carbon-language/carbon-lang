//===-- ABISysV_x86_64.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABISysV_x86_64_h_
#define liblldb_ABISysV_x86_64_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ABI.h"

class ABISysV_x86_64 :
    public lldb_private::ABI
{
public:
    
    enum gcc_dwarf_regnums
    {
        gcc_dwarf_rax = 0,
        gcc_dwarf_rdx,
        gcc_dwarf_rcx,
        gcc_dwarf_rbx,
        gcc_dwarf_rsi,
        gcc_dwarf_rdi,
        gcc_dwarf_rbp,
        gcc_dwarf_rsp,
        gcc_dwarf_r8,
        gcc_dwarf_r9,
        gcc_dwarf_r10,
        gcc_dwarf_r11,
        gcc_dwarf_r12,
        gcc_dwarf_r13,
        gcc_dwarf_r14,
        gcc_dwarf_r15,
        gcc_dwarf_rip,
        gcc_dwarf_xmm0,
        gcc_dwarf_xmm1,
        gcc_dwarf_xmm2,
        gcc_dwarf_xmm3,
        gcc_dwarf_xmm4,
        gcc_dwarf_xmm5,
        gcc_dwarf_xmm6,
        gcc_dwarf_xmm7,
        gcc_dwarf_xmm8,
        gcc_dwarf_xmm9,
        gcc_dwarf_xmm10,
        gcc_dwarf_xmm11,
        gcc_dwarf_xmm12,
        gcc_dwarf_xmm13,
        gcc_dwarf_xmm14,
        gcc_dwarf_xmm15,
        gcc_dwarf_stmm0,
        gcc_dwarf_stmm1,
        gcc_dwarf_stmm2,
        gcc_dwarf_stmm3,
        gcc_dwarf_stmm4,
        gcc_dwarf_stmm5,
        gcc_dwarf_stmm6,
        gcc_dwarf_stmm7,
        gcc_dwarf_ymm0 = gcc_dwarf_xmm0,
        gcc_dwarf_ymm1 = gcc_dwarf_xmm1,
        gcc_dwarf_ymm2 = gcc_dwarf_xmm2,
        gcc_dwarf_ymm3 = gcc_dwarf_xmm3,
        gcc_dwarf_ymm4 = gcc_dwarf_xmm4,
        gcc_dwarf_ymm5 = gcc_dwarf_xmm5,
        gcc_dwarf_ymm6 = gcc_dwarf_xmm6,
        gcc_dwarf_ymm7 = gcc_dwarf_xmm7,
        gcc_dwarf_ymm8 = gcc_dwarf_xmm8,
        gcc_dwarf_ymm9 = gcc_dwarf_xmm9,
        gcc_dwarf_ymm10 = gcc_dwarf_xmm10,
        gcc_dwarf_ymm11 = gcc_dwarf_xmm11,
        gcc_dwarf_ymm12 = gcc_dwarf_xmm12,
        gcc_dwarf_ymm13 = gcc_dwarf_xmm13,
        gcc_dwarf_ymm14 = gcc_dwarf_xmm14,
        gcc_dwarf_ymm15 = gcc_dwarf_xmm15
    };
    
    enum gdb_regnums
    {
        gdb_rax     =   0,
        gdb_rbx     =   1,
        gdb_rcx     =   2,
        gdb_rdx     =   3,
        gdb_rsi     =   4,
        gdb_rdi     =   5,
        gdb_rbp     =   6,
        gdb_rsp     =   7,
        gdb_r8      =   8,
        gdb_r9      =   9,
        gdb_r10     =  10,
        gdb_r11     =  11,
        gdb_r12     =  12,
        gdb_r13     =  13,
        gdb_r14     =  14,
        gdb_r15     =  15,
        gdb_rip     =  16,
        gdb_rflags  =  17,
        gdb_cs      =  18,
        gdb_ss      =  19,
        gdb_ds      =  20,
        gdb_es      =  21,
        gdb_fs      =  22,
        gdb_gs      =  23,
        gdb_stmm0   =  24,
        gdb_stmm1   =  25,
        gdb_stmm2   =  26,
        gdb_stmm3   =  27,
        gdb_stmm4   =  28,
        gdb_stmm5   =  29,
        gdb_stmm6   =  30,
        gdb_stmm7   =  31,
        gdb_fctrl   =  32,  gdb_fcw = gdb_fctrl,
        gdb_fstat   =  33,  gdb_fsw = gdb_fstat,
        gdb_ftag    =  34,  gdb_ftw = gdb_ftag,
        gdb_fiseg   =  35,  gdb_fpu_cs  = gdb_fiseg,
        gdb_fioff   =  36,  gdb_ip  = gdb_fioff,
        gdb_foseg   =  37,  gdb_fpu_ds  = gdb_foseg,
        gdb_fooff   =  38,  gdb_dp  = gdb_fooff,
        gdb_fop     =  39,
        gdb_xmm0    =  40,
        gdb_xmm1    =  41,
        gdb_xmm2    =  42,
        gdb_xmm3    =  43,
        gdb_xmm4    =  44,
        gdb_xmm5    =  45,
        gdb_xmm6    =  46,
        gdb_xmm7    =  47,
        gdb_xmm8    =  48,
        gdb_xmm9    =  49,
        gdb_xmm10   =  50,
        gdb_xmm11   =  51,
        gdb_xmm12   =  52,
        gdb_xmm13   =  53,
        gdb_xmm14   =  54,
        gdb_xmm15   =  55,
        gdb_mxcsr   =  56,
        gdb_ymm0    =  gdb_xmm0,
        gdb_ymm1    =  gdb_xmm1,
        gdb_ymm2    =  gdb_xmm2,
        gdb_ymm3    =  gdb_xmm3,
        gdb_ymm4    =  gdb_xmm4,
        gdb_ymm5    =  gdb_xmm5,
        gdb_ymm6    =  gdb_xmm6,
        gdb_ymm7    =  gdb_xmm7,
        gdb_ymm8    =  gdb_xmm8,
        gdb_ymm9    =  gdb_xmm9,
        gdb_ymm10   =  gdb_xmm10,
        gdb_ymm11   =  gdb_xmm11,
        gdb_ymm12   =  gdb_xmm12,
        gdb_ymm13   =  gdb_xmm13,
        gdb_ymm14   =  gdb_xmm14,
        gdb_ymm15   =  gdb_xmm15
    };

   ~ABISysV_x86_64()
    {
    }

    virtual size_t
    GetRedZoneSize () const;

    virtual bool
    PrepareTrivialCall (lldb_private::Thread &thread, 
                        lldb::addr_t sp,
                        lldb::addr_t functionAddress,
                        lldb::addr_t returnAddress, 
                        lldb::addr_t *arg1_ptr = NULL,
                        lldb::addr_t *arg2_ptr = NULL,
                        lldb::addr_t *arg3_ptr = NULL,
                        lldb::addr_t *arg4_ptr = NULL,
                        lldb::addr_t *arg5_ptr = NULL,
                        lldb::addr_t *arg6_ptr = NULL) const;
    
    virtual bool
    GetArgumentValues (lldb_private::Thread &thread,
                       lldb_private::ValueList &values) const;
    
    virtual bool
    GetReturnValue (lldb_private::Thread &thread,
                    lldb_private::Value &value) const;
    
    virtual bool
    CreateFunctionEntryUnwindPlan (lldb_private::UnwindPlan &unwind_plan);
    
    virtual bool
    CreateDefaultUnwindPlan (lldb_private::UnwindPlan &unwind_plan);
        
    virtual bool
    RegisterIsVolatile (const lldb_private::RegisterInfo *reg_info);
    
    virtual bool
    StackUsesFrames ()
    {
        return true;
    }
    
    virtual bool
    CallFrameAddressIsValid (lldb::addr_t cfa)
    {
        // Make sure the stack call frame addresses are are 8 byte aligned
        if (cfa & (8ull - 1ull))
            return false;   // Not 8 byte aligned
        if (cfa == 0)
            return false;   // Zero is not a valid stack address
        return true;
    }
    
    virtual bool
    CodeAddressIsValid (lldb::addr_t pc)
    {
        // We have a 64 bit address space, so anything is valid as opcodes
        // aren't fixed width...
        return true;
    }
    
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static lldb::ABISP
    CreateInstance (const lldb_private::ArchSpec &arch);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();

    virtual const char *
    GetShortPluginName();

    virtual uint32_t
    GetPluginVersion();

protected:
    
    bool
    RegisterIsCalleeSaved (const lldb_private::RegisterInfo *reg_info);

private:
    ABISysV_x86_64() : lldb_private::ABI() { } // Call CreateInstance instead.
};

#endif  // liblldb_ABI_h_
