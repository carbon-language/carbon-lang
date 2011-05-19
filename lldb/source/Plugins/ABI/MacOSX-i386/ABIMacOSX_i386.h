//===-- ABIMacOSX_i386.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABIMacOSX_i386_h_
#define liblldb_ABIMacOSX_i386_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ABI.h"
#include "lldb/Core/Value.h"
    
class ABIMacOSX_i386 :
    public lldb_private::ABI
{
public:

    enum
    {
        gcc_eax = 0,
        gcc_ecx,
        gcc_edx,
        gcc_ebx,
        gcc_ebp,
        gcc_esp,
        gcc_esi,
        gcc_edi,
        gcc_eip,
        gcc_eflags
    };
    
    enum
    {
        dwarf_eax = 0,
        dwarf_ecx,
        dwarf_edx,
        dwarf_ebx,
        dwarf_esp,
        dwarf_ebp,
        dwarf_esi,
        dwarf_edi,
        dwarf_eip,
        dwarf_eflags,
        dwarf_stmm0 = 11,
        dwarf_stmm1,
        dwarf_stmm2,
        dwarf_stmm3,
        dwarf_stmm4,
        dwarf_stmm5,
        dwarf_stmm6,
        dwarf_stmm7,
        dwarf_xmm0 = 21,
        dwarf_xmm1,
        dwarf_xmm2,
        dwarf_xmm3,
        dwarf_xmm4,
        dwarf_xmm5,
        dwarf_xmm6,
        dwarf_xmm7,
        dwarf_ymm0 = dwarf_xmm0,
        dwarf_ymm1 = dwarf_xmm1,
        dwarf_ymm2 = dwarf_xmm2,
        dwarf_ymm3 = dwarf_xmm3,
        dwarf_ymm4 = dwarf_xmm4,
        dwarf_ymm5 = dwarf_xmm5,
        dwarf_ymm6 = dwarf_xmm6,
        dwarf_ymm7 = dwarf_xmm7
    };
    
    enum
    {
        gdb_eax        =  0,
        gdb_ecx        =  1,
        gdb_edx        =  2,
        gdb_ebx        =  3,
        gdb_esp        =  4,
        gdb_ebp        =  5,
        gdb_esi        =  6,
        gdb_edi        =  7,
        gdb_eip        =  8,
        gdb_eflags     =  9,
        gdb_cs         = 10,
        gdb_ss         = 11,
        gdb_ds         = 12,
        gdb_es         = 13,
        gdb_fs         = 14,
        gdb_gs         = 15,
        gdb_stmm0      = 16,
        gdb_stmm1      = 17,
        gdb_stmm2      = 18,
        gdb_stmm3      = 19,
        gdb_stmm4      = 20,
        gdb_stmm5      = 21,
        gdb_stmm6      = 22,
        gdb_stmm7      = 23,
        gdb_fctrl      = 24,    gdb_fcw     = gdb_fctrl,
        gdb_fstat      = 25,    gdb_fsw     = gdb_fstat,
        gdb_ftag       = 26,    gdb_ftw     = gdb_ftag,
        gdb_fiseg      = 27,    gdb_fpu_cs  = gdb_fiseg,
        gdb_fioff      = 28,    gdb_ip      = gdb_fioff,
        gdb_foseg      = 29,    gdb_fpu_ds  = gdb_foseg,
        gdb_fooff      = 30,    gdb_dp      = gdb_fooff,
        gdb_fop        = 31,
        gdb_xmm0       = 32,
        gdb_xmm1       = 33,
        gdb_xmm2       = 34,
        gdb_xmm3       = 35,
        gdb_xmm4       = 36,
        gdb_xmm5       = 37,
        gdb_xmm6       = 38,
        gdb_xmm7       = 39,
        gdb_mxcsr      = 40,
        gdb_mm0        = 41,
        gdb_mm1        = 42,
        gdb_mm2        = 43,
        gdb_mm3        = 44,
        gdb_mm4        = 45,
        gdb_mm5        = 46,
        gdb_mm6        = 47,
        gdb_mm7        = 48,
        gdb_ymm0       = gdb_xmm0,
        gdb_ymm1       = gdb_xmm1,
        gdb_ymm2       = gdb_xmm2,
        gdb_ymm3       = gdb_xmm3,
        gdb_ymm4       = gdb_xmm4,
        gdb_ymm5       = gdb_xmm5,
        gdb_ymm6       = gdb_xmm6,
        gdb_ymm7       = gdb_xmm7
    };

    ~ABIMacOSX_i386() { }
    
    virtual size_t 
    GetRedZoneSize () const;
    
    virtual bool
    PrepareTrivialCall (lldb_private::Thread &thread, 
                        lldb::addr_t sp,
                        lldb::addr_t func_addr,
                        lldb::addr_t return_addr, 
                        lldb::addr_t *arg1_ptr = NULL,
                        lldb::addr_t *arg2_ptr = NULL,
                        lldb::addr_t *arg3_ptr = NULL,
                        lldb::addr_t *arg4_ptr = NULL,
                        lldb::addr_t *arg5_ptr = NULL,
                        lldb::addr_t *arg6_ptr = NULL) const;
    
    virtual bool
    PrepareNormalCall (lldb_private::Thread &thread,
                       lldb::addr_t sp,
                       lldb::addr_t func_addr,
                       lldb::addr_t return_addr,
                       lldb_private::ValueList &args) const;
    
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
    ABIMacOSX_i386() : lldb_private::ABI() { } // Call CreateInstance instead.
};


#endif  // liblldb_ABI_h_
