//===----------------------- ABISysV_i386.cpp -------------------*- C++ -*-===//
//
//                   The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source License.
// See LICENSE.TXT for details.
//===----------------------------------------------------------------------===//

#include "ABISysV_i386.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectRegister.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"

using namespace lldb;
using namespace lldb_private;



//   This source file uses the following document as a reference:
//====================================================================
//             System V Application Binary Interface
//    Intel386 Architecture Processor Supplement, Version 1.0
//                         Edited by
//      H.J. Lu, David L Kreitzer, Milind Girkar, Zia Ansari
//
//                        (Based on
//           System V Application Binary Interface,
//          AMD64 Architecture Processor Supplement,
//                         Edited by
//     H.J. Lu, Michael Matz, Milind Girkar, Jan Hubicka,
//               Andreas Jaeger, Mark Mitchell)
//
//                     February 3, 2015
//====================================================================



// DWARF Register Number Mapping
// See Table 2.14 of the reference document (specified on top of this file)
// Comment: Table 2.14 is followed till 'mm' entries.
// After that, all entries are ignored here.

enum gcc_dwarf_regnums
{
    gcc_dwarf_eax = 0,
    gcc_dwarf_ecx,
    gcc_dwarf_edx,
    gcc_dwarf_ebx,
    gcc_dwarf_esp,
    gcc_dwarf_ebp,
    gcc_dwarf_esi,
    gcc_dwarf_edi,
    gcc_dwarf_eip,
    gcc_dwarf_eflags,

    gcc_dwarf_st0 = 11,
    gcc_dwarf_st1,
    gcc_dwarf_st2,
    gcc_dwarf_st3,
    gcc_dwarf_st4,
    gcc_dwarf_st5,
    gcc_dwarf_st6,
    gcc_dwarf_st7,

    gcc_dwarf_xmm0 = 21,
    gcc_dwarf_xmm1,
    gcc_dwarf_xmm2,
    gcc_dwarf_xmm3,
    gcc_dwarf_xmm4,
    gcc_dwarf_xmm5,
    gcc_dwarf_xmm6,
    gcc_dwarf_xmm7,
    gcc_dwarf_ymm0 = gcc_dwarf_xmm0,
    gcc_dwarf_ymm1 = gcc_dwarf_xmm1,
    gcc_dwarf_ymm2 = gcc_dwarf_xmm2,
    gcc_dwarf_ymm3 = gcc_dwarf_xmm3,
    gcc_dwarf_ymm4 = gcc_dwarf_xmm4,
    gcc_dwarf_ymm5 = gcc_dwarf_xmm5,
    gcc_dwarf_ymm6 = gcc_dwarf_xmm6,
    gcc_dwarf_ymm7 = gcc_dwarf_xmm7,

    gcc_dwarf_mm0 = 29,
    gcc_dwarf_mm1,
    gcc_dwarf_mm2,
    gcc_dwarf_mm3,
    gcc_dwarf_mm4,
    gcc_dwarf_mm5,
    gcc_dwarf_mm6,
    gcc_dwarf_mm7
};


enum gdb_regnums
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
    gdb_st0        = 16,
    gdb_st1        = 17,
    gdb_st2        = 18,
    gdb_st3        = 19,
    gdb_st4        = 20,
    gdb_st5        = 21,
    gdb_st6        = 22,
    gdb_st7        = 23,
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


static RegisterInfo g_register_infos[] =
{
  //  NAME      ALT         SZ OFF ENCODING         FORMAT                  COMPILER                 DWARF                      GENERIC                  GDB                   LLDB NATIVE            VALUE REGS    INVALIDATE REGS
  //  ======    =======     == === =============    ============          ===================== =====================    ============================ ====================  ======================    ==========    ===============
    { "eax",    nullptr,    4,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_eax       , gcc_dwarf_eax           , LLDB_INVALID_REGNUM       , gdb_eax            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ebx"   , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_ebx       , gcc_dwarf_ebx           , LLDB_INVALID_REGNUM       , gdb_ebx            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ecx"   , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_ecx       , gcc_dwarf_ecx           , LLDB_REGNUM_GENERIC_ARG4  , gdb_ecx            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "edx"   , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_edx       , gcc_dwarf_edx           , LLDB_REGNUM_GENERIC_ARG3  , gdb_edx            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "esi"   , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_esi       , gcc_dwarf_esi           , LLDB_REGNUM_GENERIC_ARG2  , gdb_esi            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "edi"   , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_edi       , gcc_dwarf_edi           , LLDB_REGNUM_GENERIC_ARG1  , gdb_edi            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ebp"   , "fp",       4,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_ebp       , gcc_dwarf_ebp           , LLDB_REGNUM_GENERIC_FP    , gdb_ebp            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "esp"   , "sp",       4,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_esp       , gcc_dwarf_esp           , LLDB_REGNUM_GENERIC_SP    , gdb_esp            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "eip"   , "pc",       4,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_eip       , gcc_dwarf_eip           , LLDB_REGNUM_GENERIC_PC    , gdb_eip            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "eflags", nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_REGNUM_GENERIC_FLAGS , gdb_eflags         , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "cs"    , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_cs             , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ss"    , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_ss             , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ds"    , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_ds             , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "es"    , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_es             , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "fs"    , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_fs             , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "gs"    , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_gs             , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "st0"   , nullptr,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_st0           , LLDB_INVALID_REGNUM       , gdb_st0            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "st1"   , nullptr,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_st1           , LLDB_INVALID_REGNUM       , gdb_st1            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "st2"   , nullptr,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_st2           , LLDB_INVALID_REGNUM       , gdb_st2            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "st3"   , nullptr,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_st3           , LLDB_INVALID_REGNUM       , gdb_st3            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "st4"   , nullptr,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_st4           , LLDB_INVALID_REGNUM       , gdb_st4            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "st5"   , nullptr,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_st5           , LLDB_INVALID_REGNUM       , gdb_st5            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "st6"   , nullptr,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_st6           , LLDB_INVALID_REGNUM       , gdb_st6            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "st7"   , nullptr,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_st7           , LLDB_INVALID_REGNUM       , gdb_st7            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "fctrl" , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_fctrl          , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "fstat" , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_fstat          , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ftag"  , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_ftag           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "fiseg" , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_fiseg          , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "fioff" , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_fioff          , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "foseg" , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_foseg          , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "fooff" , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_fooff          , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "fop"   , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_fop            , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "xmm0"  , nullptr,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_xmm0          , LLDB_INVALID_REGNUM       , gdb_xmm0           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "xmm1"  , nullptr,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_xmm1          , LLDB_INVALID_REGNUM       , gdb_xmm1           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "xmm2"  , nullptr,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_xmm2          , LLDB_INVALID_REGNUM       , gdb_xmm2           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "xmm3"  , nullptr,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_xmm3          , LLDB_INVALID_REGNUM       , gdb_xmm3           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "xmm4"  , nullptr,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_xmm4          , LLDB_INVALID_REGNUM       , gdb_xmm4           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "xmm5"  , nullptr,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_xmm5          , LLDB_INVALID_REGNUM       , gdb_xmm5           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "xmm6"  , nullptr,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_xmm6          , LLDB_INVALID_REGNUM       , gdb_xmm6           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "xmm7"  , nullptr,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_xmm7          , LLDB_INVALID_REGNUM       , gdb_xmm7           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "mxcsr" , nullptr,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM     , LLDB_INVALID_REGNUM       , gdb_mxcsr          , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ymm0"  , nullptr,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_ymm0          , LLDB_INVALID_REGNUM       , gdb_ymm0           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ymm1"  , nullptr,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_ymm1          , LLDB_INVALID_REGNUM       , gdb_ymm1           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ymm2"  , nullptr,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_ymm2          , LLDB_INVALID_REGNUM       , gdb_ymm2           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ymm3"  , nullptr,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_ymm3          , LLDB_INVALID_REGNUM       , gdb_ymm3           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ymm4"  , nullptr,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_ymm4          , LLDB_INVALID_REGNUM       , gdb_ymm4           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ymm5"  , nullptr,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_ymm5          , LLDB_INVALID_REGNUM       , gdb_ymm5           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ymm6"  , nullptr,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_ymm6          , LLDB_INVALID_REGNUM       , gdb_ymm6           , LLDB_INVALID_REGNUM },      nullptr,        nullptr},
    { "ymm7"  , nullptr,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , gcc_dwarf_ymm7          , LLDB_INVALID_REGNUM       , gdb_ymm7           , LLDB_INVALID_REGNUM },      nullptr,        nullptr}
};

static const uint32_t k_num_register_infos = llvm::array_lengthof(g_register_infos);
static bool g_register_info_names_constified = false;

const lldb_private::RegisterInfo *
ABISysV_i386::GetRegisterInfoArray (uint32_t &count)
{
    // Make the C-string names and alt_names for the register infos into const
    // C-string values by having the ConstString unique the names in the global
    // constant C-string pool.
    if (!g_register_info_names_constified)
    {
        g_register_info_names_constified = true;
        for (uint32_t i=0; i<k_num_register_infos; ++i)
        {
            if (g_register_infos[i].name)
                g_register_infos[i].name = ConstString(g_register_infos[i].name).GetCString();
            if (g_register_infos[i].alt_name)
                g_register_infos[i].alt_name = ConstString(g_register_infos[i].alt_name).GetCString();
        }
    }
    count = k_num_register_infos;
    return g_register_infos;
}


//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
ABISP
ABISysV_i386::CreateInstance (const ArchSpec &arch)
{
    static ABISP g_abi_sp;
    if ((arch.GetTriple().getArch() == llvm::Triple::x86) &&
         arch.GetTriple().isOSLinux())
    {
        if (!g_abi_sp)
            g_abi_sp.reset (new ABISysV_i386);
        return g_abi_sp;
    }
    return ABISP();
}

bool
ABISysV_i386::PrepareTrivialCall (Thread &thread,
                                    addr_t sp,
                                    addr_t func_addr,
                                    addr_t return_addr,
                                    llvm::ArrayRef<addr_t> args) const
{
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();

    if (!reg_ctx)
        return false;

    uint32_t pc_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    uint32_t sp_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);

    // While using register info to write a register value to memory, the register info
    // just needs to have the correct size of a 32 bit register, the actual register it
    // pertains to is not important, just the size needs to be correct.
    // "eax" is used here for this purpose.
    const RegisterInfo *reg_info_32 = reg_ctx->GetRegisterInfoByName("eax");
    if (!reg_info_32)
        return false; // TODO this should actually never happen

    Error error;
    RegisterValue reg_value;

    // Make room for the argument(s) on the stack
    sp -= 4 * args.size();

    // SP Alignment
    sp &= ~(16ull-1ull); // 16-byte alignment

    // Write arguments onto the stack
    addr_t arg_pos = sp;
    for (addr_t arg : args)
    {
        reg_value.SetUInt32(arg);
        error = reg_ctx->WriteRegisterValueToMemory (reg_info_32,
                                                     arg_pos,
                                                     reg_info_32->byte_size,
                                                     reg_value);
        if (error.Fail())
            return false;
        arg_pos += 4;
    }

    // The return address is pushed onto the stack
    sp -= 4;
    reg_value.SetUInt32(return_addr);
    error = reg_ctx->WriteRegisterValueToMemory (reg_info_32,
                                                 sp,
                                                 reg_info_32->byte_size,
                                                 reg_value);
    if (error.Fail())
        return false;

    // Setting %esp to the actual stack value.
    if (!reg_ctx->WriteRegisterFromUnsigned (sp_reg_num, sp))
        return false;

    // Setting %eip to the address of the called function.
    if (!reg_ctx->WriteRegisterFromUnsigned (pc_reg_num, func_addr))
        return false;

    return true;
}


static bool
ReadIntegerArgument (Scalar           &scalar,
                     unsigned int     bit_width,
                     bool             is_signed,
                     Process          *process,
                     addr_t           &current_stack_argument)
{
    uint32_t byte_size = (bit_width + (8-1))/8;
    Error error;

    if (!process)
        return false;

    if (process->ReadScalarIntegerFromMemory(current_stack_argument, byte_size, is_signed, scalar, error))
    {
        current_stack_argument += byte_size;
        return true;
    }
    return false;
}


bool
ABISysV_i386::GetArgumentValues (Thread &thread,
                                   ValueList &values) const
{
    unsigned int num_values = values.GetSize();
    unsigned int value_index;

    RegisterContext *reg_ctx = thread.GetRegisterContext().get();

    if (!reg_ctx)
        return false;

    // Get pointer to the first stack argument
    addr_t sp = reg_ctx->GetSP(0);
    if (!sp)
        return false;

    addr_t current_stack_argument = sp + 4; // jump over return address

    for (value_index = 0;
         value_index < num_values;
         ++value_index)
    {
        Value *value = values.GetValueAtIndex(value_index);

        if (!value)
            return false;

        // Currently: Support for extracting values with Clang QualTypes only.
        ClangASTType clang_type (value->GetClangType());
        if (clang_type)
        {
            bool is_signed;
            if (clang_type.IsIntegerType (is_signed))
            {
                ReadIntegerArgument(value->GetScalar(),
                                    clang_type.GetBitSize(&thread),
                                    is_signed,
                                    thread.GetProcess().get(),
                                    current_stack_argument);
            }
            else if (clang_type.IsPointerType())
            {
                ReadIntegerArgument(value->GetScalar(),
                                    clang_type.GetBitSize(&thread),
                                    false,
                                    thread.GetProcess().get(),
                                    current_stack_argument);
            }
        }
    }
    return true;
}



Error
ABISysV_i386::SetReturnValueObject(lldb::StackFrameSP &frame_sp, lldb::ValueObjectSP &new_value_sp)
{
    Error error;
    //ToDo: Yet to be implemented
    error.SetErrorString("ABISysV_i386::SetReturnValueObject(): Not implemented yet");
    return error;
}


ValueObjectSP
ABISysV_i386::GetReturnValueObjectSimple (Thread &thread,
                                          ClangASTType &return_clang_type) const
{
    ValueObjectSP return_valobj_sp;
    Value value;

    if (!return_clang_type)
        return return_valobj_sp;

    value.SetClangType (return_clang_type);

    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return return_valobj_sp;

    const uint32_t type_flags = return_clang_type.GetTypeInfo ();

    unsigned eax_id = reg_ctx->GetRegisterInfoByName("eax", 0)->kinds[eRegisterKindLLDB];
    unsigned edx_id = reg_ctx->GetRegisterInfoByName("edx", 0)->kinds[eRegisterKindLLDB];


    // Following "IF ELSE" block categorizes various 'Fundamental Data Types'.
    // The terminology 'Fundamental Data Types' used here is adopted from
    // Table 2.1 of the reference document (specified on top of this file)

    if (type_flags & eTypeIsPointer)     // 'Pointer'
    {
        uint32_t ptr = thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff ;
        value.SetValueType(Value::eValueTypeScalar);
        value.GetScalar() = ptr;
        return_valobj_sp = ValueObjectConstResult::Create (thread.GetStackFrameAtIndex(0).get(),
                                                           value,
                                                           ConstString(""));
    }

    else if ((type_flags & eTypeIsScalar) || (type_flags & eTypeIsEnumeration)) //'Integral' + 'Floating Point'
    {
        value.SetValueType(Value::eValueTypeScalar);
        const size_t byte_size = return_clang_type.GetByteSize(nullptr);
        bool success = false;

        if (type_flags & eTypeIsInteger)    // 'Integral' except enum
        {
            const bool is_signed = ((type_flags & eTypeIsSigned) != 0);
            uint64_t raw_value = thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff ;
            raw_value |= (thread.GetRegisterContext()->ReadRegisterAsUnsigned(edx_id, 0) & 0xffffffff) << 32;

            switch (byte_size)
            {
                default:
                   break;

                case 16:
                   // For clang::BuiltinType::UInt128 & Int128
                   // ToDo: Need to decide how to handle it
                   break ;

                case 8:
                    if (is_signed)
                        value.GetScalar() = (int64_t)(raw_value);
                    else
                        value.GetScalar() = (uint64_t)(raw_value);
                    success = true;
                    break;

                case 4:
                    if (is_signed)
                        value.GetScalar() = (int32_t)(raw_value & UINT32_MAX);
                    else
                        value.GetScalar() = (uint32_t)(raw_value & UINT32_MAX);
                    success = true;
                    break;

                case 2:
                    if (is_signed)
                        value.GetScalar() = (int16_t)(raw_value & UINT16_MAX);
                    else
                        value.GetScalar() = (uint16_t)(raw_value & UINT16_MAX);
                    success = true;
                    break;

                case 1:
                    if (is_signed)
                        value.GetScalar() = (int8_t)(raw_value & UINT8_MAX);
                    else
                        value.GetScalar() = (uint8_t)(raw_value & UINT8_MAX);
                    success = true;
                    break;
             }

             if (success)
                 return_valobj_sp = ValueObjectConstResult::Create (thread.GetStackFrameAtIndex(0).get(),
                                                                    value,
                                                                    ConstString(""));
        }

        else if (type_flags & eTypeIsEnumeration)     // handles enum
        {
            uint32_t enm = thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff ;
            value.SetValueType(Value::eValueTypeScalar);
            value.GetScalar() = enm;
            return_valobj_sp = ValueObjectConstResult::Create (thread.GetStackFrameAtIndex(0).get(),
                                                               value,
                                                               ConstString(""));
        }

        else if (type_flags & eTypeIsFloat)  // 'Floating Point'
        {
            if (byte_size <= 12)      // handles float, double, long double, __float80
            {
                const RegisterInfo *st0_info = reg_ctx->GetRegisterInfoByName("st0", 0);
                RegisterValue st0_value;

                if (reg_ctx->ReadRegister (st0_info, st0_value))
                {
                    DataExtractor data;
                    if (st0_value.GetData(data))
                    {
                        lldb::offset_t offset = 0;
                        long double value_long_double = data.GetLongDouble(&offset);

                        if (byte_size == 4)    // float is 4 bytes
                        {
                            float value_float = (float)value_long_double;
                            value.GetScalar() = value_float;
                            success = true;
                        }
                        else if (byte_size == 8)   // double is 8 bytes
                        {
                            // On Android Platform: long double is also 8 bytes
                            // It will be handled here only.
                            double value_double = (double)value_long_double;
                            value.GetScalar() =  value_double;
                            success = true;
                        }
                        else if (byte_size == 12) // long double and __float80 are 12 bytes on i386
                        {
                            value.GetScalar() = value_long_double;
                            success = true;
                        }
                    }
                }

                if (success)
                    return_valobj_sp = ValueObjectConstResult::Create (thread.GetStackFrameAtIndex(0).get(),
                                                                       value,
                                                                       ConstString(""));
            }
            else if(byte_size == 16)   // handles __float128
            {
                lldb::addr_t storage_addr = (uint32_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff);
                return_valobj_sp = ValueObjectMemory::Create (&thread,
                                                               "",
                                                              Address (storage_addr, nullptr),
                                                              return_clang_type);
            }
        }

        else  // Neither 'Integral' nor 'Floating Point'
        {
            // If flow reaches here then check type_flags
            // This type_flags is unhandled
        }
    }

    else if (type_flags & eTypeIsComplex)    // 'Complex Floating Point'
    {
       // ToDo: Yet to be implemented
    }

    else if (type_flags & eTypeIsVector)    // 'Packed'
    {
        const size_t byte_size = return_clang_type.GetByteSize(nullptr);
        if (byte_size > 0)
        {
            const RegisterInfo *vec_reg = reg_ctx->GetRegisterInfoByName("ymm0", 0);
            if (vec_reg == nullptr)
            {
                vec_reg = reg_ctx->GetRegisterInfoByName("xmm0", 0);
                if (vec_reg == nullptr)
                    vec_reg = reg_ctx->GetRegisterInfoByName("mm0", 0);
            }

            if (vec_reg)
            {
                if (byte_size <= vec_reg->byte_size)
                {
                    ProcessSP process_sp (thread.GetProcess());
                    if (process_sp)
                    {
                        std::unique_ptr<DataBufferHeap> heap_data_ap (new DataBufferHeap(byte_size, 0));
                        const ByteOrder byte_order = process_sp->GetByteOrder();
                        RegisterValue reg_value;
                        if (reg_ctx->ReadRegister(vec_reg, reg_value))
                        {
                            Error error;
                            if (reg_value.GetAsMemoryData (vec_reg,
                                                           heap_data_ap->GetBytes(),
                                                           heap_data_ap->GetByteSize(),
                                                           byte_order,
                                                           error))
                            {
                                DataExtractor data (DataBufferSP (heap_data_ap.release()),
                                                    byte_order,
                                                    process_sp->GetTarget().GetArchitecture().GetAddressByteSize());
                                return_valobj_sp = ValueObjectConstResult::Create (&thread,
                                                                                   return_clang_type,
                                                                                   ConstString(""),
                                                                                   data);
                            }
                        }
                    }
                }
            }
        }
    }

    else    // 'Decimal Floating Point'
    {
       //ToDo: Yet to be implemented
    }
    return return_valobj_sp;
}


ValueObjectSP
ABISysV_i386::GetReturnValueObjectImpl (Thread &thread, ClangASTType &return_clang_type) const
{
    ValueObjectSP return_valobj_sp;

    if (!return_clang_type)
        return return_valobj_sp;

    ExecutionContext exe_ctx (thread.shared_from_this());
    return_valobj_sp = GetReturnValueObjectSimple(thread, return_clang_type);
    if (return_valobj_sp)
        return return_valobj_sp;

    RegisterContextSP reg_ctx_sp = thread.GetRegisterContext();
    if (!reg_ctx_sp)
       return return_valobj_sp;

    if (return_clang_type.IsAggregateType())
    {
        unsigned eax_id = reg_ctx_sp->GetRegisterInfoByName("eax", 0)->kinds[eRegisterKindLLDB];
        lldb::addr_t storage_addr = (uint32_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff);
        return_valobj_sp = ValueObjectMemory::Create (&thread,
                                                      "",
                                                      Address (storage_addr, nullptr),
                                                      return_clang_type);
    }

    return return_valobj_sp;
}

// This defines CFA as esp+4
// The saved pc is at CFA-4 (i.e. esp+0)
// The saved esp is CFA+0

bool
ABISysV_i386::CreateFunctionEntryUnwindPlan (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);

    uint32_t sp_reg_num = gcc_dwarf_esp;
    uint32_t pc_reg_num = gcc_dwarf_eip;

    UnwindPlan::RowSP row(new UnwindPlan::Row);
    row->GetCFAValue().SetIsRegisterPlusOffset(sp_reg_num, 4);
    row->SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, -4, false);
    row->SetRegisterLocationToIsCFAPlusOffset(sp_reg_num, 0, true);
    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("i386 at-func-entry default");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    return true;
}

// This defines CFA as ebp+8
// The saved pc is at CFA-4 (i.e. ebp+4)
// The saved ebp is at CFA-8 (i.e. ebp+0)
// The saved esp is CFA+0

bool
ABISysV_i386::CreateDefaultUnwindPlan (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);

    uint32_t fp_reg_num = gcc_dwarf_ebp;
    uint32_t sp_reg_num = gcc_dwarf_esp;
    uint32_t pc_reg_num = gcc_dwarf_eip;

    UnwindPlan::RowSP row(new UnwindPlan::Row);
    const int32_t ptr_size = 4;

    row->GetCFAValue().SetIsRegisterPlusOffset(fp_reg_num, 2 * ptr_size);
    row->SetOffset (0);

    row->SetRegisterLocationToAtCFAPlusOffset(fp_reg_num, ptr_size * -2, true);
    row->SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, ptr_size * -1, true);
    row->SetRegisterLocationToIsCFAPlusOffset(sp_reg_num, 0, true);

    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("i386 default unwind plan");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolNo);
    return true;
}


// According to "Register Usage" in reference document (specified on top
// of this source file) ebx, ebp, esi, edi and esp registers are preserved
// i.e. non-volatile i.e. callee-saved on i386
bool
ABISysV_i386::RegisterIsCalleeSaved (const RegisterInfo *reg_info)
{
    if (!reg_info)
        return false;

    // Saved registers are ebx, ebp, esi, edi, esp, eip
    const char *name = reg_info->name;
    if (name[0] == 'e')
    {
        switch (name[1])
        {
            case 'b':
                if (name[2] == 'x' || name[2] == 'p')
                    return name[3] == '\0';
                break;
            case 'd':
                if (name[2] == 'i')
                    return name[3] == '\0';
                break;
            case 'i':
                if (name[2] == 'p')
                    return name[3] == '\0';
                break;
            case 's':
                if (name[2] == 'i' || name[2] == 'p')
                    return name[3] == '\0';
                break;
        }
    }

    if (name[0] == 's' && name[1] == 'p' && name[2] == '\0')   // sp
        return true;
    if (name[0] == 'f' && name[1] == 'p' && name[2] == '\0')   // fp
        return true;
    if (name[0] == 'p' && name[1] == 'c' && name[2] == '\0')   // pc
        return true;

    return false;
}


void
ABISysV_i386::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "System V ABI for i386 targets",
                                   CreateInstance);
}


void
ABISysV_i386::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
ABISysV_i386::GetPluginNameStatic()
{
    static ConstString g_name("sysv-i386");
    return g_name;
}


lldb_private::ConstString
ABISysV_i386::GetPluginName()
{
    return GetPluginNameStatic();
}
