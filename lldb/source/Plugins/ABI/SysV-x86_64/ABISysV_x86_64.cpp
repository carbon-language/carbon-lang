//===-- ABISysV_x86_64.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ABISysV_x86_64.h"

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

#include "llvm/ADT/Triple.h"

using namespace lldb;
using namespace lldb_private;

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
    gcc_dwarf_ymm0,
    gcc_dwarf_ymm1,
    gcc_dwarf_ymm2,
    gcc_dwarf_ymm3,
    gcc_dwarf_ymm4,
    gcc_dwarf_ymm5,
    gcc_dwarf_ymm6,
    gcc_dwarf_ymm7,
    gcc_dwarf_ymm8,
    gcc_dwarf_ymm9,
    gcc_dwarf_ymm10,
    gcc_dwarf_ymm11,
    gcc_dwarf_ymm12,
    gcc_dwarf_ymm13,
    gcc_dwarf_ymm14,
    gcc_dwarf_ymm15
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
    gdb_ymm0    =  57,
    gdb_ymm1    =  58,
    gdb_ymm2    =  59,
    gdb_ymm3    =  60,
    gdb_ymm4    =  61,
    gdb_ymm5    =  62,
    gdb_ymm6    =  63,
    gdb_ymm7    =  64,
    gdb_ymm8    =  65,
    gdb_ymm9    =  66,
    gdb_ymm10   =  67,
    gdb_ymm11   =  68,
    gdb_ymm12   =  69,
    gdb_ymm13   =  70,
    gdb_ymm14   =  71,
    gdb_ymm15   =  72
};


static RegisterInfo g_register_infos[] = 
{
  //  NAME      ALT      SZ OFF ENCODING         FORMAT              COMPILER                DWARF                 GENERIC                     GDB                   LLDB NATIVE            VALUE REGS    INVALIDATE REGS
  //  ========  =======  == === =============    =================== ======================= ===================== =========================== ===================== ====================== ==========    ===============
    { "rax"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rax       , gcc_dwarf_rax       , LLDB_INVALID_REGNUM       , gdb_rax            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "rbx"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rbx       , gcc_dwarf_rbx       , LLDB_INVALID_REGNUM       , gdb_rbx            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "rcx"   , "arg4",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rcx       , gcc_dwarf_rcx       , LLDB_REGNUM_GENERIC_ARG4  , gdb_rcx            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "rdx"   , "arg3",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rdx       , gcc_dwarf_rdx       , LLDB_REGNUM_GENERIC_ARG3  , gdb_rdx            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "rsi"   , "arg2",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rsi       , gcc_dwarf_rsi       , LLDB_REGNUM_GENERIC_ARG2  , gdb_rsi            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "rdi"   , "arg1",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rdi       , gcc_dwarf_rdi       , LLDB_REGNUM_GENERIC_ARG1  , gdb_rdi            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "rbp"   , "fp",    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rbp       , gcc_dwarf_rbp       , LLDB_REGNUM_GENERIC_FP    , gdb_rbp            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "rsp"   , "sp",    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rsp       , gcc_dwarf_rsp       , LLDB_REGNUM_GENERIC_SP    , gdb_rsp            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "r8"    , "arg5",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r8        , gcc_dwarf_r8        , LLDB_REGNUM_GENERIC_ARG5  , gdb_r8             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "r9"    , "arg6",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r9        , gcc_dwarf_r9        , LLDB_REGNUM_GENERIC_ARG6  , gdb_r9             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "r10"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r10       , gcc_dwarf_r10       , LLDB_INVALID_REGNUM       , gdb_r10            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "r11"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r11       , gcc_dwarf_r11       , LLDB_INVALID_REGNUM       , gdb_r11            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "r12"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r12       , gcc_dwarf_r12       , LLDB_INVALID_REGNUM       , gdb_r12            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "r13"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r13       , gcc_dwarf_r13       , LLDB_INVALID_REGNUM       , gdb_r13            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "r14"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r14       , gcc_dwarf_r14       , LLDB_INVALID_REGNUM       , gdb_r14            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "r15"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r15       , gcc_dwarf_r15       , LLDB_INVALID_REGNUM       , gdb_r15            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "rip"   , "pc",    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rip       , gcc_dwarf_rip       , LLDB_REGNUM_GENERIC_PC    , gdb_rip            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "rflags", NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_REGNUM_GENERIC_FLAGS , gdb_rflags         , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "cs"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_cs             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ss"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_ss             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ds"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_ds             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "es"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_es             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fs"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fs             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "gs"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_gs             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm0" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm0     , gcc_dwarf_stmm0     , LLDB_INVALID_REGNUM       , gdb_stmm0          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm1" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm1     , gcc_dwarf_stmm1     , LLDB_INVALID_REGNUM       , gdb_stmm1          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm2" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm2     , gcc_dwarf_stmm2     , LLDB_INVALID_REGNUM       , gdb_stmm2          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm3" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm3     , gcc_dwarf_stmm3     , LLDB_INVALID_REGNUM       , gdb_stmm3          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm4" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm4     , gcc_dwarf_stmm4     , LLDB_INVALID_REGNUM       , gdb_stmm4          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm5" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm5     , gcc_dwarf_stmm5     , LLDB_INVALID_REGNUM       , gdb_stmm5          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm6" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm6     , gcc_dwarf_stmm6     , LLDB_INVALID_REGNUM       , gdb_stmm6          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm7" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm7     , gcc_dwarf_stmm7     , LLDB_INVALID_REGNUM       , gdb_stmm7          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fctrl" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fctrl          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fstat" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fstat          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ftag"  , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_ftag           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fiseg" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fiseg          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fioff" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fioff          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "foseg" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_foseg          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fooff" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fooff          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fop"   , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fop            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm0"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm0      , gcc_dwarf_xmm0      , LLDB_INVALID_REGNUM       , gdb_xmm0           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm1"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm1      , gcc_dwarf_xmm1      , LLDB_INVALID_REGNUM       , gdb_xmm1           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm2"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm2      , gcc_dwarf_xmm2      , LLDB_INVALID_REGNUM       , gdb_xmm2           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm3"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm3      , gcc_dwarf_xmm3      , LLDB_INVALID_REGNUM       , gdb_xmm3           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm4"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm4      , gcc_dwarf_xmm4      , LLDB_INVALID_REGNUM       , gdb_xmm4           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm5"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm5      , gcc_dwarf_xmm5      , LLDB_INVALID_REGNUM       , gdb_xmm5           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm6"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm6      , gcc_dwarf_xmm6      , LLDB_INVALID_REGNUM       , gdb_xmm6           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm7"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm7      , gcc_dwarf_xmm7      , LLDB_INVALID_REGNUM       , gdb_xmm7           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm8"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm8      , gcc_dwarf_xmm8      , LLDB_INVALID_REGNUM       , gdb_xmm8           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm9"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm9      , gcc_dwarf_xmm9      , LLDB_INVALID_REGNUM       , gdb_xmm9           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm10" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm10     , gcc_dwarf_xmm10     , LLDB_INVALID_REGNUM       , gdb_xmm10          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm11" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm11     , gcc_dwarf_xmm11     , LLDB_INVALID_REGNUM       , gdb_xmm11          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm12" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm12     , gcc_dwarf_xmm12     , LLDB_INVALID_REGNUM       , gdb_xmm12          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm13" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm13     , gcc_dwarf_xmm13     , LLDB_INVALID_REGNUM       , gdb_xmm13          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm14" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm14     , gcc_dwarf_xmm14     , LLDB_INVALID_REGNUM       , gdb_xmm14          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm15" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm15     , gcc_dwarf_xmm15     , LLDB_INVALID_REGNUM       , gdb_xmm15          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "mxcsr" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_mxcsr          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm0"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm0      , gcc_dwarf_ymm0      , LLDB_INVALID_REGNUM       , gdb_ymm0           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm1"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm1      , gcc_dwarf_ymm1      , LLDB_INVALID_REGNUM       , gdb_ymm1           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm2"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm2      , gcc_dwarf_ymm2      , LLDB_INVALID_REGNUM       , gdb_ymm2           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm3"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm3      , gcc_dwarf_ymm3      , LLDB_INVALID_REGNUM       , gdb_ymm3           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm4"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm4      , gcc_dwarf_ymm4      , LLDB_INVALID_REGNUM       , gdb_ymm4           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm5"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm5      , gcc_dwarf_ymm5      , LLDB_INVALID_REGNUM       , gdb_ymm5           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm6"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm6      , gcc_dwarf_ymm6      , LLDB_INVALID_REGNUM       , gdb_ymm6           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm7"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm7      , gcc_dwarf_ymm7      , LLDB_INVALID_REGNUM       , gdb_ymm7           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm8"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm8      , gcc_dwarf_ymm8      , LLDB_INVALID_REGNUM       , gdb_ymm8           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm9"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm9      , gcc_dwarf_ymm9      , LLDB_INVALID_REGNUM       , gdb_ymm9           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm10" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm10     , gcc_dwarf_ymm10     , LLDB_INVALID_REGNUM       , gdb_ymm10          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm11" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm11     , gcc_dwarf_ymm11     , LLDB_INVALID_REGNUM       , gdb_ymm11          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm12" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm12     , gcc_dwarf_ymm12     , LLDB_INVALID_REGNUM       , gdb_ymm12          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm13" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm13     , gcc_dwarf_ymm13     , LLDB_INVALID_REGNUM       , gdb_ymm13          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm14" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm14     , gcc_dwarf_ymm14     , LLDB_INVALID_REGNUM       , gdb_ymm14          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm15" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm15     , gcc_dwarf_ymm15     , LLDB_INVALID_REGNUM       , gdb_ymm15          , LLDB_INVALID_REGNUM },      NULL,              NULL}
};

static const uint32_t k_num_register_infos = sizeof(g_register_infos)/sizeof(RegisterInfo);
static bool g_register_info_names_constified = false;

const lldb_private::RegisterInfo *
ABISysV_x86_64::GetRegisterInfoArray (uint32_t &count)
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


size_t
ABISysV_x86_64::GetRedZoneSize () const
{
    return 128;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
ABISP
ABISysV_x86_64::CreateInstance (const ArchSpec &arch)
{
    static ABISP g_abi_sp;
    if (arch.GetTriple().getArch() == llvm::Triple::x86_64)
    {
        if (!g_abi_sp)
            g_abi_sp.reset (new ABISysV_x86_64);
        return g_abi_sp;
    }
    return ABISP();
}

bool
ABISysV_x86_64::PrepareTrivialCall (Thread &thread, 
                                    addr_t sp, 
                                    addr_t func_addr, 
                                    addr_t return_addr, 
                                    llvm::ArrayRef<addr_t> args) const
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
    {
        StreamString s;
        s.Printf("ABISysV_x86_64::PrepareTrivialCall (tid = 0x%" PRIx64 ", sp = 0x%" PRIx64 ", func_addr = 0x%" PRIx64 ", return_addr = 0x%" PRIx64,
                    thread.GetID(),
                    (uint64_t)sp,
                    (uint64_t)func_addr,
                    (uint64_t)return_addr);

        for (int i = 0; i < args.size(); ++i)
            s.Printf (", arg%d = 0x%" PRIx64, i + 1, args[i]);
        s.PutCString (")");
        log->PutCString(s.GetString().c_str());
    }
    
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return false;
    
    const RegisterInfo *reg_info = NULL;
    
    if (args.size() > 6) // TODO handle more than 6 arguments
        return false;
    
    for (int i = 0; i < args.size(); ++i)
    {
        reg_info = reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1 + i);
        if (log)
            log->Printf("About to write arg%d (0x%" PRIx64 ") into %s", i + 1, args[i], reg_info->name);
        if (!reg_ctx->WriteRegisterFromUnsigned(reg_info, args[i]))
            return false;
    }

    // First, align the SP

    if (log)
        log->Printf("16-byte aligning SP: 0x%" PRIx64 " to 0x%" PRIx64, (uint64_t)sp, (uint64_t)(sp & ~0xfull));

    sp &= ~(0xfull); // 16-byte alignment

    sp -= 8;

    Error error;
    const RegisterInfo *pc_reg_info = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    const RegisterInfo *sp_reg_info = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
    ProcessSP process_sp (thread.GetProcess());

    RegisterValue reg_value;

#if 0
    // This code adds an extra frame so that we don't lose the function that we came from
    // by pushing the PC and the FP and then writing the current FP to point to the FP value
    // we just pushed. It is disabled for now until the stack backtracing code can be debugged.

    // Save current PC
    const RegisterInfo *fp_reg_info = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FP);
    if (reg_ctx->ReadRegister(pc_reg_info, reg_value))
    {
        if (log)
            log->Printf("Pushing the current PC onto the stack: 0x%" PRIx64 ": 0x%" PRIx64, (uint64_t)sp, reg_value.GetAsUInt64());
        
        if (!process_sp->WritePointerToMemory(sp, reg_value.GetAsUInt64(), error))
            return false;

        sp -= 8;
        
        // Save current FP
        if (reg_ctx->ReadRegister(fp_reg_info, reg_value))
        {
            if (log)
                log->Printf("Pushing the current FP onto the stack: 0x%" PRIx64 ": 0x%" PRIx64, (uint64_t)sp, reg_value.GetAsUInt64());
            
            if (!process_sp->WritePointerToMemory(sp, reg_value.GetAsUInt64(), error))
                return false;
        }
        // Setup FP backchain
        reg_value.SetUInt64 (sp);
        
        if (log)
            log->Printf("Writing FP:  0x%" PRIx64 " (for FP backchain)", reg_value.GetAsUInt64());

        if (!reg_ctx->WriteRegister(fp_reg_info, reg_value))
        {
            return false;
        }
        
        sp -= 8;
    }
#endif 
    
    if (log)
        log->Printf("Pushing the return address onto the stack: 0x%" PRIx64 ": 0x%" PRIx64, (uint64_t)sp, (uint64_t)return_addr);

    // Save return address onto the stack
    if (!process_sp->WritePointerToMemory(sp, return_addr, error))
        return false;

    // %rsp is set to the actual stack value.

    if (log)
        log->Printf("Writing SP: 0x%" PRIx64, (uint64_t)sp);
    
    if (!reg_ctx->WriteRegisterFromUnsigned (sp_reg_info, sp))
        return false;

    // %rip is set to the address of the called function.
    
    if (log)
        log->Printf("Writing IP: 0x%" PRIx64, (uint64_t)func_addr);

    if (!reg_ctx->WriteRegisterFromUnsigned (pc_reg_info, func_addr))
        return false;

    return true;
}

static bool ReadIntegerArgument(Scalar           &scalar,
                                unsigned int     bit_width,
                                bool             is_signed,
                                Thread           &thread,
                                uint32_t         *argument_register_ids,
                                unsigned int     &current_argument_register,
                                addr_t           &current_stack_argument)
{
    if (bit_width > 64)
        return false; // Scalar can't hold large integer arguments
    
    if (current_argument_register < 6)
    {
        scalar = thread.GetRegisterContext()->ReadRegisterAsUnsigned(argument_register_ids[current_argument_register], 0);
        current_argument_register++;
        if (is_signed)
            scalar.SignExtend (bit_width);
    }
    else
    {
        uint32_t byte_size = (bit_width + (8-1))/8;
        Error error;
        if (thread.GetProcess()->ReadScalarIntegerFromMemory(current_stack_argument, byte_size, is_signed, scalar, error))
        {
            current_stack_argument += byte_size;
            return true;
        }
        return false;
    }
    return true;
}

bool
ABISysV_x86_64::GetArgumentValues (Thread &thread,
                                   ValueList &values) const
{
    unsigned int num_values = values.GetSize();
    unsigned int value_index;
        
    // Extract the register context so we can read arguments from registers
    
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    
    if (!reg_ctx)
        return false;
    
    // Get the pointer to the first stack argument so we have a place to start 
    // when reading data
    
    addr_t sp = reg_ctx->GetSP(0);
    
    if (!sp)
        return false;
    
    addr_t current_stack_argument = sp + 8; // jump over return address
    
    uint32_t argument_register_ids[6];
    
    argument_register_ids[0] = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1)->kinds[eRegisterKindLLDB];
    argument_register_ids[1] = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG2)->kinds[eRegisterKindLLDB];
    argument_register_ids[2] = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG3)->kinds[eRegisterKindLLDB];
    argument_register_ids[3] = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG4)->kinds[eRegisterKindLLDB];
    argument_register_ids[4] = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG5)->kinds[eRegisterKindLLDB];
    argument_register_ids[5] = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG6)->kinds[eRegisterKindLLDB];
    
    unsigned int current_argument_register = 0;
    
    for (value_index = 0;
         value_index < num_values;
         ++value_index)
    {
        Value *value = values.GetValueAtIndex(value_index);
    
        if (!value)
            return false;
        
        // We currently only support extracting values with Clang QualTypes.
        // Do we care about others?
        ClangASTType clang_type = value->GetClangType();
        if (!clang_type)
            return false;
        bool is_signed;
        
        if (clang_type.IsIntegerType (is_signed))
        {
            ReadIntegerArgument(value->GetScalar(),
                                clang_type.GetBitSize(),
                                is_signed,
                                thread, 
                                argument_register_ids, 
                                current_argument_register,
                                current_stack_argument);
        }
        else if (clang_type.IsPointerType ())
        {
            ReadIntegerArgument(value->GetScalar(),
                                clang_type.GetBitSize(),
                                false,
                                thread,
                                argument_register_ids, 
                                current_argument_register,
                                current_stack_argument);
        }
    }
    
    return true;
}

Error
ABISysV_x86_64::SetReturnValueObject(lldb::StackFrameSP &frame_sp, lldb::ValueObjectSP &new_value_sp)
{
    Error error;
    if (!new_value_sp)
    {
        error.SetErrorString("Empty value object for return value.");
        return error;
    }
    
    ClangASTType clang_type = new_value_sp->GetClangType();
    if (!clang_type)
    {
        error.SetErrorString ("Null clang type for return value.");
        return error;
    }
    
    Thread *thread = frame_sp->GetThread().get();
    
    bool is_signed;
    uint32_t count;
    bool is_complex;
    
    RegisterContext *reg_ctx = thread->GetRegisterContext().get();

    bool set_it_simple = false;
    if (clang_type.IsIntegerType (is_signed) || clang_type.IsPointerType())
    {
        const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName("rax", 0);

        DataExtractor data;
        size_t num_bytes = new_value_sp->GetData(data);
        lldb::offset_t offset = 0;
        if (num_bytes <= 8)
        {
            uint64_t raw_value = data.GetMaxU64(&offset, num_bytes);
            
            if (reg_ctx->WriteRegisterFromUnsigned (reg_info, raw_value))
                set_it_simple = true;
        }
        else
        {
            error.SetErrorString("We don't support returning longer than 64 bit integer values at present.");
        }

    }
    else if (clang_type.IsFloatingPointType (count, is_complex))
    {
        if (is_complex)
            error.SetErrorString ("We don't support returning complex values at present");
        else
        {
            size_t bit_width = clang_type.GetBitSize();
            if (bit_width <= 64)
            {
                const RegisterInfo *xmm0_info = reg_ctx->GetRegisterInfoByName("xmm0", 0);
                RegisterValue xmm0_value;
                DataExtractor data;
                size_t num_bytes = new_value_sp->GetData(data);

                unsigned char buffer[16];
                ByteOrder byte_order = data.GetByteOrder();
                
                data.CopyByteOrderedData (0, num_bytes, buffer, 16, byte_order);
                xmm0_value.SetBytes(buffer, 16, byte_order);
                reg_ctx->WriteRegister(xmm0_info, xmm0_value);
                set_it_simple = true;
            }
            else
            {
                // FIXME - don't know how to do 80 bit long doubles yet.
                error.SetErrorString ("We don't support returning float values > 64 bits at present");
            }
        }
    }
    
    if (!set_it_simple)
    {
        // Okay we've got a structure or something that doesn't fit in a simple register.
        // We should figure out where it really goes, but we don't support this yet.
        error.SetErrorString ("We only support setting simple integer and float return types at present.");
    }
    
    return error;
}


ValueObjectSP
ABISysV_x86_64::GetReturnValueObjectSimple (Thread &thread,
                                            ClangASTType &return_clang_type) const
{
    ValueObjectSP return_valobj_sp;
    Value value;
    
    if (!return_clang_type)
        return return_valobj_sp;

    //value.SetContext (Value::eContextTypeClangType, return_value_type);
    value.SetClangType (return_clang_type);
    
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return return_valobj_sp;
    
    const uint32_t type_flags = return_clang_type.GetTypeInfo ();
    if (type_flags & ClangASTType::eTypeIsScalar)
    {
        value.SetValueType(Value::eValueTypeScalar);

        bool success = false;
        if (type_flags & ClangASTType::eTypeIsInteger)
        {
            // Extract the register context so we can read arguments from registers
            
            const size_t byte_size = return_clang_type.GetByteSize();
            uint64_t raw_value = thread.GetRegisterContext()->ReadRegisterAsUnsigned(reg_ctx->GetRegisterInfoByName("rax", 0), 0);
            const bool is_signed = (type_flags & ClangASTType::eTypeIsSigned) != 0;
            switch (byte_size)
            {
            default:
                break;

            case sizeof(uint64_t):
                if (is_signed)
                    value.GetScalar() = (int64_t)(raw_value);
                else
                    value.GetScalar() = (uint64_t)(raw_value);
                success = true;
                break;

            case sizeof(uint32_t):
                if (is_signed)
                    value.GetScalar() = (int32_t)(raw_value & UINT32_MAX);
                else
                    value.GetScalar() = (uint32_t)(raw_value & UINT32_MAX);
                success = true;
                break;

            case sizeof(uint16_t):
                if (is_signed)
                    value.GetScalar() = (int16_t)(raw_value & UINT16_MAX);
                else
                    value.GetScalar() = (uint16_t)(raw_value & UINT16_MAX);
                success = true;
                break;

            case sizeof(uint8_t):
                if (is_signed)
                    value.GetScalar() = (int8_t)(raw_value & UINT8_MAX);
                else
                    value.GetScalar() = (uint8_t)(raw_value & UINT8_MAX);
                success = true;
                break;
            }
        }
        else if (type_flags & ClangASTType::eTypeIsFloat)
        {
            if (type_flags & ClangASTType::eTypeIsComplex)
            {
                // Don't handle complex yet.
            }
            else
            {
                const size_t byte_size = return_clang_type.GetByteSize();
                if (byte_size <= sizeof(long double))
                {
                    const RegisterInfo *xmm0_info = reg_ctx->GetRegisterInfoByName("xmm0", 0);
                    RegisterValue xmm0_value;
                    if (reg_ctx->ReadRegister (xmm0_info, xmm0_value))
                    {
                        DataExtractor data;
                        if (xmm0_value.GetData(data))
                        {
                            lldb::offset_t offset = 0;
                            if (byte_size == sizeof(float))
                            {
                                value.GetScalar() = (float) data.GetFloat(&offset);
                                success = true;
                            }
                            else if (byte_size == sizeof(double))
                            {
                                value.GetScalar() = (double) data.GetDouble(&offset);
                                success = true;
                            }
                            else if (byte_size == sizeof(long double))
                            {
                                // Don't handle long double since that can be encoded as 80 bit floats...
                            }
                        }
                    }
                }
            }
        }
        
        if (success)
            return_valobj_sp = ValueObjectConstResult::Create (thread.GetStackFrameAtIndex(0).get(),
                                                               value,
                                                               ConstString(""));

    }
    else if (type_flags & ClangASTType::eTypeIsPointer)
    {
        unsigned rax_id = reg_ctx->GetRegisterInfoByName("rax", 0)->kinds[eRegisterKindLLDB];
        value.GetScalar() = (uint64_t)thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0);
        value.SetValueType(Value::eValueTypeScalar);
        return_valobj_sp = ValueObjectConstResult::Create (thread.GetStackFrameAtIndex(0).get(),
                                                           value,
                                                           ConstString(""));
    }
    else if (type_flags & ClangASTType::eTypeIsVector)
    {
        const size_t byte_size = return_clang_type.GetByteSize();
        if (byte_size > 0)
        {

            const RegisterInfo *altivec_reg = reg_ctx->GetRegisterInfoByName("ymm0", 0);
            if (altivec_reg == NULL)
            {
                altivec_reg = reg_ctx->GetRegisterInfoByName("xmm0", 0);
                if (altivec_reg == NULL)
                    altivec_reg = reg_ctx->GetRegisterInfoByName("mm0", 0);
            }
            
            if (altivec_reg)
            {
                if (byte_size <= altivec_reg->byte_size)
                {
                    ProcessSP process_sp (thread.GetProcess());
                    if (process_sp)
                    {
                        std::unique_ptr<DataBufferHeap> heap_data_ap (new DataBufferHeap(byte_size, 0));
                        const ByteOrder byte_order = process_sp->GetByteOrder();
                        RegisterValue reg_value;
                        if (reg_ctx->ReadRegister(altivec_reg, reg_value))
                        {
                            Error error;
                            if (reg_value.GetAsMemoryData (altivec_reg,
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
    
    return return_valobj_sp;
}

ValueObjectSP
ABISysV_x86_64::GetReturnValueObjectImpl (Thread &thread, ClangASTType &return_clang_type) const
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
        
    const size_t bit_width = return_clang_type.GetBitSize();
    if (return_clang_type.IsAggregateType())
    {
        Target *target = exe_ctx.GetTargetPtr();
        bool is_memory = true;
        if (bit_width <= 128)
        {
            ByteOrder target_byte_order = target->GetArchitecture().GetByteOrder();
            DataBufferSP data_sp (new DataBufferHeap(16, 0));
            DataExtractor return_ext (data_sp, 
                                      target_byte_order, 
                                      target->GetArchitecture().GetAddressByteSize());
                                                           
            const RegisterInfo *rax_info = reg_ctx_sp->GetRegisterInfoByName("rax", 0);
            const RegisterInfo *rdx_info = reg_ctx_sp->GetRegisterInfoByName("rdx", 0);
            const RegisterInfo *xmm0_info = reg_ctx_sp->GetRegisterInfoByName("xmm0", 0);
            const RegisterInfo *xmm1_info = reg_ctx_sp->GetRegisterInfoByName("xmm1", 0);
            
            RegisterValue rax_value, rdx_value, xmm0_value, xmm1_value;
            reg_ctx_sp->ReadRegister (rax_info, rax_value);
            reg_ctx_sp->ReadRegister (rdx_info, rdx_value);
            reg_ctx_sp->ReadRegister (xmm0_info, xmm0_value);
            reg_ctx_sp->ReadRegister (xmm1_info, xmm1_value);

            DataExtractor rax_data, rdx_data, xmm0_data, xmm1_data;
            
            rax_value.GetData(rax_data);
            rdx_value.GetData(rdx_data);
            xmm0_value.GetData(xmm0_data);
            xmm1_value.GetData(xmm1_data);
            
            uint32_t fp_bytes = 0;       // Tracks how much of the xmm registers we've consumed so far
            uint32_t integer_bytes = 0;  // Tracks how much of the rax/rds registers we've consumed so far
            
            const uint32_t num_children = return_clang_type.GetNumFields ();
            
            // Since we are in the small struct regime, assume we are not in memory.
            is_memory = false;
            
            for (uint32_t idx = 0; idx < num_children; idx++)
            {
                std::string name;
                uint64_t field_bit_offset = 0;
                bool is_signed;
                bool is_complex;
                uint32_t count;
                
                ClangASTType field_clang_type = return_clang_type.GetFieldAtIndex (idx, name, &field_bit_offset, NULL, NULL);
                const size_t field_bit_width = field_clang_type.GetBitSize();

                // If there are any unaligned fields, this is stored in memory.
                if (field_bit_offset % field_bit_width != 0)
                {
                    is_memory = true;
                    break;
                }
                
                uint32_t field_byte_width = field_bit_width/8;
                uint32_t field_byte_offset = field_bit_offset/8;
                

                DataExtractor *copy_from_extractor = NULL;
                uint32_t       copy_from_offset    = 0;
                
                if (field_clang_type.IsIntegerType (is_signed) || field_clang_type.IsPointerType ())
                {
                    if (integer_bytes < 8)
                    {
                        if (integer_bytes + field_byte_width <= 8)
                        {
                            // This is in RAX, copy from register to our result structure:
                            copy_from_extractor = &rax_data;
                            copy_from_offset = integer_bytes;
                            integer_bytes += field_byte_width;
                        }
                        else
                        {
                            // The next field wouldn't fit in the remaining space, so we pushed it to rdx.
                            copy_from_extractor = &rdx_data;
                            copy_from_offset = 0;
                            integer_bytes = 8 + field_byte_width;
                        
                        }
                    }
                    else if (integer_bytes + field_byte_width <= 16)
                    {
                        copy_from_extractor = &rdx_data;
                        copy_from_offset = integer_bytes - 8;
                        integer_bytes += field_byte_width;
                    }
                    else
                    {
                        // The last field didn't fit.  I can't see how that would happen w/o the overall size being 
                        // greater than 16 bytes.  For now, return a NULL return value object.
                        return return_valobj_sp;
                    }
                }
                else if (field_clang_type.IsFloatingPointType (count, is_complex))
                {
                    // Structs with long doubles are always passed in memory.
                    if (field_bit_width == 128)
                    {
                        is_memory = true;
                        break;
                    }
                    else if (field_bit_width == 64)
                    {
                        // These have to be in a single xmm register.
                        if (fp_bytes == 0)
                            copy_from_extractor = &xmm0_data;
                        else
                            copy_from_extractor = &xmm1_data;

                        copy_from_offset = 0;
                        fp_bytes += field_byte_width;
                    }
                    else if (field_bit_width == 32)
                    {
                        // This one is kind of complicated.  If we are in an "eightbyte" with another float, we'll
                        // be stuffed into an xmm register with it.  If we are in an "eightbyte" with one or more ints,
                        // then we will be stuffed into the appropriate GPR with them.
                        bool in_gpr;
                        if (field_byte_offset % 8 == 0) 
                        {
                            // We are at the beginning of one of the eightbytes, so check the next element (if any)
                            if (idx == num_children - 1)
                                in_gpr = false;
                            else
                            {
                                uint64_t next_field_bit_offset = 0;
                                ClangASTType next_field_clang_type = return_clang_type.GetFieldAtIndex (idx + 1,
                                                                                                        name,
                                                                                                        &next_field_bit_offset,
                                                                                                        NULL,
                                                                                                        NULL);
                                if (next_field_clang_type.IsIntegerType (is_signed))
                                    in_gpr = true;
                                else
                                {
                                    copy_from_offset = 0;
                                    in_gpr = false;
                                }
                            }
                                
                        }
                        else if (field_byte_offset % 4 == 0)
                        {
                            // We are inside of an eightbyte, so see if the field before us is floating point:
                            // This could happen if somebody put padding in the structure.
                            if (idx == 0)
                                in_gpr = false;
                            else
                            {
                                uint64_t prev_field_bit_offset = 0;
                                ClangASTType prev_field_clang_type = return_clang_type.GetFieldAtIndex (idx - 1,
                                                                                                        name,
                                                                                                        &prev_field_bit_offset,
                                                                                                        NULL,
                                                                                                        NULL);
                                if (prev_field_clang_type.IsIntegerType (is_signed))
                                    in_gpr = true;
                                else
                                {
                                    copy_from_offset = 4;
                                    in_gpr = false;
                                }
                            }
                            
                        }
                        else
                        {
                            is_memory = true;
                            continue;
                        }
                        
                        // Okay, we've figured out whether we are in GPR or XMM, now figure out which one.
                        if (in_gpr)
                        {
                            if (integer_bytes < 8)
                            {
                                // This is in RAX, copy from register to our result structure:
                                copy_from_extractor = &rax_data;
                                copy_from_offset = integer_bytes;
                                integer_bytes += field_byte_width;
                            }
                            else
                            {
                                copy_from_extractor = &rdx_data;
                                copy_from_offset = integer_bytes - 8;
                                integer_bytes += field_byte_width;
                            }
                        }
                        else
                        {
                            if (fp_bytes < 8)
                                copy_from_extractor = &xmm0_data;
                            else
                                copy_from_extractor = &xmm1_data;

                            fp_bytes += field_byte_width;
                        }
                    } 
                }
                
                // These two tests are just sanity checks.  If I somehow get the
                // type calculation wrong above it is better to just return nothing
                // than to assert or crash.
                if (!copy_from_extractor)
                    return return_valobj_sp;
                if (copy_from_offset + field_byte_width > copy_from_extractor->GetByteSize())
                    return return_valobj_sp;
                    
                copy_from_extractor->CopyByteOrderedData (copy_from_offset, 
                                                          field_byte_width, 
                                                          data_sp->GetBytes() + field_byte_offset, 
                                                          field_byte_width, 
                                                          target_byte_order);
            }
            
            if (!is_memory)
            {
                // The result is in our data buffer.  Let's make a variable object out of it:
                return_valobj_sp = ValueObjectConstResult::Create (&thread, 
                                                                   return_clang_type,
                                                                   ConstString(""),
                                                                   return_ext);
            }
        }
        
        
        // FIXME: This is just taking a guess, rax may very well no longer hold the return storage location.
        // If we are going to do this right, when we make a new frame we should check to see if it uses a memory
        // return, and if we are at the first instruction and if so stash away the return location.  Then we would
        // only return the memory return value if we know it is valid.
        
        if (is_memory)
        {
            unsigned rax_id = reg_ctx_sp->GetRegisterInfoByName("rax", 0)->kinds[eRegisterKindLLDB];
            lldb::addr_t storage_addr = (uint64_t)thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0);
            return_valobj_sp = ValueObjectMemory::Create (&thread,
                                                          "",
                                                          Address (storage_addr, NULL),
                                                          return_clang_type); 
        }
    }
        
    return return_valobj_sp;
}

bool
ABISysV_x86_64::CreateFunctionEntryUnwindPlan (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);
    
    uint32_t sp_reg_num = gcc_dwarf_rsp;
    uint32_t pc_reg_num = gcc_dwarf_rip;
    
    UnwindPlan::RowSP row(new UnwindPlan::Row);
    row->SetCFARegister (sp_reg_num);
    row->SetCFAOffset (8);
    row->SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, -8, false);
    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("x86_64 at-func-entry default");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    return true;
}

bool
ABISysV_x86_64::CreateDefaultUnwindPlan (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);

    uint32_t fp_reg_num = gcc_dwarf_rbp;
    uint32_t sp_reg_num = gcc_dwarf_rsp;
    uint32_t pc_reg_num = gcc_dwarf_rip;
    
    UnwindPlan::RowSP row(new UnwindPlan::Row);

    const int32_t ptr_size = 8;
    row->SetCFARegister (gcc_dwarf_rbp);
    row->SetCFAOffset (2 * ptr_size);
    row->SetOffset (0);
    
    row->SetRegisterLocationToAtCFAPlusOffset(fp_reg_num, ptr_size * -2, true);
    row->SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, ptr_size * -1, true);
    row->SetRegisterLocationToIsCFAPlusOffset(sp_reg_num, 0, true);

    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("x86_64 default unwind plan");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolNo);
    return true;
}

bool
ABISysV_x86_64::RegisterIsVolatile (const RegisterInfo *reg_info)
{
    return !RegisterIsCalleeSaved (reg_info);
}



// See "Register Usage" in the 
// "System V Application Binary Interface"
// "AMD64 Architecture Processor Supplement" 
// (or "x86-64(tm) Architecture Processor Supplement" in earlier revisions)
// (this doc is also commonly referred to as the x86-64/AMD64 psABI)
// Edited by Michael Matz, Jan Hubicka, Andreas Jaeger, and Mark Mitchell
// current version is 0.99.6 released 2012-07-02 at http://refspecs.linuxfoundation.org/elf/x86-64-abi-0.99.pdf

bool
ABISysV_x86_64::RegisterIsCalleeSaved (const RegisterInfo *reg_info)
{
    if (reg_info)
    {
        // Preserved registers are :
        //    rbx, rsp, rbp, r12, r13, r14, r15
        //    mxcsr (partially preserved)
        //    x87 control word

        const char *name = reg_info->name;
        if (name[0] == 'r')
        {
            switch (name[1])
            {
            case '1': // r12, r13, r14, r15
                if (name[2] >= '2' && name[2] <= '5')
                    return name[3] == '\0';
                break;

            default:
                break;
            }
        }

        // Accept shorter-variant versions, rbx/ebx, rip/ eip, etc.
        if (name[0] == 'r' || name[0] == 'e')
        {
            switch (name[1])
            {
            case 'b': // rbp, rbx
                if (name[2] == 'p' || name[2] == 'x')
                    return name[3] == '\0';
                break;

            case 'i': // rip
                if (name[2] == 'p')
                    return name[3] == '\0'; 
                break;

            case 's': // rsp
                if (name[2] == 'p')
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
    }
    return false;
}



void
ABISysV_x86_64::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "System V ABI for x86_64 targets",
                                   CreateInstance);
}

void
ABISysV_x86_64::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

lldb_private::ConstString
ABISysV_x86_64::GetPluginNameStatic()
{
    static ConstString g_name("sysv-x86_64");
    return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
ABISysV_x86_64::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ABISysV_x86_64::GetPluginVersion()
{
    return 1;
}

