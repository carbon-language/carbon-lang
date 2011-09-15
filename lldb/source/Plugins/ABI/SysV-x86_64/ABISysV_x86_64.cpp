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

static const char *pluginName = "ABISysV_x86_64";
static const char *pluginDesc = "System V ABI for x86_64 targets";
static const char *pluginShort = "abi.sysv-x86_64";


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


static RegisterInfo g_register_infos[] = 
{
  //  NAME      ALT      SZ OFF ENCODING         FORMAT              COMPILER                DWARF                 GENERIC                     GDB                   LLDB NATIVE
  //  ========  =======  == === =============    =================== ======================= ===================== =========================== ===================== ======================
    { "rax"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rax       , gcc_dwarf_rax       , LLDB_INVALID_REGNUM       , gdb_rax            , LLDB_INVALID_REGNUM }},
    { "rbx"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rbx       , gcc_dwarf_rbx       , LLDB_INVALID_REGNUM       , gdb_rbx            , LLDB_INVALID_REGNUM }},
    { "rcx"   , "arg4",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rcx       , gcc_dwarf_rcx       , LLDB_REGNUM_GENERIC_ARG4  , gdb_rcx            , LLDB_INVALID_REGNUM }},
    { "rdx"   , "arg3",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rdx       , gcc_dwarf_rdx       , LLDB_REGNUM_GENERIC_ARG3  , gdb_rdx            , LLDB_INVALID_REGNUM }},
    { "rsi"   , "arg2",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rsi       , gcc_dwarf_rsi       , LLDB_REGNUM_GENERIC_ARG2  , gdb_rsi            , LLDB_INVALID_REGNUM }},
    { "rdi"   , "arg1",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rdi       , gcc_dwarf_rdi       , LLDB_REGNUM_GENERIC_ARG1  , gdb_rdi            , LLDB_INVALID_REGNUM }},
    { "rbp"   , "fp",    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rbp       , gcc_dwarf_rbp       , LLDB_REGNUM_GENERIC_FP    , gdb_rbp            , LLDB_INVALID_REGNUM }},
    { "rsp"   , "sp",    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rsp       , gcc_dwarf_rsp       , LLDB_REGNUM_GENERIC_SP    , gdb_rsp            , LLDB_INVALID_REGNUM }},
    { "r8"    , "arg5",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r8        , gcc_dwarf_r8        , LLDB_REGNUM_GENERIC_ARG5  , gdb_r8             , LLDB_INVALID_REGNUM }},
    { "r9"    , "arg6",  8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r9        , gcc_dwarf_r9        , LLDB_REGNUM_GENERIC_ARG6  , gdb_r9             , LLDB_INVALID_REGNUM }},
    { "r10"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r10       , gcc_dwarf_r10       , LLDB_INVALID_REGNUM       , gdb_r10            , LLDB_INVALID_REGNUM }},
    { "r11"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r11       , gcc_dwarf_r11       , LLDB_INVALID_REGNUM       , gdb_r11            , LLDB_INVALID_REGNUM }},
    { "r12"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r12       , gcc_dwarf_r12       , LLDB_INVALID_REGNUM       , gdb_r12            , LLDB_INVALID_REGNUM }},
    { "r13"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r13       , gcc_dwarf_r13       , LLDB_INVALID_REGNUM       , gdb_r13            , LLDB_INVALID_REGNUM }},
    { "r14"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r14       , gcc_dwarf_r14       , LLDB_INVALID_REGNUM       , gdb_r14            , LLDB_INVALID_REGNUM }},
    { "r15"   , NULL,    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_r15       , gcc_dwarf_r15       , LLDB_INVALID_REGNUM       , gdb_r15            , LLDB_INVALID_REGNUM }},
    { "rip"   , "pc",    8,  0, eEncodingUint  , eFormatHex          , { gcc_dwarf_rip       , gcc_dwarf_rip       , LLDB_REGNUM_GENERIC_PC    , gdb_rip            , LLDB_INVALID_REGNUM }},
    { "rflags", NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_REGNUM_GENERIC_FLAGS , gdb_rflags         , LLDB_INVALID_REGNUM }},
    { "cs"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_cs             , LLDB_INVALID_REGNUM }},
    { "ss"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_ss             , LLDB_INVALID_REGNUM }},
    { "ds"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_ds             , LLDB_INVALID_REGNUM }},
    { "es"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_es             , LLDB_INVALID_REGNUM }},
    { "fs"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fs             , LLDB_INVALID_REGNUM }},
    { "gs"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_gs             , LLDB_INVALID_REGNUM }},
    { "stmm0" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm0     , gcc_dwarf_stmm0     , LLDB_INVALID_REGNUM       , gdb_stmm0          , LLDB_INVALID_REGNUM }},
    { "stmm1" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm1     , gcc_dwarf_stmm1     , LLDB_INVALID_REGNUM       , gdb_stmm1          , LLDB_INVALID_REGNUM }},
    { "stmm2" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm2     , gcc_dwarf_stmm2     , LLDB_INVALID_REGNUM       , gdb_stmm2          , LLDB_INVALID_REGNUM }},
    { "stmm3" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm3     , gcc_dwarf_stmm3     , LLDB_INVALID_REGNUM       , gdb_stmm3          , LLDB_INVALID_REGNUM }},
    { "stmm4" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm4     , gcc_dwarf_stmm4     , LLDB_INVALID_REGNUM       , gdb_stmm4          , LLDB_INVALID_REGNUM }},
    { "stmm5" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm5     , gcc_dwarf_stmm5     , LLDB_INVALID_REGNUM       , gdb_stmm5          , LLDB_INVALID_REGNUM }},
    { "stmm6" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm6     , gcc_dwarf_stmm6     , LLDB_INVALID_REGNUM       , gdb_stmm6          , LLDB_INVALID_REGNUM }},
    { "stmm7" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_stmm7     , gcc_dwarf_stmm7     , LLDB_INVALID_REGNUM       , gdb_stmm7          , LLDB_INVALID_REGNUM }},
    { "fctrl" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fctrl          , LLDB_INVALID_REGNUM }},
    { "fstat" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fstat          , LLDB_INVALID_REGNUM }},
    { "ftag"  , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_ftag           , LLDB_INVALID_REGNUM }},
    { "fiseg" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fiseg          , LLDB_INVALID_REGNUM }},
    { "fioff" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fioff          , LLDB_INVALID_REGNUM }},
    { "foseg" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_foseg          , LLDB_INVALID_REGNUM }},
    { "fooff" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fooff          , LLDB_INVALID_REGNUM }},
    { "fop"   , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fop            , LLDB_INVALID_REGNUM }},
    { "xmm0"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm0      , gcc_dwarf_xmm0      , LLDB_INVALID_REGNUM       , gdb_xmm0           , LLDB_INVALID_REGNUM }},
    { "xmm1"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm1      , gcc_dwarf_xmm1      , LLDB_INVALID_REGNUM       , gdb_xmm1           , LLDB_INVALID_REGNUM }},
    { "xmm2"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm2      , gcc_dwarf_xmm2      , LLDB_INVALID_REGNUM       , gdb_xmm2           , LLDB_INVALID_REGNUM }},
    { "xmm3"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm3      , gcc_dwarf_xmm3      , LLDB_INVALID_REGNUM       , gdb_xmm3           , LLDB_INVALID_REGNUM }},
    { "xmm4"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm4      , gcc_dwarf_xmm4      , LLDB_INVALID_REGNUM       , gdb_xmm4           , LLDB_INVALID_REGNUM }},
    { "xmm5"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm5      , gcc_dwarf_xmm5      , LLDB_INVALID_REGNUM       , gdb_xmm5           , LLDB_INVALID_REGNUM }},
    { "xmm6"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm6      , gcc_dwarf_xmm6      , LLDB_INVALID_REGNUM       , gdb_xmm6           , LLDB_INVALID_REGNUM }},
    { "xmm7"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm7      , gcc_dwarf_xmm7      , LLDB_INVALID_REGNUM       , gdb_xmm7           , LLDB_INVALID_REGNUM }},
    { "xmm8"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm8      , gcc_dwarf_xmm8      , LLDB_INVALID_REGNUM       , gdb_xmm8           , LLDB_INVALID_REGNUM }},
    { "xmm9"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm9      , gcc_dwarf_xmm9      , LLDB_INVALID_REGNUM       , gdb_xmm9           , LLDB_INVALID_REGNUM }},
    { "xmm10" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm10     , gcc_dwarf_xmm10     , LLDB_INVALID_REGNUM       , gdb_xmm10          , LLDB_INVALID_REGNUM }},
    { "xmm11" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm11     , gcc_dwarf_xmm11     , LLDB_INVALID_REGNUM       , gdb_xmm11          , LLDB_INVALID_REGNUM }},
    { "xmm12" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm12     , gcc_dwarf_xmm12     , LLDB_INVALID_REGNUM       , gdb_xmm12          , LLDB_INVALID_REGNUM }},
    { "xmm13" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm13     , gcc_dwarf_xmm13     , LLDB_INVALID_REGNUM       , gdb_xmm13          , LLDB_INVALID_REGNUM }},
    { "xmm14" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm14     , gcc_dwarf_xmm14     , LLDB_INVALID_REGNUM       , gdb_xmm14          , LLDB_INVALID_REGNUM }},
    { "xmm15" , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_xmm15     , gcc_dwarf_xmm15     , LLDB_INVALID_REGNUM       , gdb_xmm15          , LLDB_INVALID_REGNUM }},
    { "mxcsr" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_mxcsr          , LLDB_INVALID_REGNUM }},
    { "ymm0"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm0      , gcc_dwarf_ymm0      , LLDB_INVALID_REGNUM       , gdb_ymm0           , LLDB_INVALID_REGNUM }},
    { "ymm1"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm1      , gcc_dwarf_ymm1      , LLDB_INVALID_REGNUM       , gdb_ymm1           , LLDB_INVALID_REGNUM }},
    { "ymm2"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm2      , gcc_dwarf_ymm2      , LLDB_INVALID_REGNUM       , gdb_ymm2           , LLDB_INVALID_REGNUM }},
    { "ymm3"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm3      , gcc_dwarf_ymm3      , LLDB_INVALID_REGNUM       , gdb_ymm3           , LLDB_INVALID_REGNUM }},
    { "ymm4"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm4      , gcc_dwarf_ymm4      , LLDB_INVALID_REGNUM       , gdb_ymm4           , LLDB_INVALID_REGNUM }},
    { "ymm5"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm5      , gcc_dwarf_ymm5      , LLDB_INVALID_REGNUM       , gdb_ymm5           , LLDB_INVALID_REGNUM }},
    { "ymm6"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm6      , gcc_dwarf_ymm6      , LLDB_INVALID_REGNUM       , gdb_ymm6           , LLDB_INVALID_REGNUM }},
    { "ymm7"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm7      , gcc_dwarf_ymm7      , LLDB_INVALID_REGNUM       , gdb_ymm7           , LLDB_INVALID_REGNUM }},
    { "ymm8"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm8      , gcc_dwarf_ymm8      , LLDB_INVALID_REGNUM       , gdb_ymm8           , LLDB_INVALID_REGNUM }},
    { "ymm9"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm9      , gcc_dwarf_ymm9      , LLDB_INVALID_REGNUM       , gdb_ymm9           , LLDB_INVALID_REGNUM }},
    { "ymm10" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm10     , gcc_dwarf_ymm10     , LLDB_INVALID_REGNUM       , gdb_ymm10          , LLDB_INVALID_REGNUM }},
    { "ymm11" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm11     , gcc_dwarf_ymm11     , LLDB_INVALID_REGNUM       , gdb_ymm11          , LLDB_INVALID_REGNUM }},
    { "ymm12" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm12     , gcc_dwarf_ymm12     , LLDB_INVALID_REGNUM       , gdb_ymm12          , LLDB_INVALID_REGNUM }},
    { "ymm13" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm13     , gcc_dwarf_ymm13     , LLDB_INVALID_REGNUM       , gdb_ymm13          , LLDB_INVALID_REGNUM }},
    { "ymm14" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm14     , gcc_dwarf_ymm14     , LLDB_INVALID_REGNUM       , gdb_ymm14          , LLDB_INVALID_REGNUM }},
    { "ymm15" , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { gcc_dwarf_ymm15     , gcc_dwarf_ymm15     , LLDB_INVALID_REGNUM       , gdb_ymm15          , LLDB_INVALID_REGNUM }}
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
                                    addr_t *arg1_ptr,
                                    addr_t *arg2_ptr,
                                    addr_t *arg3_ptr,
                                    addr_t *arg4_ptr,
                                    addr_t *arg5_ptr,
                                    addr_t *arg6_ptr) const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
        log->Printf("ABISysV_x86_64::PrepareTrivialCall\n(\n  thread = %p\n  sp = 0x%llx\n  func_addr = 0x%llx\n  return_addr = 0x%llx\n  arg1_ptr = %p (0x%llx)\n  arg2_ptr = %p (0x%llx)\n  arg3_ptr = %p (0x%llx)\n)",
                    (void*)&thread,
                    (uint64_t)sp,
                    (uint64_t)func_addr,
                    (uint64_t)return_addr,
                    arg1_ptr, arg1_ptr ? (uint64_t)*arg1_ptr : (uint64_t) 0,
                    arg2_ptr, arg2_ptr ? (uint64_t)*arg2_ptr : (uint64_t) 0,
                    arg3_ptr, arg3_ptr ? (uint64_t)*arg3_ptr : (uint64_t) 0);
    
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return false;
    
    const RegisterInfo *reg_info = NULL;
    if (arg1_ptr)
    {
        reg_info = reg_ctx->GetRegisterInfoByName("rdi", 0);
        if (log)
            log->Printf("About to write arg1 (0x%llx) into %s", (uint64_t)*arg1_ptr, reg_info->name);

        if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg1_ptr))
            return false;

        if (arg2_ptr)
        {
            reg_info = reg_ctx->GetRegisterInfoByName("rsi", 0);
            if (log)
                log->Printf("About to write arg2 (0x%llx) into %s", (uint64_t)*arg2_ptr, reg_info->name);
            if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg2_ptr))
                return false;

            if (arg3_ptr)
            {
                reg_info = reg_ctx->GetRegisterInfoByName("rdx", 0);
                if (log)
                    log->Printf("About to write arg3 (0x%llx) into %s", (uint64_t)*arg3_ptr, reg_info->name);
                if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg3_ptr))
                    return false;

                if (arg4_ptr)
                {
                    reg_info = reg_ctx->GetRegisterInfoByName("rcx", 0);
                    if (log)
                        log->Printf("About to write arg4 (0x%llx) into %s", (uint64_t)*arg4_ptr, reg_info->name);
                    if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg4_ptr))
                        return false;

                    if (arg5_ptr)
                    {
                        reg_info = reg_ctx->GetRegisterInfoByName("r8", 0);
                        if (log)
                            log->Printf("About to write arg5 (0x%llx) into %s", (uint64_t)*arg5_ptr, reg_info->name);
                        if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg5_ptr))
                            return false;

                        if (arg6_ptr)
                        {
                            reg_info = reg_ctx->GetRegisterInfoByName("r9", 0);
                            if (log)
                                log->Printf("About to write arg6 (0x%llx) into %s", (uint64_t)*arg6_ptr, reg_info->name);
                            if (!reg_ctx->WriteRegisterFromUnsigned (reg_info, *arg6_ptr))
                                return false;
                        }
                    }
                }
            }
        }
    }


    // First, align the SP

    if (log)
        log->Printf("16-byte aligning SP: 0x%llx to 0x%llx", (uint64_t)sp, (uint64_t)(sp & ~0xfull));

    sp &= ~(0xfull); // 16-byte alignment

    // The return address is pushed onto the stack (yes after the alignment...)
    sp -= 8;

    RegisterValue reg_value;
    reg_value.SetUInt64 (return_addr);

    if (log)
        log->Printf("Pushing the return address onto the stack: new SP 0x%llx, return address 0x%llx", (uint64_t)sp, (uint64_t)return_addr);

    const RegisterInfo *pc_reg_info = reg_ctx->GetRegisterInfoByName("rip");
    Error error (reg_ctx->WriteRegisterValueToMemory(pc_reg_info, sp, pc_reg_info->byte_size, reg_value));
    if (error.Fail())
        return false;

    // %rsp is set to the actual stack value.

    if (log)
        log->Printf("Writing SP (0x%llx) down", (uint64_t)sp);
    
    if (!reg_ctx->WriteRegisterFromUnsigned (reg_ctx->GetRegisterInfoByName("rsp"), sp))
        return false;

    // %rip is set to the address of the called function.
    
    if (log)
        log->Printf("Writing new IP (0x%llx) down", (uint64_t)func_addr);

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
        if (thread.GetProcess().ReadScalarIntegerFromMemory(current_stack_argument, byte_size, is_signed, scalar, error))
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
    
    // For now, assume that the types in the AST values come from the Target's 
    // scratch AST.    
    
    clang::ASTContext *ast_context = thread.CalculateTarget()->GetScratchClangASTContext()->getASTContext();
    
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
    
    argument_register_ids[0] = reg_ctx->GetRegisterInfoByName("rdi", 0)->kinds[eRegisterKindLLDB];
    argument_register_ids[1] = reg_ctx->GetRegisterInfoByName("rsi", 0)->kinds[eRegisterKindLLDB];
    argument_register_ids[2] = reg_ctx->GetRegisterInfoByName("rdx", 0)->kinds[eRegisterKindLLDB];
    argument_register_ids[3] = reg_ctx->GetRegisterInfoByName("rcx", 0)->kinds[eRegisterKindLLDB];
    argument_register_ids[4] = reg_ctx->GetRegisterInfoByName("r8", 0)->kinds[eRegisterKindLLDB];
    argument_register_ids[5] = reg_ctx->GetRegisterInfoByName("r9", 0)->kinds[eRegisterKindLLDB];
    
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
        switch (value->GetContextType())
        {
        default:
            return false;
        case Value::eContextTypeClangType:
            {
                void *value_type = value->GetClangType();
                bool is_signed;
                
                if (ClangASTContext::IsIntegerType (value_type, is_signed))
                {
                    size_t bit_width = ClangASTType::GetClangTypeBitWidth(ast_context, value_type);
                    
                    ReadIntegerArgument(value->GetScalar(),
                                        bit_width, 
                                        is_signed,
                                        thread, 
                                        argument_register_ids, 
                                        current_argument_register,
                                        current_stack_argument);
                }
                else if (ClangASTContext::IsPointerType (value_type))
                {
                    ReadIntegerArgument(value->GetScalar(),
                                        64,
                                        false,
                                        thread,
                                        argument_register_ids, 
                                        current_argument_register,
                                        current_stack_argument);
                }
            }
            break;
        }
    }
    
    return true;
}

bool
ABISysV_x86_64::GetReturnValue (Thread &thread,
                                Value &value) const
{
    switch (value.GetContextType())
    {
        default:
            return false;
        case Value::eContextTypeClangType:
        {
            void *value_type = value.GetClangType();
            bool is_signed;
            
            RegisterContext *reg_ctx = thread.GetRegisterContext().get();
            
            if (!reg_ctx)
                return false;
            
            if (ClangASTContext::IsIntegerType (value_type, is_signed))
            {
                // For now, assume that the types in the AST values come from the Target's 
                // scratch AST.    
                
                clang::ASTContext *ast_context = thread.CalculateTarget()->GetScratchClangASTContext()->getASTContext();
                
                // Extract the register context so we can read arguments from registers
                
                size_t bit_width = ClangASTType::GetClangTypeBitWidth(ast_context, value_type);
                unsigned rax_id = reg_ctx->GetRegisterInfoByName("rax", 0)->kinds[eRegisterKindLLDB];
                
                switch (bit_width)
                {
                default:
                case 128:
                    // Scalar can't hold 128-bit literals, so we don't handle this
                    return false;
                case 64:
                    if (is_signed)
                        value.GetScalar() = (int64_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0));
                    else
                        value.GetScalar() = (uint64_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0));
                    break;
                case 32:
                    if (is_signed)
                        value.GetScalar() = (int32_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xffffffff);
                    else
                        value.GetScalar() = (uint32_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xffffffff);
                    break;
                case 16:
                    if (is_signed)
                        value.GetScalar() = (int16_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xffff);
                    else
                        value.GetScalar() = (uint16_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xffff);
                    break;
                case 8:
                    if (is_signed)
                        value.GetScalar() = (int8_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xff);
                    else
                        value.GetScalar() = (uint8_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0) & 0xff);
                    break;
                }
            }
            else if (ClangASTContext::IsPointerType (value_type))
            {
                unsigned rax_id = reg_ctx->GetRegisterInfoByName("rax", 0)->kinds[eRegisterKindLLDB];
                value.GetScalar() = (uint64_t)thread.GetRegisterContext()->ReadRegisterAsUnsigned(rax_id, 0);
            }
            else
            {
                // not handled yet
                return false;
            }
        }
        break;
    }
    
    return true;
}

bool
ABISysV_x86_64::CreateFunctionEntryUnwindPlan (UnwindPlan &unwind_plan)
{
    uint32_t reg_kind = unwind_plan.GetRegisterKind();
    uint32_t sp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t pc_reg_num = LLDB_INVALID_REGNUM;
    
    switch (reg_kind)
    {
    case eRegisterKindDWARF:
    case eRegisterKindGCC:
        sp_reg_num = gcc_dwarf_rsp;
        pc_reg_num = gcc_dwarf_rip;
        break;

    case eRegisterKindGDB:
        sp_reg_num = gdb_rsp;
        pc_reg_num = gdb_rip;
        break;

    case eRegisterKindGeneric:
        sp_reg_num = LLDB_REGNUM_GENERIC_SP;
        pc_reg_num = LLDB_REGNUM_GENERIC_PC;
        break;
    }

    if (sp_reg_num == LLDB_INVALID_REGNUM ||
        pc_reg_num == LLDB_INVALID_REGNUM)
        return false;

    UnwindPlan::Row row;
    row.SetCFARegister (sp_reg_num);
    row.SetCFAOffset (8);
    row.SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, -8, false);    
    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("x86_64 at-func-entry default");
    return true;
}

bool
ABISysV_x86_64::CreateDefaultUnwindPlan (UnwindPlan &unwind_plan)
{
    uint32_t reg_kind = unwind_plan.GetRegisterKind();
    uint32_t fp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t sp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t pc_reg_num = LLDB_INVALID_REGNUM;
    
    switch (reg_kind)
    {
        case eRegisterKindDWARF:
        case eRegisterKindGCC:
            fp_reg_num = gcc_dwarf_rbp;
            sp_reg_num = gcc_dwarf_rsp;
            pc_reg_num = gcc_dwarf_rip;
            break;
            
        case eRegisterKindGDB:
            fp_reg_num = gdb_rbp;
            sp_reg_num = gdb_rsp;
            pc_reg_num = gdb_rip;
            break;
            
        case eRegisterKindGeneric:
            fp_reg_num = LLDB_REGNUM_GENERIC_FP;
            sp_reg_num = LLDB_REGNUM_GENERIC_SP;
            pc_reg_num = LLDB_REGNUM_GENERIC_PC;
            break;
    }

    if (fp_reg_num == LLDB_INVALID_REGNUM ||
        sp_reg_num == LLDB_INVALID_REGNUM ||
        pc_reg_num == LLDB_INVALID_REGNUM)
        return false;

    UnwindPlan::Row row;

    const int32_t ptr_size = 8;
    row.SetCFARegister (LLDB_REGNUM_GENERIC_FP);
    row.SetCFAOffset (2 * ptr_size);
    row.SetOffset (0);
    
    row.SetRegisterLocationToAtCFAPlusOffset(fp_reg_num, ptr_size * -2, true);
    row.SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, ptr_size * -1, true);
    row.SetRegisterLocationToAtCFAPlusOffset(sp_reg_num, ptr_size *  0, true);

    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("x86_64 default unwind plan");
    return true;
}

bool
ABISysV_x86_64::RegisterIsVolatile (const RegisterInfo *reg_info)
{
    return !RegisterIsCalleeSaved (reg_info);
}





bool
ABISysV_x86_64::RegisterIsCalleeSaved (const RegisterInfo *reg_info)
{
    if (reg_info)
    {
        // Volatile registers include: rbx, rbp, rsp, r12, r13, r14, r15, rip
        const char *name = reg_info->name;
        if (name[0] == 'r')
        {
            switch (name[1])
            {
            case '1': // r12, r13, r14, r15
                if (name[2] >= '2' && name[2] <= '5')
                    return name[3] == '\0';
                break;

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

            default:
                break;
            }
        }
    }
    return false;
}



void
ABISysV_x86_64::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);
}

void
ABISysV_x86_64::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ABISysV_x86_64::GetPluginName()
{
    return pluginName;
}

const char *
ABISysV_x86_64::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
ABISysV_x86_64::GetPluginVersion()
{
    return 1;
}

