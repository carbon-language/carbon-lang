//===-- CodeGen/RuntimeLibcall.h - Runtime Library Calls --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the enum representing the list of runtime library calls
// the backend may emit during code generation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_RUNTIMELIBCALLS_H
#define LLVM_CODEGEN_RUNTIMELIBCALLS_H

namespace llvm {
namespace RTLIB {
  /// RTLIB::Libcall enum - This enum defines all of the runtime library calls
  /// the backend can emit.
  ///
  enum Libcall {
    // Integer
    SHL_I32,
    SHL_I64,
    SRL_I32,
    SRL_I64,
    SRA_I32,
    SRA_I64,
    MUL_I32,
    MUL_I64,
    SDIV_I32,
    SDIV_I64,
    UDIV_I32,
    UDIV_I64,
    SREM_I32,
    SREM_I64,
    UREM_I32,
    UREM_I64,
    NEG_I32,
    NEG_I64,

    // FLOATING POINT
    ADD_F32,
    ADD_F64,
    SUB_F32,
    SUB_F64,
    MUL_F32,
    MUL_F64,
    DIV_F32,
    DIV_F64,
    REM_F32,
    REM_F64,
    NEG_F32,
    NEG_F64,
    POWI_F32,
    POWI_F64,
    SQRT_F32,
    SQRT_F64,
    SIN_F32,
    SIN_F64,
    COS_F32,
    COS_F64,

    // CONVERSION
    FPEXT_F32_F64,
    FPROUND_F64_F32,
    FPTOSINT_F32_I32,
    FPTOSINT_F32_I64,
    FPTOSINT_F64_I32,
    FPTOSINT_F64_I64,
    FPTOUINT_F32_I32,
    FPTOUINT_F32_I64,
    FPTOUINT_F64_I32,
    FPTOUINT_F64_I64,
    SINTTOFP_I32_F32,
    SINTTOFP_I32_F64,
    SINTTOFP_I64_F32,
    SINTTOFP_I64_F64,
    UINTTOFP_I32_F32,
    UINTTOFP_I32_F64,
    UINTTOFP_I64_F32,
    UINTTOFP_I64_F64,

    // COMPARISON
    OEQ_F32,
    OEQ_F64,
    UNE_F32,
    UNE_F64,
    OGE_F32,
    OGE_F64,
    OLT_F32,
    OLT_F64,
    OLE_F32,
    OLE_F64,
    OGT_F32,
    OGT_F64,
    UO_F32,
    UO_F64,
    O_F32,
    O_F64,

    UNKNOWN_LIBCALL
  };
}
}

#endif
