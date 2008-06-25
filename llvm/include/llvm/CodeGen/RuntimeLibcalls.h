//===-- CodeGen/RuntimeLibcall.h - Runtime Library Calls --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  /// the backend can emit.  The various long double types cannot be merged,
  /// because 80-bit library functions use "xf" and 128-bit use "tf".
  /// 
  /// When adding PPCF128 functions here, note that their names generally need
  /// to be overridden for Darwin with the xxx$LDBL128 form.  See
  /// PPCISelLowering.cpp.
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
    ADD_F80,
    ADD_PPCF128,
    SUB_F32,
    SUB_F64,
    SUB_F80,
    SUB_PPCF128,
    MUL_F32,
    MUL_F64,
    MUL_F80,
    MUL_PPCF128,
    DIV_F32,
    DIV_F64,
    DIV_F80,
    DIV_PPCF128,
    REM_F32,
    REM_F64,
    REM_F80,
    REM_PPCF128,
    POWI_F32,
    POWI_F64,
    POWI_F80,
    POWI_PPCF128,
    SQRT_F32,
    SQRT_F64,
    SQRT_F80,
    SQRT_PPCF128,
    SIN_F32,
    SIN_F64,
    SIN_F80,
    SIN_PPCF128,
    COS_F32,
    COS_F64,
    COS_F80,
    COS_PPCF128,
    POW_F32,
    POW_F64,
    POW_F80,
    POW_PPCF128,

    // CONVERSION
    FPEXT_F32_F64,
    FPROUND_F64_F32,
    FPTOSINT_F32_I32,
    FPTOSINT_F32_I64,
    FPTOSINT_F32_I128,
    FPTOSINT_F64_I32,
    FPTOSINT_F64_I64,
    FPTOSINT_F64_I128,
    FPTOSINT_F80_I64,
    FPTOSINT_F80_I128,
    FPTOSINT_PPCF128_I32,
    FPTOSINT_PPCF128_I64,
    FPTOSINT_PPCF128_I128,
    FPTOUINT_F32_I32,
    FPTOUINT_F32_I64,
    FPTOUINT_F32_I128,
    FPTOUINT_F64_I32,
    FPTOUINT_F64_I64,
    FPTOUINT_F64_I128,
    FPTOUINT_F80_I32,
    FPTOUINT_F80_I64,
    FPTOUINT_F80_I128,
    FPTOUINT_PPCF128_I32,
    FPTOUINT_PPCF128_I64,
    FPTOUINT_PPCF128_I128,
    SINTTOFP_I32_F32,
    SINTTOFP_I32_F64,
    SINTTOFP_I64_F32,
    SINTTOFP_I64_F64,
    SINTTOFP_I64_F80,
    SINTTOFP_I64_PPCF128,
    SINTTOFP_I128_F32,
    SINTTOFP_I128_F64,
    SINTTOFP_I128_F80,
    SINTTOFP_I128_PPCF128,
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
