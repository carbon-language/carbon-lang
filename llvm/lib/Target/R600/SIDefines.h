//===-- SIDefines.h - SI Helper Macros ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef SIDEFINES_H_
#define SIDEFINES_H_

namespace SIInstrFlags {
enum {
  MIMG = 1 << 3,
  SMRD = 1 << 4,
  VOP1 = 1 << 5,
  VOP2 = 1 << 6,
  VOP3 = 1 << 7,
  VOPC = 1 << 8,
  SALU = 1 << 9
};
}

#define R_00B028_SPI_SHADER_PGM_RSRC1_PS                                0x00B028
#define R_00B02C_SPI_SHADER_PGM_RSRC2_PS                                0x00B02C
#define   S_00B02C_EXTRA_LDS_SIZE(x)                                  (((x) & 0xFF) << 8)
#define R_00B128_SPI_SHADER_PGM_RSRC1_VS                                0x00B128
#define R_00B228_SPI_SHADER_PGM_RSRC1_GS                                0x00B228
#define R_00B848_COMPUTE_PGM_RSRC1                                      0x00B848
#define   S_00B028_VGPRS(x)                                           (((x) & 0x3F) << 0)
#define   S_00B028_SGPRS(x)                                           (((x) & 0x0F) << 6)
#define R_00B84C_COMPUTE_PGM_RSRC2                                      0x00B84C
#define   S_00B84C_LDS_SIZE(x)                                        (((x) & 0x1FF) << 15)
#define R_0286CC_SPI_PS_INPUT_ENA                                       0x0286CC


#define R_00B848_COMPUTE_PGM_RSRC1                                      0x00B848
#define   S_00B848_VGPRS(x)                                           (((x) & 0x3F) << 0)
#define   G_00B848_VGPRS(x)                                           (((x) >> 0) & 0x3F)
#define   C_00B848_VGPRS                                              0xFFFFFFC0
#define   S_00B848_SGPRS(x)                                           (((x) & 0x0F) << 6)
#define   G_00B848_SGPRS(x)                                           (((x) >> 6) & 0x0F)
#define   C_00B848_SGPRS                                              0xFFFFFC3F
#define   S_00B848_PRIORITY(x)                                        (((x) & 0x03) << 10)
#define   G_00B848_PRIORITY(x)                                        (((x) >> 10) & 0x03)
#define   C_00B848_PRIORITY                                           0xFFFFF3FF
#define   S_00B848_FLOAT_MODE(x)                                      (((x) & 0xFF) << 12)
#define   G_00B848_FLOAT_MODE(x)                                      (((x) >> 12) & 0xFF)
#define   C_00B848_FLOAT_MODE                                         0xFFF00FFF
#define   S_00B848_PRIV(x)                                            (((x) & 0x1) << 20)
#define   G_00B848_PRIV(x)                                            (((x) >> 20) & 0x1)
#define   C_00B848_PRIV                                               0xFFEFFFFF
#define   S_00B848_DX10_CLAMP(x)                                      (((x) & 0x1) << 21)
#define   G_00B848_DX10_CLAMP(x)                                      (((x) >> 21) & 0x1)
#define   C_00B848_DX10_CLAMP                                         0xFFDFFFFF
#define   S_00B848_DEBUG_MODE(x)                                      (((x) & 0x1) << 22)
#define   G_00B848_DEBUG_MODE(x)                                      (((x) >> 22) & 0x1)
#define   C_00B848_DEBUG_MODE                                         0xFFBFFFFF
#define   S_00B848_IEEE_MODE(x)                                       (((x) & 0x1) << 23)
#define   G_00B848_IEEE_MODE(x)                                       (((x) >> 23) & 0x1)
#define   C_00B848_IEEE_MODE                                          0xFF7FFFFF


// Helpers for setting FLOAT_MODE
#define FP_ROUND_ROUND_TO_NEAREST 0
#define FP_ROUND_ROUND_TO_INF 1
#define FP_ROUND_ROUND_TO_NEGINF 2
#define FP_ROUND_ROUND_TO_ZERO 3

// Bits 3:0 control rounding mode. 1:0 control single precision, 3:2 double
// precision.
#define FP_ROUND_MODE_SP(x) ((x) & 0x3)
#define FP_ROUND_MODE_DP(x) (((x) & 0x3) << 2)

#define FP_DENORM_FLUSH_IN_FLUSH_OUT 0
#define FP_DENORM_FLUSH_OUT 1
#define FP_DENORM_FLUSH_IN 2
#define FP_DENORM_FLUSH_NONE 3


// Bits 7:4 control denormal handling. 5:4 control single precision, 6:7 double
// precision.
#define FP_DENORM_MODE_SP(x) (((x) & 0x3) << 4)
#define FP_DENORM_MODE_DP(x) (((x) & 0x3) << 6)

#endif // SIDEFINES_H_
