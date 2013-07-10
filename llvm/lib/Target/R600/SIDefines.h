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

#endif // SIDEFINES_H_
