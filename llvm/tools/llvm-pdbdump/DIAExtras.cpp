//===- DIAExtras.cpp - Helper classes and functions for DIA C++ ---------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm-pdbdump.h"
#include "DIAExtras.h"

using namespace llvm;
using namespace llvm::sys::windows;

#define PRINT_ENUM_VALUE_CASE(Value)                                           \
  case Value:                                                                  \
    outs() << #Value;                                                          \
    break;

raw_ostream &llvm::operator<<(raw_ostream &Stream, DiaSymTagEnum SymTag) {
  switch (SymTag) {
    PRINT_ENUM_VALUE_CASE(SymTagNull)
    PRINT_ENUM_VALUE_CASE(SymTagExe)
    PRINT_ENUM_VALUE_CASE(SymTagCompiland)
    PRINT_ENUM_VALUE_CASE(SymTagCompilandDetails)
    PRINT_ENUM_VALUE_CASE(SymTagCompilandEnv)
    PRINT_ENUM_VALUE_CASE(SymTagFunction)
    PRINT_ENUM_VALUE_CASE(SymTagBlock)
    PRINT_ENUM_VALUE_CASE(SymTagData)
    PRINT_ENUM_VALUE_CASE(SymTagAnnotation)
    PRINT_ENUM_VALUE_CASE(SymTagLabel)
    PRINT_ENUM_VALUE_CASE(SymTagPublicSymbol)
    PRINT_ENUM_VALUE_CASE(SymTagUDT)
    PRINT_ENUM_VALUE_CASE(SymTagEnum)
    PRINT_ENUM_VALUE_CASE(SymTagFunctionType)
    PRINT_ENUM_VALUE_CASE(SymTagPointerType)
    PRINT_ENUM_VALUE_CASE(SymTagArrayType)
    PRINT_ENUM_VALUE_CASE(SymTagBaseType)
    PRINT_ENUM_VALUE_CASE(SymTagTypedef)
    PRINT_ENUM_VALUE_CASE(SymTagBaseClass)
    PRINT_ENUM_VALUE_CASE(SymTagFriend)
    PRINT_ENUM_VALUE_CASE(SymTagFunctionArgType)
    PRINT_ENUM_VALUE_CASE(SymTagFuncDebugStart)
    PRINT_ENUM_VALUE_CASE(SymTagFuncDebugEnd)
    PRINT_ENUM_VALUE_CASE(SymTagUsingNamespace)
    PRINT_ENUM_VALUE_CASE(SymTagVTableShape)
    PRINT_ENUM_VALUE_CASE(SymTagVTable)
    PRINT_ENUM_VALUE_CASE(SymTagCustom)
    PRINT_ENUM_VALUE_CASE(SymTagThunk)
    PRINT_ENUM_VALUE_CASE(SymTagCustomType)
    PRINT_ENUM_VALUE_CASE(SymTagManagedType)
    PRINT_ENUM_VALUE_CASE(SymTagDimension)
    PRINT_ENUM_VALUE_CASE(SymTagCallSite)
    PRINT_ENUM_VALUE_CASE(SymTagInlineSite)
    PRINT_ENUM_VALUE_CASE(SymTagBaseInterface)
    PRINT_ENUM_VALUE_CASE(SymTagVectorType)
    PRINT_ENUM_VALUE_CASE(SymTagMatrixType)
    PRINT_ENUM_VALUE_CASE(SymTagHLSLType)
#if (_MSC_FULL_VER >= 180031101)
    PRINT_ENUM_VALUE_CASE(SymTagCaller)
    PRINT_ENUM_VALUE_CASE(SymTagCallee)
#endif
    PRINT_ENUM_VALUE_CASE(SymTagMax)
  }
  outs() << " {" << (DWORD)SymTag << "}";
  return Stream;
}

raw_ostream &llvm::operator<<(raw_ostream &Stream, CV_CPU_TYPE_e CpuType) {
  switch (CpuType) {
    PRINT_ENUM_VALUE_CASE(CV_CFL_8080)
    PRINT_ENUM_VALUE_CASE(CV_CFL_8086)
    PRINT_ENUM_VALUE_CASE(CV_CFL_80286)
    PRINT_ENUM_VALUE_CASE(CV_CFL_80386)
    PRINT_ENUM_VALUE_CASE(CV_CFL_80486)
    PRINT_ENUM_VALUE_CASE(CV_CFL_PENTIUM)
    PRINT_ENUM_VALUE_CASE(CV_CFL_PENTIUMPRO)
    PRINT_ENUM_VALUE_CASE(CV_CFL_PENTIUMIII)
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPS)
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPS16)
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPS32)
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPS64)
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPSI)
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPSII)
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPSIII)
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPSIV)
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPSV)
    PRINT_ENUM_VALUE_CASE(CV_CFL_M68000)
    PRINT_ENUM_VALUE_CASE(CV_CFL_M68010)
    PRINT_ENUM_VALUE_CASE(CV_CFL_M68020)
    PRINT_ENUM_VALUE_CASE(CV_CFL_M68030)
    PRINT_ENUM_VALUE_CASE(CV_CFL_M68040)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ALPHA)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ALPHA_21164)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ALPHA_21164A)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ALPHA_21264)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ALPHA_21364)
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPC601)
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPC603)
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPC604)
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPC620)
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPCFP)
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPCBE)
    PRINT_ENUM_VALUE_CASE(CV_CFL_SH3)
    PRINT_ENUM_VALUE_CASE(CV_CFL_SH3E)
    PRINT_ENUM_VALUE_CASE(CV_CFL_SH3DSP)
    PRINT_ENUM_VALUE_CASE(CV_CFL_SH4)
    PRINT_ENUM_VALUE_CASE(CV_CFL_SHMEDIA)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM3)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM4)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM4T)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM5)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM5T)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM6)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM_XMAC)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM_WMMX)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM7)
    PRINT_ENUM_VALUE_CASE(CV_CFL_OMNI)
    PRINT_ENUM_VALUE_CASE(CV_CFL_IA64)
    PRINT_ENUM_VALUE_CASE(CV_CFL_IA64_2)
    PRINT_ENUM_VALUE_CASE(CV_CFL_CEE)
    PRINT_ENUM_VALUE_CASE(CV_CFL_AM33)
    PRINT_ENUM_VALUE_CASE(CV_CFL_M32R)
    PRINT_ENUM_VALUE_CASE(CV_CFL_TRICORE)
    PRINT_ENUM_VALUE_CASE(CV_CFL_X64)
    PRINT_ENUM_VALUE_CASE(CV_CFL_EBC)
    PRINT_ENUM_VALUE_CASE(CV_CFL_THUMB)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARMNT)
#if (_MSC_FULL_VER >= 180031101)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM64)
#endif
    PRINT_ENUM_VALUE_CASE(CV_CFL_D3D11_SHADER)
  }
  outs() << " {" << llvm::format_hex((DWORD)CpuType, 2, true) << "}";
  return Stream;
}

raw_ostream &llvm::operator<<(raw_ostream &Stream,
                              MachineTypeEnum MachineType) {
  switch (MachineType) {
    PRINT_ENUM_VALUE_CASE(MachineTypeUnknown)
    PRINT_ENUM_VALUE_CASE(MachineTypeX86)
    PRINT_ENUM_VALUE_CASE(MachineTypeR3000)
    PRINT_ENUM_VALUE_CASE(MachineTypeR4000)
    PRINT_ENUM_VALUE_CASE(MachineTypeR10000)
    PRINT_ENUM_VALUE_CASE(MachineTypeWCEMIPSv2)
    PRINT_ENUM_VALUE_CASE(MachineTypeAlpha)
    PRINT_ENUM_VALUE_CASE(MachineTypeSH3)
    PRINT_ENUM_VALUE_CASE(MachineTypeSH3DSP)
    PRINT_ENUM_VALUE_CASE(MachineTypeSH3E)
    PRINT_ENUM_VALUE_CASE(MachineTypeSH4)
    PRINT_ENUM_VALUE_CASE(MachineTypeSH5)
    PRINT_ENUM_VALUE_CASE(MachineTypeArm)
    PRINT_ENUM_VALUE_CASE(MachineTypeThumb)
    PRINT_ENUM_VALUE_CASE(MachineTypeArmNT)
    PRINT_ENUM_VALUE_CASE(MachineTypeAM33)
    PRINT_ENUM_VALUE_CASE(MachineTypePowerPC)
    PRINT_ENUM_VALUE_CASE(MachineTypePowerPCFP)
    PRINT_ENUM_VALUE_CASE(MachineTypeIa64)
    PRINT_ENUM_VALUE_CASE(MachineTypeMips16)
    PRINT_ENUM_VALUE_CASE(MachineTypeAlpha64)
    PRINT_ENUM_VALUE_CASE(MachineTypeMipsFpu)
    PRINT_ENUM_VALUE_CASE(MachineTypeMipsFpu16)
    PRINT_ENUM_VALUE_CASE(MachineTypeTriCore)
    PRINT_ENUM_VALUE_CASE(MachineTypeCEF)
    PRINT_ENUM_VALUE_CASE(MachineTypeEBC)
    PRINT_ENUM_VALUE_CASE(MachineTypeAmd64)
    PRINT_ENUM_VALUE_CASE(MachineTypeM32R)
    PRINT_ENUM_VALUE_CASE(MachineTypeCEE)
  }
  outs() << " {" << llvm::format_hex((DWORD)MachineType, 2, true) << "}";
  return Stream;
}
