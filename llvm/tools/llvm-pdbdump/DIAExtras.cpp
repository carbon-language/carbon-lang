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

#define PRINT_ENUM_VALUE_CASE(Value, Name)                                     \
  case Value:                                                                  \
    outs() << Name;                                                            \
    break;

raw_ostream &llvm::operator<<(raw_ostream &Stream, DiaSymTagEnum SymTag) {
  switch (SymTag) {
    PRINT_ENUM_VALUE_CASE(SymTagNull, "Null")
    PRINT_ENUM_VALUE_CASE(SymTagExe, "Exe")
    PRINT_ENUM_VALUE_CASE(SymTagCompiland, "Compiland")
    PRINT_ENUM_VALUE_CASE(SymTagCompilandDetails, "CompilandDetails")
    PRINT_ENUM_VALUE_CASE(SymTagCompilandEnv, "CompilandEnv")
    PRINT_ENUM_VALUE_CASE(SymTagFunction, "Function")
    PRINT_ENUM_VALUE_CASE(SymTagBlock, "Block")
    PRINT_ENUM_VALUE_CASE(SymTagData, "Data")
    PRINT_ENUM_VALUE_CASE(SymTagAnnotation, "Annotation")
    PRINT_ENUM_VALUE_CASE(SymTagLabel, "Label")
    PRINT_ENUM_VALUE_CASE(SymTagPublicSymbol, "PublicSymbol")
    PRINT_ENUM_VALUE_CASE(SymTagUDT, "UDT")
    PRINT_ENUM_VALUE_CASE(SymTagEnum, "Enum")
    PRINT_ENUM_VALUE_CASE(SymTagFunctionType, "FunctionType")
    PRINT_ENUM_VALUE_CASE(SymTagPointerType, "PointerType")
    PRINT_ENUM_VALUE_CASE(SymTagArrayType, "ArrayType")
    PRINT_ENUM_VALUE_CASE(SymTagBaseType, "BaseType")
    PRINT_ENUM_VALUE_CASE(SymTagTypedef, "Typedef")
    PRINT_ENUM_VALUE_CASE(SymTagBaseClass, "BaseClass")
    PRINT_ENUM_VALUE_CASE(SymTagFriend, "Friend")
    PRINT_ENUM_VALUE_CASE(SymTagFunctionArgType, "FunctionArgType")
    PRINT_ENUM_VALUE_CASE(SymTagFuncDebugStart, "FuncDebugStart")
    PRINT_ENUM_VALUE_CASE(SymTagFuncDebugEnd, "FuncDebugEnd")
    PRINT_ENUM_VALUE_CASE(SymTagUsingNamespace, "UsingNamespace")
    PRINT_ENUM_VALUE_CASE(SymTagVTableShape, "VTableShape")
    PRINT_ENUM_VALUE_CASE(SymTagVTable, "VTable")
    PRINT_ENUM_VALUE_CASE(SymTagCustom, "Custom")
    PRINT_ENUM_VALUE_CASE(SymTagThunk, "Thunk")
    PRINT_ENUM_VALUE_CASE(SymTagCustomType, "CustomType")
    PRINT_ENUM_VALUE_CASE(SymTagManagedType, "ManagedType")
    PRINT_ENUM_VALUE_CASE(SymTagDimension, "Dimension")
    PRINT_ENUM_VALUE_CASE(SymTagCallSite, "CallSite")
    PRINT_ENUM_VALUE_CASE(SymTagInlineSite, "InlineSite")
    PRINT_ENUM_VALUE_CASE(SymTagBaseInterface, "BaseInterface")
    PRINT_ENUM_VALUE_CASE(SymTagVectorType, "VectorType")
    PRINT_ENUM_VALUE_CASE(SymTagMatrixType, "MatrixType")
    PRINT_ENUM_VALUE_CASE(SymTagHLSLType, "HLSLType")
#if (_MSC_FULL_VER >= 180031101)
    PRINT_ENUM_VALUE_CASE(SymTagCaller, "Caller")
    PRINT_ENUM_VALUE_CASE(SymTagCallee, "Callee")
#endif
    PRINT_ENUM_VALUE_CASE(SymTagMax, "Max")
  }
  outs() << " {" << (DWORD)SymTag << "}";
  return Stream;
}

raw_ostream &llvm::operator<<(raw_ostream &Stream, CV_CPU_TYPE_e CpuType) {
  switch (CpuType) {
    PRINT_ENUM_VALUE_CASE(CV_CFL_8080, "8080")
    PRINT_ENUM_VALUE_CASE(CV_CFL_8086, "8086")
    PRINT_ENUM_VALUE_CASE(CV_CFL_80286, "80286")
    PRINT_ENUM_VALUE_CASE(CV_CFL_80386, "80386")
    PRINT_ENUM_VALUE_CASE(CV_CFL_80486, "80486")
    PRINT_ENUM_VALUE_CASE(CV_CFL_PENTIUM, "Pentium")
    PRINT_ENUM_VALUE_CASE(CV_CFL_PENTIUMPRO, "Pentium Pro")
    PRINT_ENUM_VALUE_CASE(CV_CFL_PENTIUMIII, "Pentium III")
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPS, "MIPS")
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPS16, "MIPS 16")
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPS32, "MIPS 32")
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPS64, "MIPS 64")
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPSI, "MIPS I")
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPSII, "MIPS II")
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPSIII, "MIPS III")
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPSIV, "MIPS IV")
    PRINT_ENUM_VALUE_CASE(CV_CFL_MIPSV, "MIPS V")
    PRINT_ENUM_VALUE_CASE(CV_CFL_M68000, "M68000")
    PRINT_ENUM_VALUE_CASE(CV_CFL_M68010, "M68010")
    PRINT_ENUM_VALUE_CASE(CV_CFL_M68020, "M68020")
    PRINT_ENUM_VALUE_CASE(CV_CFL_M68030, "M68030")
    PRINT_ENUM_VALUE_CASE(CV_CFL_M68040, "M68040")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ALPHA, "Alpha")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ALPHA_21164, "Alpha 21164")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ALPHA_21164A, "Alpha 21164A")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ALPHA_21264, "Alpha 21264")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ALPHA_21364, "21364")
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPC601, "PowerPC 601")
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPC603, "PowerPC 603")
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPC604, "PowerPC 604")
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPC620, "PowerPC 620")
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPCFP, "PowerPC FP")
    PRINT_ENUM_VALUE_CASE(CV_CFL_PPCBE, "PowerPC BE")
    PRINT_ENUM_VALUE_CASE(CV_CFL_SH3, "SH3")
    PRINT_ENUM_VALUE_CASE(CV_CFL_SH3E, "SH3-E")
    PRINT_ENUM_VALUE_CASE(CV_CFL_SH3DSP, "SH3-DSP")
    PRINT_ENUM_VALUE_CASE(CV_CFL_SH4, "SH4")
    PRINT_ENUM_VALUE_CASE(CV_CFL_SHMEDIA, "SH Media")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM3, "ARM 3")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM4, "ARM 4")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM4T, "ARM 4T")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM5, "ARM 5")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM5T, "ARM 5T")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM6, "ARM 6")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM_XMAC, "ARM XMAC")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM_WMMX, "ARM WMMX")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM7, "ARM 7")
    PRINT_ENUM_VALUE_CASE(CV_CFL_OMNI, "Omni")
    PRINT_ENUM_VALUE_CASE(CV_CFL_IA64, "IA64")
    PRINT_ENUM_VALUE_CASE(CV_CFL_IA64_2, "IA64-2")
    PRINT_ENUM_VALUE_CASE(CV_CFL_CEE, "CEE")
    PRINT_ENUM_VALUE_CASE(CV_CFL_AM33, "AM33")
    PRINT_ENUM_VALUE_CASE(CV_CFL_M32R, "M32R")
    PRINT_ENUM_VALUE_CASE(CV_CFL_TRICORE, "TriCore")
    PRINT_ENUM_VALUE_CASE(CV_CFL_X64, "X64")
    PRINT_ENUM_VALUE_CASE(CV_CFL_EBC, "EBC")
    PRINT_ENUM_VALUE_CASE(CV_CFL_THUMB, "Thumb")
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARMNT, "ARM NT")
#if (_MSC_FULL_VER >= 180031101)
    PRINT_ENUM_VALUE_CASE(CV_CFL_ARM64, "ARM 64")
#endif
    PRINT_ENUM_VALUE_CASE(CV_CFL_D3D11_SHADER, "D3D11 Shader")
  }
  outs() << " {" << llvm::format_hex((DWORD)CpuType, 2, true) << "}";
  return Stream;
}

raw_ostream &llvm::operator<<(raw_ostream &Stream,
                              MachineTypeEnum MachineType) {
  switch (MachineType) {
    PRINT_ENUM_VALUE_CASE(MachineTypeUnknown, "Unknown")
    PRINT_ENUM_VALUE_CASE(MachineTypeX86, "x86")
    PRINT_ENUM_VALUE_CASE(MachineTypeR3000, "R3000")
    PRINT_ENUM_VALUE_CASE(MachineTypeR4000, "R4000")
    PRINT_ENUM_VALUE_CASE(MachineTypeR10000, "R10000")
    PRINT_ENUM_VALUE_CASE(MachineTypeWCEMIPSv2, "WCE MIPSv2")
    PRINT_ENUM_VALUE_CASE(MachineTypeAlpha, "Alpha")
    PRINT_ENUM_VALUE_CASE(MachineTypeSH3, "SH3")
    PRINT_ENUM_VALUE_CASE(MachineTypeSH3DSP, "SH3-DSP")
    PRINT_ENUM_VALUE_CASE(MachineTypeSH3E, "SH3-E")
    PRINT_ENUM_VALUE_CASE(MachineTypeSH4, "SH4")
    PRINT_ENUM_VALUE_CASE(MachineTypeSH5, "SH5")
    PRINT_ENUM_VALUE_CASE(MachineTypeArm, "ARM")
    PRINT_ENUM_VALUE_CASE(MachineTypeThumb, "Thumb")
    PRINT_ENUM_VALUE_CASE(MachineTypeArmNT, "ARM NT")
    PRINT_ENUM_VALUE_CASE(MachineTypeAM33, "AM33")
    PRINT_ENUM_VALUE_CASE(MachineTypePowerPC, "PowerPC")
    PRINT_ENUM_VALUE_CASE(MachineTypePowerPCFP, "PowerPC FP")
    PRINT_ENUM_VALUE_CASE(MachineTypeIa64, "IA 64")
    PRINT_ENUM_VALUE_CASE(MachineTypeMips16, "MIPS 16")
    PRINT_ENUM_VALUE_CASE(MachineTypeAlpha64, "Alpha 64")
    PRINT_ENUM_VALUE_CASE(MachineTypeMipsFpu, "FPU")
    PRINT_ENUM_VALUE_CASE(MachineTypeMipsFpu16, "FPU 16")
    PRINT_ENUM_VALUE_CASE(MachineTypeTriCore, "TriCore")
    PRINT_ENUM_VALUE_CASE(MachineTypeCEF, "CEF")
    PRINT_ENUM_VALUE_CASE(MachineTypeEBC, "EBC")
    PRINT_ENUM_VALUE_CASE(MachineTypeAmd64, "x64")
    PRINT_ENUM_VALUE_CASE(MachineTypeM32R, "M32R")
    PRINT_ENUM_VALUE_CASE(MachineTypeCEE, "CEE")
  }
  outs() << " {" << llvm::format_hex((DWORD)MachineType, 2, true) << "}";
  return Stream;
}

raw_ostream &llvm::operator<<(raw_ostream &Stream, HashAlgorithm Algorithm) {
  switch (Algorithm) {
    PRINT_ENUM_VALUE_CASE(HashNone, "None")
    PRINT_ENUM_VALUE_CASE(HashMD5, "MD5")
    PRINT_ENUM_VALUE_CASE(HashSHA1, "SHA-1")
  default:
    outs() << "(Unknown)";
    break;
  }
  outs() << " {" << (DWORD)Algorithm << "}";
  return Stream;
}
