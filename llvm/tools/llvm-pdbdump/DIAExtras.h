//===- DIAExtras.h - Helper classes and functions for accessing DIA C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines helper types, classes, and functions for working with DIA.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_DIAEXTRAS_H
#define LLVM_TOOLS_LLVMPDBDUMP_DIAEXTRAS_H

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include "COMExtras.h"

namespace llvm {
namespace sys {
namespace windows {

typedef llvm::SmallString<16> DIAString;

template <class T> class DIAResult {
public:
  DIAResult() : IsValid(false) {}
  DIAResult(const T &ResultValue) : Result(ResultValue), IsValid(true) {}

  bool hasValue() const { return IsValid; }
  T value() const { return Result; }

  void dump(StringRef Name, unsigned IndentLevel) const {
    if (!hasValue())
      return;
    outs().indent(IndentLevel);
    outs() << Name << ": " << value() << "\n";
  }

private:
  T Result;
  bool IsValid;
};

template <>
void DIAResult<BOOL>::dump(StringRef Name, unsigned IndentLevel) const {
  if (!hasValue())
    return;
  outs().indent(IndentLevel);
  outs() << Name << ": " << (value() ? "true" : "false") << "\n";
}

template <>
void DIAResult<GUID>::dump(StringRef Name, unsigned IndentLevel) const {
  if (!hasValue())
    return;
  std::string Guid8;
  CComBSTR GuidBSTR(value());
  BSTRToUTF8(GuidBSTR.m_str, Guid8);

  outs().indent(IndentLevel);
  outs() << Name << ": " << Guid8 << "\n";
}

// MSDN documents the IDiaSymbol::get_machineType method as returning a value
// from the CV_CPU_TYPE_e enumeration.  This documentation is wrong, however,
// and this method actually returns a value from the IMAGE_FILE_xxx set of
// defines from winnt.h.  These correspond to the platform magic number in
// the COFF file.  This enumeration is built from the set of values in winnt.h
enum MachineTypeEnum {
  MachineTypeUnknown = IMAGE_FILE_MACHINE_UNKNOWN,
  MachineTypeX86 = IMAGE_FILE_MACHINE_I386,
  MachineTypeR3000 = IMAGE_FILE_MACHINE_R3000,
  MachineTypeR4000 = IMAGE_FILE_MACHINE_R4000,
  MachineTypeR10000 = IMAGE_FILE_MACHINE_R10000,
  MachineTypeWCEMIPSv2 = IMAGE_FILE_MACHINE_WCEMIPSV2,
  MachineTypeAlpha = IMAGE_FILE_MACHINE_ALPHA,
  MachineTypeSH3 = IMAGE_FILE_MACHINE_SH3,
  MachineTypeSH3DSP = IMAGE_FILE_MACHINE_SH3DSP,
  MachineTypeSH3E = IMAGE_FILE_MACHINE_SH3E,
  MachineTypeSH4 = IMAGE_FILE_MACHINE_SH4,
  MachineTypeSH5 = IMAGE_FILE_MACHINE_SH5,
  MachineTypeArm = IMAGE_FILE_MACHINE_ARM,
  MachineTypeThumb = IMAGE_FILE_MACHINE_THUMB,
  MachineTypeArmNT = IMAGE_FILE_MACHINE_ARMNT,
  MachineTypeAM33 = IMAGE_FILE_MACHINE_AM33,
  MachineTypePowerPC = IMAGE_FILE_MACHINE_POWERPC,
  MachineTypePowerPCFP = IMAGE_FILE_MACHINE_POWERPCFP,
  MachineTypeIa64 = IMAGE_FILE_MACHINE_IA64,
  MachineTypeMips16 = IMAGE_FILE_MACHINE_MIPS16,
  MachineTypeAlpha64 = IMAGE_FILE_MACHINE_ALPHA64,
  MachineTypeMipsFpu = IMAGE_FILE_MACHINE_MIPSFPU,
  MachineTypeMipsFpu16 = IMAGE_FILE_MACHINE_MIPSFPU16,
  MachineTypeTriCore = IMAGE_FILE_MACHINE_TRICORE,
  MachineTypeCEF = IMAGE_FILE_MACHINE_CEF,
  MachineTypeEBC = IMAGE_FILE_MACHINE_EBC,
  MachineTypeAmd64 = IMAGE_FILE_MACHINE_AMD64,
  MachineTypeM32R = IMAGE_FILE_MACHINE_M32R,
  MachineTypeCEE = IMAGE_FILE_MACHINE_CEE,
};

enum HashAlgorithm { HashNone = 0, HashMD5 = 1, HashSHA1 = 2 };

// SymTagEnum has the unfortunate property that it is not only the name of
// the enum, but also the name of one of the values of the enum.  So that we
// don't have to always type "enum SymTagEnum", we typedef this to a different
// name so that we can refer to it more easily.
typedef enum SymTagEnum DiaSymTagEnum;

typedef CComPtr<IDiaSymbol> DiaSymbolPtr;

} // namespace windows
} // namespace sys
} // namespace llvm

namespace llvm {
class raw_ostream;

raw_ostream &operator<<(raw_ostream &Stream,
                        llvm::sys::windows::DiaSymTagEnum SymTag);
raw_ostream &operator<<(raw_ostream &Stream, CV_CPU_TYPE_e CpuType);
raw_ostream &operator<<(raw_ostream &Stream,
                        llvm::sys::windows::MachineTypeEnum CpuType);
raw_ostream &operator<<(raw_ostream &Stream,
                        llvm::sys::windows::HashAlgorithm Algorithm);
}

#endif
