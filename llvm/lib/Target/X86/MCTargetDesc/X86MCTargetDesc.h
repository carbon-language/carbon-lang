//===-- X86MCTargetDesc.h - X86 Target Descriptions -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides X86 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef X86MCTARGETDESC_H
#define X86MCTARGETDESC_H

#include <string>

namespace llvm {
class MCSubtargetInfo;
class Target;
class StringRef;

extern Target TheX86_32Target, TheX86_64Target;

namespace X86_MC {
  std::string ParseX86Triple(StringRef TT);

  /// GetCpuIDAndInfo - Execute the specified cpuid and return the 4 values in
  /// the specified arguments.  If we can't run cpuid on the host, return true.
  bool GetCpuIDAndInfo(unsigned value, unsigned *rEAX,
                       unsigned *rEBX, unsigned *rECX, unsigned *rEDX);

  void DetectFamilyModel(unsigned EAX, unsigned &Family, unsigned &Model);

  /// createARMMCSubtargetInfo - Create a X86 MCSubtargetInfo instance.
  /// This is exposed so Asm parser, etc. do not need to go through
  /// TargetRegistry.
  MCSubtargetInfo *createX86MCSubtargetInfo(StringRef TT, StringRef CPU,
                                            StringRef FS);
}

} // End llvm namespace


// Defines symbolic names for X86 registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "X86GenRegisterInfo.inc"

// Defines symbolic names for the X86 instructions.
//
#define GET_INSTRINFO_ENUM
#include "X86GenInstrInfo.inc"

#endif
