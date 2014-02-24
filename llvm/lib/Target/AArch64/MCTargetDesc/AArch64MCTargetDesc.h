//===-- AArch64MCTargetDesc.h - AArch64 Target Descriptions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides AArch64 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AARCH64MCTARGETDESC_H
#define LLVM_AARCH64MCTARGETDESC_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class StringRef;
class Target;
class raw_ostream;

extern Target TheAArch64leTarget;
extern Target TheAArch64beTarget;

namespace AArch64_MC {
  MCSubtargetInfo *createAArch64MCSubtargetInfo(StringRef TT, StringRef CPU,
                                                StringRef FS);
}

MCCodeEmitter *createAArch64MCCodeEmitter(const MCInstrInfo &MCII,
                                          const MCRegisterInfo &MRI,
                                          const MCSubtargetInfo &STI,
                                          MCContext &Ctx);

MCObjectWriter *createAArch64ELFObjectWriter(raw_ostream &OS,
                                             uint8_t OSABI,
                                             bool IsLittleEndian);

MCAsmBackend *createAArch64leAsmBackend(const Target &T,
                                        const MCRegisterInfo &MRI,
                                        StringRef TT, StringRef CPU);

MCAsmBackend *createAArch64beAsmBackend(const Target &T,
                                        const MCRegisterInfo &MRI,
                                        StringRef TT, StringRef CPU);

} // End llvm namespace

// Defines symbolic names for AArch64 registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "AArch64GenRegisterInfo.inc"

// Defines symbolic names for the AArch64 instructions.
//
#define GET_INSTRINFO_ENUM
#include "AArch64GenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "AArch64GenSubtargetInfo.inc"

#endif
