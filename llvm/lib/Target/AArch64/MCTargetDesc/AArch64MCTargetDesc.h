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

#ifndef AArch64MCTARGETDESC_H
#define AArch64MCTARGETDESC_H

#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCRegisterInfo;
class MCObjectWriter;
class MCSubtargetInfo;
class StringRef;
class Target;
class raw_ostream;

extern Target TheAArch64leTarget;
extern Target TheAArch64beTarget;
extern Target TheARM64leTarget;
extern Target TheARM64beTarget;

MCCodeEmitter *createAArch64MCCodeEmitter(const MCInstrInfo &MCII,
                                        const MCRegisterInfo &MRI,
                                        const MCSubtargetInfo &STI,
                                        MCContext &Ctx);
MCAsmBackend *createAArch64leAsmBackend(const Target &T,
                                        const MCRegisterInfo &MRI, StringRef TT,
                                        StringRef CPU);
MCAsmBackend *createAArch64beAsmBackend(const Target &T,
                                        const MCRegisterInfo &MRI, StringRef TT,
                                        StringRef CPU);

MCObjectWriter *createAArch64ELFObjectWriter(raw_ostream &OS, uint8_t OSABI,
                                             bool IsLittleEndian);

MCObjectWriter *createAArch64MachObjectWriter(raw_ostream &OS, uint32_t CPUType,
                                            uint32_t CPUSubtype);

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
