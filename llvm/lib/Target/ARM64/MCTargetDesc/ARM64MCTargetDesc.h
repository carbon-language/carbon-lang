//===-- ARM64MCTargetDesc.h - ARM64 Target Descriptions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides ARM64 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef ARM64MCTARGETDESC_H
#define ARM64MCTARGETDESC_H

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

extern Target TheARM64leTarget;
extern Target TheARM64beTarget;
extern Target TheAArch64leTarget;
extern Target TheAArch64beTarget;

MCCodeEmitter *createARM64MCCodeEmitter(const MCInstrInfo &MCII,
                                        const MCRegisterInfo &MRI,
                                        const MCSubtargetInfo &STI,
                                        MCContext &Ctx);
MCAsmBackend *createARM64leAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                      StringRef TT, StringRef CPU);
MCAsmBackend *createARM64beAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                      StringRef TT, StringRef CPU);

        MCObjectWriter *createARM64ELFObjectWriter(raw_ostream &OS, uint8_t OSABI,
                                                   bool IsLittleEndian);

MCObjectWriter *createARM64MachObjectWriter(raw_ostream &OS, uint32_t CPUType,
                                            uint32_t CPUSubtype);

} // End llvm namespace

// Defines symbolic names for ARM64 registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "ARM64GenRegisterInfo.inc"

// Defines symbolic names for the ARM64 instructions.
//
#define GET_INSTRINFO_ENUM
#include "ARM64GenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "ARM64GenSubtargetInfo.inc"

#endif
