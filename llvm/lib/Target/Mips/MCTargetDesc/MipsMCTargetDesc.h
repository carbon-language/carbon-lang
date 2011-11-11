//===-- MipsMCTargetDesc.h - Mips Target Descriptions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Mips specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSMCTARGETDESC_H
#define MIPSMCTARGETDESC_H

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCSubtargetInfo;
class StringRef;
class Target;
class raw_ostream;

extern Target TheMipsTarget;
extern Target TheMipselTarget;
extern Target TheMips64Target;
extern Target TheMips64elTarget;

MCCodeEmitter *createMipsMCCodeEmitter(const MCInstrInfo &MCII,
                                       const MCSubtargetInfo &STI,
                                       MCContext &Ctx);

MCAsmBackend *createMipsAsmBackend(const Target &T, StringRef TT);

} // End llvm namespace

// Defines symbolic names for Mips registers.  This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "MipsGenRegisterInfo.inc"

// Defines symbolic names for the Mips instructions.
#define GET_INSTRINFO_ENUM
#include "MipsGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "MipsGenSubtargetInfo.inc"

#endif
