//===-- PPCMCTargetDesc.h - PowerPC Target Descriptions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides PowerPC specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef PPCMCTARGETDESC_H
#define PPCMCTARGETDESC_H

#include <string>

namespace llvm {
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCSubtargetInfo;
class Target;
class TargetAsmBackend;
class StringRef;

extern Target ThePPC32Target;
extern Target ThePPC64Target;
  
MCCodeEmitter *createPPCMCCodeEmitter(const MCInstrInfo &MCII,
                                      const MCSubtargetInfo &STI,
                                      MCContext &Ctx);

TargetAsmBackend *createPPCAsmBackend(const Target &, const std::string &);
  
} // End llvm namespace

// Defines symbolic names for PowerPC registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "PPCGenRegisterInfo.inc"

// Defines symbolic names for the PowerPC instructions.
//
#define GET_INSTRINFO_ENUM
#include "PPCGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "PPCGenSubtargetInfo.inc"

#endif
