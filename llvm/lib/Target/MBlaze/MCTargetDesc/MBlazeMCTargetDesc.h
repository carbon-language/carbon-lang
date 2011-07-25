//===-- MBlazeMCTargetDesc.h - MBlaze Target Descriptions -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides MBlaze specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZEMCTARGETDESC_H
#define MBLAZEMCTARGETDESC_H

#include <string>

namespace llvm {
class MCContext;
class MCCodeEmitter;
class MCInstrInfo;
class MCSubtargetInfo;
class Target;
class TargetAsmBackend;
class StringRef;
class formatted_raw_ostream;

extern Target TheMBlazeTarget;

MCCodeEmitter *createMBlazeMCCodeEmitter(const MCInstrInfo &MCII,
                                         const MCSubtargetInfo &STI,
                                         MCContext &Ctx);
  
TargetAsmBackend *createMBlazeAsmBackend(const Target &, const std::string &);

} // End llvm namespace

// Defines symbolic names for MBlaze registers.  This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "MBlazeGenRegisterInfo.inc"

// Defines symbolic names for the MBlaze instructions.
#define GET_INSTRINFO_ENUM
#include "MBlazeGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "MBlazeGenSubtargetInfo.inc"

#endif
