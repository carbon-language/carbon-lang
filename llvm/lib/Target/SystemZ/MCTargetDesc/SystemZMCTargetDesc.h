//===-- SystemZMCTargetDesc.h - SystemZ Target Descriptions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides SystemZ specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEMZMCTARGETDESC_H
#define SYSTEMZMCTARGETDESC_H

namespace llvm {
class MCSubtargetInfo;
class Target;
class StringRef;

extern Target TheSystemZTarget;

} // End llvm namespace

// Defines symbolic names for SystemZ registers.
// This defines a mapping from register name to register number.
#define GET_REGINFO_ENUM
#include "SystemZGenRegisterInfo.inc"

// Defines symbolic names for the SystemZ instructions.
#define GET_INSTRINFO_ENUM
#include "SystemZGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "SystemZGenSubtargetInfo.inc"

#endif
