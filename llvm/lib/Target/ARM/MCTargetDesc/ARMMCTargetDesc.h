//===-- ARMMCTargetDesc.h - ARM Target Descriptions -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides ARM specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef ARMMCTARGETDESC_H
#define ARMMCTARGETDESC_H

namespace llvm {
class Target;

extern Target TheARMTarget, TheThumbTarget;
} // End llvm namespace

// Defines symbolic names for ARM registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "ARMGenRegisterInfo.inc"

// Defines symbolic names for the ARM instructions.
//
#define GET_INSTRINFO_ENUM
#include "ARMGenInstrInfo.inc"

#endif
