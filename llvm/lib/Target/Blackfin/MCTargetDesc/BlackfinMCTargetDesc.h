//===-- BlackfinMCTargetDesc.h - Blackfin Target Descriptions ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Blackfin specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef BLACKFINMCTARGETDESC_H
#define BLACKFINMCTARGETDESC_H

namespace llvm {
class MCSubtargetInfo;
class Target;
class StringRef;

extern Target TheBlackfinTarget;

} // End llvm namespace

// Defines symbolic names for Blackfin registers.  This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "BlackfinGenRegisterInfo.inc"

// Defines symbolic names for the Blackfin instructions.
#define GET_INSTRINFO_ENUM
#include "BlackfinGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "BlackfinGenSubtargetInfo.inc"

#endif
