//===-- PTXMCTargetDesc.h - PTX Target Descriptions ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides PTX specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef PTXMCTARGETDESC_H
#define PTXMCTARGETDESC_H

namespace llvm {
class Target;

extern Target ThePTX32Target;
extern Target ThePTX64Target;

} // End llvm namespace

// Defines symbolic names for PTX registers.
#define GET_REGINFO_ENUM
#include "PTXGenRegisterInfo.inc"

// Defines symbolic names for the PTX instructions.
#define GET_INSTRINFO_ENUM
#include "PTXGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "PTXGenSubtargetInfo.inc"

#endif
