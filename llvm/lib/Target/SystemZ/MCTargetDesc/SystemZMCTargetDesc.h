//===-- SystemZMCTargetDesc.h - SystemZ target descriptions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEMZMCTARGETDESC_H
#define SYSTEMZMCTARGETDESC_H

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

extern Target TheSystemZTarget;

namespace SystemZMC {
  // How many bytes are in the ABI-defined, caller-allocated part of
  // a stack frame.
  const int64_t CallFrameSize = 160;

  // The offset of the DWARF CFA from the incoming stack pointer.
  const int64_t CFAOffsetFromInitialSP = CallFrameSize;
}

MCCodeEmitter *createSystemZMCCodeEmitter(const MCInstrInfo &MCII,
                                          const MCRegisterInfo &MRI,
                                          const MCSubtargetInfo &STI,
                                          MCContext &Ctx);

MCAsmBackend *createSystemZMCAsmBackend(const Target &T, StringRef TT,
                                        StringRef CPU);

MCObjectWriter *createSystemZObjectWriter(raw_ostream &OS, uint8_t OSABI);
} // end namespace llvm

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
