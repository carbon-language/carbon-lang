//===-- MBlaze.h - Top-level interface for MBlaze ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM MBlaze back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_MBLAZE_H
#define TARGET_MBLAZE_H

#include "MCTargetDesc/MBlazeMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class MBlazeTargetMachine;
  class FunctionPass;
  class MachineCodeEmitter;
  class MCCodeEmitter;
  class MCInstrInfo;
  class MCSubtargetInfo;
  class TargetAsmBackend;
  class formatted_raw_ostream;

  MCCodeEmitter *createMBlazeMCCodeEmitter(const MCInstrInfo &MCII,
                                           const MCSubtargetInfo &STI,
                                           MCContext &Ctx);
  
  TargetAsmBackend *createMBlazeAsmBackend(const Target &, const std::string &);

  FunctionPass *createMBlazeISelDag(MBlazeTargetMachine &TM);
  FunctionPass *createMBlazeDelaySlotFillerPass(MBlazeTargetMachine &TM);

} // end namespace llvm;

#endif
