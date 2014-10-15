//===-------- MipsELFStreamer.cpp - ELF Object Output ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsELFStreamer.h"
#include "llvm/MC/MCInst.h"
using namespace llvm;

void MipsELFStreamer::EmitInstruction(const MCInst &Inst,
                                      const MCSubtargetInfo &STI) {
  MCELFStreamer::EmitInstruction(Inst, STI);

  MCContext &Context = getContext();
  const MCRegisterInfo *MCRegInfo = Context.getRegisterInfo();

  for (unsigned OpIndex = 0; OpIndex < Inst.getNumOperands(); ++OpIndex) {
    const MCOperand &Op = Inst.getOperand(OpIndex);

    if (!Op.isReg())
      continue;

    unsigned Reg = Op.getReg();
    RegInfoRecord->SetPhysRegUsed(Reg, MCRegInfo);
  }
}

void MipsELFStreamer::EmitMipsOptionRecords() {
  for (const auto &I : MipsOptionRecords)
    I->EmitMipsOptionRecord();
}

namespace llvm {
MCELFStreamer *createMipsELFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                     raw_ostream &OS, MCCodeEmitter *Emitter,
                                     const MCSubtargetInfo &STI,
                                     bool RelaxAll) {
  return new MipsELFStreamer(Context, MAB, OS, Emitter, STI);
}
}
