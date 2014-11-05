//===-------- MipsELFStreamer.cpp - ELF Object Output ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsELFStreamer.h"
#include "MipsTargetStreamer.h"
#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ELF.h"

using namespace llvm;

void MipsELFStreamer::EmitInstruction(const MCInst &Inst,
                                      const MCSubtargetInfo &STI) {
  MCELFStreamer::EmitInstruction(Inst, STI);

  MCContext &Context = getContext();
  const MCRegisterInfo *MCRegInfo = Context.getRegisterInfo();
  MipsTargetELFStreamer *ELFTargetStreamer =
      static_cast<MipsTargetELFStreamer *>(getTargetStreamer());

  for (unsigned OpIndex = 0; OpIndex < Inst.getNumOperands(); ++OpIndex) {
    const MCOperand &Op = Inst.getOperand(OpIndex);

    if (!Op.isReg())
      continue;

    unsigned Reg = Op.getReg();
    RegInfoRecord->SetPhysRegUsed(Reg, MCRegInfo);
  }

  if (ELFTargetStreamer->isMicroMipsEnabled()) {
    for (auto Label : Labels) {
      MCSymbolData &Data = getOrCreateSymbolData(Label);
      // The "other" values are stored in the last 6 bits of the second byte.
      // The traditional defines for STO values assume the full byte and thus
      // the shift to pack it.
      MCELF::setOther(Data, ELF::STO_MIPS_MICROMIPS >> 2);
    }
  }

  Labels.clear();
}

void MipsELFStreamer::EmitLabel(MCSymbol *Symbol) {
  MCELFStreamer::EmitLabel(Symbol);
  Labels.push_back(Symbol);
}

void MipsELFStreamer::SwitchSection(const MCSection * Section,
                                    const MCExpr *Subsection) {
  MCELFStreamer::SwitchSection(Section, Subsection);
  Labels.clear();
}

void MipsELFStreamer::EmitValueImpl(const MCExpr *Value, unsigned Size,
                                    const SMLoc &Loc) {
  MCELFStreamer::EmitValueImpl(Value, Size, Loc);
  Labels.clear();
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
