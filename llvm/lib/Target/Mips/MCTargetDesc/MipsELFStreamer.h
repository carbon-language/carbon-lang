//===-------- MipsELFStreamer.h - ELF Object Output -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a custom MCELFStreamer which allows us to insert some hooks before
// emitting data into an actual object file.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSELFSTREAMER_H
#define MIPSELFSTREAMER_H

#include "MipsOptionRecord.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCELFStreamer.h"
#include <memory>

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCSubtargetInfo;

class MipsELFStreamer : public MCELFStreamer {
  SmallVector<std::unique_ptr<MipsOptionRecord>, 8> MipsOptionRecords;
  MipsRegInfoRecord *RegInfoRecord;

public:
  MipsELFStreamer(MCContext &Context, MCAsmBackend &MAB, raw_ostream &OS,
                  MCCodeEmitter *Emitter, const MCSubtargetInfo &STI)
      : MCELFStreamer(Context, MAB, OS, Emitter) {

    RegInfoRecord = new MipsRegInfoRecord(this, Context, STI);
    MipsOptionRecords.push_back(
        std::unique_ptr<MipsRegInfoRecord>(RegInfoRecord));
  }

  /// Overriding this function allows us to add arbitrary behaviour before the
  /// \p Inst is actually emitted. For example, we can inspect the operands and
  /// gather sufficient information that allows us to reason about the register
  /// usage for the translation unit.
  void EmitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;

  /// Emits all the option records stored up until the point it's called.
  void EmitMipsOptionRecords();
};

MCELFStreamer *createMipsELFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                     raw_ostream &OS, MCCodeEmitter *Emitter,
                                     const MCSubtargetInfo &STI, bool RelaxAll,
                                     bool NoExecStack);
} // namespace llvm.
#endif
