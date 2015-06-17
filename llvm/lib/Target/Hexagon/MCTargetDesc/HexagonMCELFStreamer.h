//===- HexagonMCELFStreamer.h - Hexagon subclass of MCElfStreamer ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONMCELFSTREAMER_H
#define HEXAGONMCELFSTREAMER_H

#include "MCTargetDesc/HexagonMCCodeEmitter.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/MC/MCELFStreamer.h"
#include "HexagonTargetStreamer.h"

namespace llvm {

class HexagonMCELFStreamer : public MCELFStreamer {
  std::unique_ptr<MCInstrInfo> MCII;

public:
  HexagonMCELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                       raw_pwrite_stream &OS, MCCodeEmitter *Emitter)
      : MCELFStreamer(Context, TAB, OS, Emitter),
        MCII(createHexagonMCInstrInfo()) {}

  virtual void EmitInstruction(const MCInst &Inst,
                               const MCSubtargetInfo &STI) override;
  void EmitSymbol(const MCInst &Inst);
  void HexagonMCEmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                      unsigned ByteAlignment,
                                      unsigned AccessSize);
  void HexagonMCEmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                 unsigned ByteAlignment, unsigned AccessSize);
};

MCStreamer *createHexagonELFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                     raw_pwrite_stream &OS, MCCodeEmitter *CE);

} // namespace llvm

#endif
