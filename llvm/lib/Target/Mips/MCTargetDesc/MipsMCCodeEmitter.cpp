//===-- MipsMCCodeEmitter.cpp - Convert Mips code to machine code ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MipsMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//
//
#define DEBUG_TYPE "mccodeemitter"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/raw_ostream.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"

using namespace llvm;

namespace {
class MipsMCCodeEmitter : public MCCodeEmitter {
  MipsMCCodeEmitter(const MipsMCCodeEmitter &); // DO NOT IMPLEMENT
  void operator=(const MipsMCCodeEmitter &); // DO NOT IMPLEMENT
  const MCInstrInfo &MCII;
  const MCSubtargetInfo &STI;

public:
  MipsMCCodeEmitter(const MCInstrInfo &mcii, const MCSubtargetInfo &sti,
                    MCContext &ctx)
    : MCII(mcii), STI(sti) {}

  ~MipsMCCodeEmitter() {}

  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const {
  }
}; // class MipsMCCodeEmitter
}  // namespace

MCCodeEmitter *llvm::createMipsMCCodeEmitter(const MCInstrInfo &MCII,
                                             const MCSubtargetInfo &STI,
                                             MCContext &Ctx) {
  return new MipsMCCodeEmitter(MCII, STI, Ctx);
}
