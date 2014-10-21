//===-- HexagonMCCodeEmitter.h - Hexagon Target Descriptions ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Definition for classes that emit Hexagon machine code from MCInsts
///
//===----------------------------------------------------------------------===//

#ifndef HEXAGONMCCODEEMITTER_H
#define HEXAGONMCCODEEMITTER_H

#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class HexagonMCCodeEmitter : public MCCodeEmitter {
  MCSubtargetInfo const &MST;
  MCContext &MCT;

public:
  HexagonMCCodeEmitter(MCInstrInfo const &aMII, MCSubtargetInfo const &aMST,
                       MCContext &aMCT);

  MCSubtargetInfo const &getSubtargetInfo() const;

  void EncodeInstruction(MCInst const &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups,
                         MCSubtargetInfo const &STI) const override;

  // \brief TableGen'erated function for getting the
  // binary encoding for an instruction.
  uint64_t getBinaryCodeForInstr(MCInst const &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 MCSubtargetInfo const &STI) const;

  /// \brief Return binary encoding of operand.
  unsigned getMachineOpValue(MCInst const &MI, MCOperand const &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             MCSubtargetInfo const &STI) const;

private:
  HexagonMCCodeEmitter(HexagonMCCodeEmitter const &) LLVM_DELETED_FUNCTION;
  void operator=(HexagonMCCodeEmitter const &) LLVM_DELETED_FUNCTION;
}; // class HexagonMCCodeEmitter

} // namespace llvm

#endif /* HEXAGONMCCODEEMITTER_H */
