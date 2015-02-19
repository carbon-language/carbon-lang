//===- HexagonMCInst.h - Hexagon sub-class of MCInst ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class extends MCInst to allow some VLIW annotations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINST_H
#define LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINST_H

#include "HexagonTargetMachine.h"
#include "llvm/MC/MCInst.h"
#include <memory>

extern "C" void LLVMInitializeHexagonTargetMC();
namespace llvm {
class MCOperand;

class HexagonMCInst : public MCInst {
  // Used to access TSFlags
  std::unique_ptr <MCInstrInfo const> MCII;

public:
  explicit HexagonMCInst();
  HexagonMCInst(const MCInstrDesc &mcid);

  static void AppendImplicitOperands(MCInst &MCI);
  static std::bitset<16> GetImplicitBits(MCInst const &MCI);
  static void SetImplicitBits(MCInst &MCI, std::bitset<16> Bits);
  static void SanityCheckImplicitOperands(MCInst const &MCI) {
    assert(MCI.getNumOperands() >= 2 && "At least the two implicit operands");
    assert(MCI.getOperand(MCI.getNumOperands() - 1).isInst() &&
           "Implicit bits and flags");
    assert(MCI.getOperand(MCI.getNumOperands() - 2).isImm() &&
           "Parent pointer");
  }

  void setPacketBegin(bool Y);
  bool isPacketBegin() const;
  static const size_t packetBeginIndex = 0;
  void setPacketEnd(bool Y);
  bool isPacketEnd() const;
  static const size_t packetEndIndex = 1;
  void resetPacket();
};
}

#endif
