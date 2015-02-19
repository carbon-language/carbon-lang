//===- HexagonMCInst.cpp - Hexagon sub-class of MCInst --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class extends MCInst to allow some Hexagon VLIW annotations.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCInst.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"

using namespace llvm;

HexagonMCInst::HexagonMCInst() : MCII (createHexagonMCInstrInfo ()) {}
HexagonMCInst::HexagonMCInst(MCInstrDesc const &mcid) :
  MCII (createHexagonMCInstrInfo ()){}

void HexagonMCInst::AppendImplicitOperands(MCInst &MCI) {
  MCI.addOperand(MCOperand::CreateImm(0));
  MCI.addOperand(MCOperand::CreateInst(nullptr));
}

std::bitset<16> HexagonMCInst::GetImplicitBits(MCInst const &MCI) {
  SanityCheckImplicitOperands(MCI);
  std::bitset<16> Bits(MCI.getOperand(MCI.getNumOperands() - 2).getImm());
  return Bits;
}

void HexagonMCInst::SetImplicitBits(MCInst &MCI, std::bitset<16> Bits) {
  SanityCheckImplicitOperands(MCI);
  MCI.getOperand(MCI.getNumOperands() - 2).setImm(Bits.to_ulong());
}

void HexagonMCInst::setPacketBegin(bool f) {
  std::bitset<16> Bits(GetImplicitBits(*this));
  Bits.set(packetBeginIndex, f);
  SetImplicitBits(*this, Bits);
}

bool HexagonMCInst::isPacketBegin() const {
  std::bitset<16> Bits(GetImplicitBits(*this));
  return Bits.test(packetBeginIndex);
}

void HexagonMCInst::setPacketEnd(bool f) {
  std::bitset<16> Bits(GetImplicitBits(*this));
  Bits.set(packetEndIndex, f);
  SetImplicitBits(*this, Bits);
}

bool HexagonMCInst::isPacketEnd() const {
  std::bitset<16> Bits(GetImplicitBits(*this));
  return Bits.test(packetEndIndex);
}

void HexagonMCInst::resetPacket() {
  setPacketBegin(false);
  setPacketEnd(false);
}
