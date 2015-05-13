//===- HexagonMCInstrInfo.cpp - Hexagon sub-class of MCInst ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class extends MCInstrInfo to allow Hexagon specific MCInstr queries
//
//===----------------------------------------------------------------------===//

#include "HexagonMCInstrInfo.h"
#include "HexagonBaseInfo.h"

namespace llvm {
void HexagonMCInstrInfo::AppendImplicitOperands(MCInst &MCI) {
  MCI.addOperand(MCOperand::createImm(0));
  MCI.addOperand(MCOperand::createInst(nullptr));
}

HexagonII::MemAccessSize
HexagonMCInstrInfo::getAccessSize(MCInstrInfo const &MCII, MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;

  return (HexagonII::MemAccessSize((F >> HexagonII::MemAccessSizePos) &
                                   HexagonII::MemAccesSizeMask));
}

unsigned HexagonMCInstrInfo::getBitCount(MCInstrInfo const &MCII,
                                         MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask);
}

// Return constant extended operand number.
unsigned short HexagonMCInstrInfo::getCExtOpNum(MCInstrInfo const &MCII,
                                                MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask);
}

MCInstrDesc const &HexagonMCInstrInfo::getDesc(MCInstrInfo const &MCII,
                                               MCInst const &MCI) {
  return (MCII.get(MCI.getOpcode()));
}

unsigned HexagonMCInstrInfo::getExtentAlignment(MCInstrInfo const &MCII,
                                                MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtentAlignPos) & HexagonII::ExtentAlignMask);
}

unsigned HexagonMCInstrInfo::getExtentBits(MCInstrInfo const &MCII,
                                           MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask);
}

std::bitset<16> HexagonMCInstrInfo::GetImplicitBits(MCInst const &MCI) {
  SanityCheckImplicitOperands(MCI);
  std::bitset<16> Bits(MCI.getOperand(MCI.getNumOperands() - 2).getImm());
  return Bits;
}

// Return the max value that a constant extendable operand can have
// without being extended.
int HexagonMCInstrInfo::getMaxValue(MCInstrInfo const &MCII,
                                    MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  unsigned isSigned =
      (F >> HexagonII::ExtentSignedPos) & HexagonII::ExtentSignedMask;
  unsigned bits = (F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask;

  if (isSigned) // if value is signed
    return ~(-1U << (bits - 1));
  else
    return ~(-1U << bits);
}

// Return the min value that a constant extendable operand can have
// without being extended.
int HexagonMCInstrInfo::getMinValue(MCInstrInfo const &MCII,
                                    MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  unsigned isSigned =
      (F >> HexagonII::ExtentSignedPos) & HexagonII::ExtentSignedMask;
  unsigned bits = (F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask;

  if (isSigned) // if value is signed
    return -1U << (bits - 1);
  else
    return 0;
}

char const *HexagonMCInstrInfo::getName(MCInstrInfo const &MCII,
                                        MCInst const &MCI) {
  return MCII.getName(MCI.getOpcode());
}

// Return the operand that consumes or produces a new value.
MCOperand const &HexagonMCInstrInfo::getNewValue(MCInstrInfo const &MCII,
                                                 MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  unsigned const O =
      (F >> HexagonII::NewValueOpPos) & HexagonII::NewValueOpMask;
  MCOperand const &MCO = MCI.getOperand(O);

  assert((HexagonMCInstrInfo::isNewValue(MCII, MCI) ||
          HexagonMCInstrInfo::hasNewValue(MCII, MCI)) &&
         MCO.isReg());
  return (MCO);
}

// Return the Hexagon ISA class for the insn.
unsigned HexagonMCInstrInfo::getType(MCInstrInfo const &MCII,
                                     MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;

  return ((F >> HexagonII::TypePos) & HexagonII::TypeMask);
}

// Return whether the instruction is a legal new-value producer.
bool HexagonMCInstrInfo::hasNewValue(MCInstrInfo const &MCII,
                                     MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::hasNewValuePos) & HexagonII::hasNewValueMask);
}

// Return whether the insn is an actual insn.
bool HexagonMCInstrInfo::isCanon(MCInstrInfo const &MCII, MCInst const &MCI) {
  return (!HexagonMCInstrInfo::getDesc(MCII, MCI).isPseudo() &&
          !HexagonMCInstrInfo::isPrefix(MCII, MCI) &&
          HexagonMCInstrInfo::getType(MCII, MCI) != HexagonII::TypeENDLOOP);
}

// Return whether the instruction needs to be constant extended.
// 1) Always return true if the instruction has 'isExtended' flag set.
//
// isExtendable:
// 2) For immediate extended operands, return true only if the value is
//    out-of-range.
// 3) For global address, always return true.

bool HexagonMCInstrInfo::isConstExtended(MCInstrInfo const &MCII,
                                         MCInst const &MCI) {
  if (HexagonMCInstrInfo::isExtended(MCII, MCI))
    return true;

  if (!HexagonMCInstrInfo::isExtendable(MCII, MCI))
    return false;

  short ExtOpNum = HexagonMCInstrInfo::getCExtOpNum(MCII, MCI);
  int MinValue = HexagonMCInstrInfo::getMinValue(MCII, MCI);
  int MaxValue = HexagonMCInstrInfo::getMaxValue(MCII, MCI);
  MCOperand const &MO = MCI.getOperand(ExtOpNum);

  // We could be using an instruction with an extendable immediate and shoehorn
  // a global address into it. If it is a global address it will be constant
  // extended. We do this for COMBINE.
  // We currently only handle isGlobal() because it is the only kind of
  // object we are going to end up with here for now.
  // In the future we probably should add isSymbol(), etc.
  if (MO.isExpr())
    return true;

  // If the extendable operand is not 'Immediate' type, the instruction should
  // have 'isExtended' flag set.
  assert(MO.isImm() && "Extendable operand must be Immediate type");

  int ImmValue = MO.getImm();
  return (ImmValue < MinValue || ImmValue > MaxValue);
}

// Return true if the instruction may be extended based on the operand value.
bool HexagonMCInstrInfo::isExtendable(MCInstrInfo const &MCII,
                                      MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return (F >> HexagonII::ExtendablePos) & HexagonII::ExtendableMask;
}

// Return whether the instruction must be always extended.
bool HexagonMCInstrInfo::isExtended(MCInstrInfo const &MCII,
                                    MCInst const &MCI) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return (F >> HexagonII::ExtendedPos) & HexagonII::ExtendedMask;
}

// Return whether the insn is a new-value consumer.
bool HexagonMCInstrInfo::isNewValue(MCInstrInfo const &MCII,
                                    MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::NewValuePos) & HexagonII::NewValueMask);
}

// Return whether the operand can be constant extended.
bool HexagonMCInstrInfo::isOperandExtended(MCInstrInfo const &MCII,
                                           MCInst const &MCI,
                                           unsigned short OperandNum) {
  uint64_t const F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask) ==
         OperandNum;
}

bool HexagonMCInstrInfo::isPacketBegin(MCInst const &MCI) {
  std::bitset<16> Bits(GetImplicitBits(MCI));
  return Bits.test(packetBeginIndex);
}

bool HexagonMCInstrInfo::isPacketEnd(MCInst const &MCI) {
  std::bitset<16> Bits(GetImplicitBits(MCI));
  return Bits.test(packetEndIndex);
}

// Return whether the insn is a prefix.
bool HexagonMCInstrInfo::isPrefix(MCInstrInfo const &MCII, MCInst const &MCI) {
  return (HexagonMCInstrInfo::getType(MCII, MCI) == HexagonII::TypePREFIX);
}

// Return whether the insn is solo, i.e., cannot be in a packet.
bool HexagonMCInstrInfo::isSolo(MCInstrInfo const &MCII, MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::SoloPos) & HexagonII::SoloMask);
}

void HexagonMCInstrInfo::resetPacket(MCInst &MCI) {
  setPacketBegin(MCI, false);
  setPacketEnd(MCI, false);
}

void HexagonMCInstrInfo::SetImplicitBits(MCInst &MCI, std::bitset<16> Bits) {
  SanityCheckImplicitOperands(MCI);
  MCI.getOperand(MCI.getNumOperands() - 2).setImm(Bits.to_ulong());
}

void HexagonMCInstrInfo::setPacketBegin(MCInst &MCI, bool f) {
  std::bitset<16> Bits(GetImplicitBits(MCI));
  Bits.set(packetBeginIndex, f);
  SetImplicitBits(MCI, Bits);
}

void HexagonMCInstrInfo::setPacketEnd(MCInst &MCI, bool f) {
  std::bitset<16> Bits(GetImplicitBits(MCI));
  Bits.set(packetEndIndex, f);
  SetImplicitBits(MCI, Bits);
}
}
