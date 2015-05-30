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

#include "Hexagon.h"
#include "HexagonBaseInfo.h"
#include "HexagonMCInstrInfo.h"

namespace llvm {
iterator_range<MCInst::const_iterator>
HexagonMCInstrInfo::bundleInstructions(MCInst const &MCI) {
  assert(isBundle(MCI));
  return iterator_range<MCInst::const_iterator>(
      MCI.begin() + bundleInstructionsOffset, MCI.end());
}

size_t HexagonMCInstrInfo::bundleSize(MCInst const &MCI) {
  if (HexagonMCInstrInfo::isBundle(MCI))
    return (MCI.size() - bundleInstructionsOffset);
  else
    return (1);
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

unsigned short HexagonMCInstrInfo::getExtendableOp(MCInstrInfo const &MCII,
                                                   MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask);
}

MCOperand const &
HexagonMCInstrInfo::getExtendableOperand(MCInstrInfo const &MCII,
                                         MCInst const &MCI) {
  unsigned O = HexagonMCInstrInfo::getExtendableOp(MCII, MCI);
  MCOperand const &MO = MCI.getOperand(O);

  assert((HexagonMCInstrInfo::isExtendable(MCII, MCI) ||
          HexagonMCInstrInfo::isExtended(MCII, MCI)) &&
         (MO.isImm() || MO.isExpr()));
  return (MO);
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

unsigned short HexagonMCInstrInfo::getNewValueOp(MCInstrInfo const &MCII,
                                                 MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::NewValueOpPos) & HexagonII::NewValueOpMask);
}

MCOperand const &HexagonMCInstrInfo::getNewValueOperand(MCInstrInfo const &MCII,
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

MCInst const &HexagonMCInstrInfo::instruction(MCInst const &MCB, size_t Index) {
  assert(isBundle(MCB));
  assert(Index < HEXAGON_PACKET_SIZE);
  return *MCB.getOperand(bundleInstructionsOffset + Index).getInst();
}

bool HexagonMCInstrInfo::isBundle(MCInst const &MCI) {
  auto Result = Hexagon::BUNDLE == MCI.getOpcode();
  assert(!Result || (MCI.size() > 0 && MCI.getOperand(0).isImm()));
  return Result;
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

bool HexagonMCInstrInfo::isImmext(MCInst const &MCI) {
  auto Op = MCI.getOpcode();
  return (Op == Hexagon::A4_ext_b || Op == Hexagon::A4_ext_c ||
          Op == Hexagon::A4_ext_g || Op == Hexagon::A4_ext);
}

bool HexagonMCInstrInfo::isInnerLoop(MCInst const &MCI) {
  assert(isBundle(MCI));
  int64_t Flags = MCI.getOperand(0).getImm();
  return (Flags & innerLoopMask) != 0;
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

bool HexagonMCInstrInfo::isOuterLoop(MCInst const &MCI) {
  assert(isBundle(MCI));
  int64_t Flags = MCI.getOperand(0).getImm();
  return (Flags & outerLoopMask) != 0;
}

bool HexagonMCInstrInfo::isPredicated(MCInstrInfo const &MCII,
                                      MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return ((F >> HexagonII::PredicatedPos) & HexagonII::PredicatedMask);
}

bool HexagonMCInstrInfo::isPredicatedTrue(MCInstrInfo const &MCII,
                                          MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  return (
      !((F >> HexagonII::PredicatedFalsePos) & HexagonII::PredicatedFalseMask));
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

void HexagonMCInstrInfo::padEndloop(MCInst &MCB) {
  MCInst Nop;
  Nop.setOpcode(Hexagon::A2_nop);
  assert(isBundle(MCB));
  while ((HexagonMCInstrInfo::isInnerLoop(MCB) &&
          (HexagonMCInstrInfo::bundleSize(MCB) < HEXAGON_PACKET_INNER_SIZE)) ||
         ((HexagonMCInstrInfo::isOuterLoop(MCB) &&
           (HexagonMCInstrInfo::bundleSize(MCB) < HEXAGON_PACKET_OUTER_SIZE))))
    MCB.addOperand(MCOperand::createInst(new MCInst(Nop)));
}

void HexagonMCInstrInfo::setInnerLoop(MCInst &MCI) {
  assert(isBundle(MCI));
  MCOperand &Operand = MCI.getOperand(0);
  Operand.setImm(Operand.getImm() | innerLoopMask);
}

void HexagonMCInstrInfo::setOuterLoop(MCInst &MCI) {
  assert(isBundle(MCI));
  MCOperand &Operand = MCI.getOperand(0);
  Operand.setImm(Operand.getImm() | outerLoopMask);
}
}
