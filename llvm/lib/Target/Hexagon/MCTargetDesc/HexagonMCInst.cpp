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

#include "HexagonInstrInfo.h"
#include "HexagonTargetMachine.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCInst.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

HexagonMCInst::HexagonMCInst(unsigned op)
    : packetBegin(false), packetEnd(false),
      MCID(llvm::TheHexagonTarget.createMCInstrInfo()->get(op)) {
  assert(MCID.getSize() == 4 && "All instructions should be 32bit");
  setOpcode(op);
}

bool HexagonMCInst::isPacketBegin() const { return packetBegin; }
bool HexagonMCInst::isPacketEnd() const { return packetEnd; }
void HexagonMCInst::setPacketEnd(bool Y) { packetEnd = Y; }
void HexagonMCInst::setPacketBegin(bool Y) { packetBegin = Y; }

unsigned HexagonMCInst::getUnits(HexagonTargetMachine const &TM) const {
  const HexagonInstrInfo *QII = TM.getSubtargetImpl()->getInstrInfo();
  const InstrItineraryData *II = TM.getSubtargetImpl()->getInstrItineraryData();
  const InstrStage *IS =
      II->beginStage(QII->get(this->getOpcode()).getSchedClass());

  return (IS->getUnits());
}

bool HexagonMCInst::isNewValue() const {
  const uint64_t F = MCID.TSFlags;
  return ((F >> HexagonII::NewValuePos) & HexagonII::NewValueMask);
}

bool HexagonMCInst::hasNewValue() const {
  const uint64_t F = MCID.TSFlags;
  return ((F >> HexagonII::hasNewValuePos) & HexagonII::hasNewValueMask);
}

MCOperand const &HexagonMCInst::getNewValue() const {
  const uint64_t F = MCID.TSFlags;
  const unsigned O =
      (F >> HexagonII::NewValueOpPos) & HexagonII::NewValueOpMask;
  const MCOperand &MCO = getOperand(O);

  assert((isNewValue() || hasNewValue()) && MCO.isReg());
  return (MCO);
}

// Return whether the instruction needs to be constant extended.
// 1) Always return true if the instruction has 'isExtended' flag set.
//
// isExtendable:
// 2) For immediate extended operands, return true only if the value is
//    out-of-range.
// 3) For global address, always return true.
bool HexagonMCInst::isConstExtended(void) const {
  if (isExtended())
    return true;

  if (!isExtendable())
    return false;

  short ExtOpNum = getCExtOpNum();
  int MinValue = getMinValue();
  int MaxValue = getMaxValue();
  const MCOperand &MO = getOperand(ExtOpNum);

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

bool HexagonMCInst::isExtended(void) const {
  const uint64_t F = MCID.TSFlags;
  return (F >> HexagonII::ExtendedPos) & HexagonII::ExtendedMask;
}

bool HexagonMCInst::isExtendable(void) const {
  const uint64_t F = MCID.TSFlags;
  return (F >> HexagonII::ExtendablePos) & HexagonII::ExtendableMask;
}

unsigned HexagonMCInst::getBitCount(void) const {
  const uint64_t F = MCID.TSFlags;
  return ((F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask);
}

unsigned short HexagonMCInst::getCExtOpNum(void) const {
  const uint64_t F = MCID.TSFlags;
  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask);
}

bool HexagonMCInst::isOperandExtended(const unsigned short OperandNum) const {
  const uint64_t F = MCID.TSFlags;
  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask) ==
         OperandNum;
}

int HexagonMCInst::getMinValue(void) const {
  const uint64_t F = MCID.TSFlags;
  unsigned isSigned =
      (F >> HexagonII::ExtentSignedPos) & HexagonII::ExtentSignedMask;
  unsigned bits = (F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask;

  if (isSigned)
    return -1U << (bits - 1);
  else
    return 0;
}

int HexagonMCInst::getMaxValue(void) const {
  const uint64_t F = MCID.TSFlags;
  unsigned isSigned =
      (F >> HexagonII::ExtentSignedPos) & HexagonII::ExtentSignedMask;
  unsigned bits = (F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask;

  if (isSigned)
    return ~(-1U << (bits - 1));
  else
    return ~(-1U << bits);
}
