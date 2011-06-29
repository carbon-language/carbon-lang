//===-- TargetInstrInfo.cpp - Target Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  TargetInstrInfo
//===----------------------------------------------------------------------===//

TargetInstrInfo::TargetInstrInfo(const MCInstrDesc* Desc, unsigned numOpcodes,
                                 int CFSetupOpcode, int CFDestroyOpcode)
  : CallFrameSetupOpcode(CFSetupOpcode),
    CallFrameDestroyOpcode(CFDestroyOpcode) {
  InitMCInstrInfo(Desc, numOpcodes);
}

TargetInstrInfo::~TargetInstrInfo() {
}

const TargetRegisterClass*
TargetInstrInfo::getRegClass(const MCInstrDesc &MCID, unsigned OpNum,
                             const TargetRegisterInfo *TRI) const {
  if (OpNum >= MCID.getNumOperands())
    return 0;

  short RegClass = MCID.OpInfo[OpNum].RegClass;
  if (MCID.OpInfo[OpNum].isLookupPtrRegClass())
    return TRI->getPointerRegClass(RegClass);

  // Instructions like INSERT_SUBREG do not have fixed register classes.
  if (RegClass < 0)
    return 0;

  // Otherwise just look it up normally.
  return TRI->getRegClass(RegClass);
}

unsigned
TargetInstrInfo::getNumMicroOps(const InstrItineraryData *ItinData,
                                const MachineInstr *MI) const {
  if (!ItinData || ItinData->isEmpty())
    return 1;

  unsigned Class = MI->getDesc().getSchedClass();
  unsigned UOps = ItinData->Itineraries[Class].NumMicroOps;
  if (UOps)
    return UOps;

  // The # of u-ops is dynamically determined. The specific target should
  // override this function to return the right number.
  return 1;
}

int
TargetInstrInfo::getOperandLatency(const InstrItineraryData *ItinData,
                             const MachineInstr *DefMI, unsigned DefIdx,
                             const MachineInstr *UseMI, unsigned UseIdx) const {
  if (!ItinData || ItinData->isEmpty())
    return -1;

  unsigned DefClass = DefMI->getDesc().getSchedClass();
  unsigned UseClass = UseMI->getDesc().getSchedClass();
  return ItinData->getOperandLatency(DefClass, DefIdx, UseClass, UseIdx);
}

int
TargetInstrInfo::getOperandLatency(const InstrItineraryData *ItinData,
                                   SDNode *DefNode, unsigned DefIdx,
                                   SDNode *UseNode, unsigned UseIdx) const {
  if (!ItinData || ItinData->isEmpty())
    return -1;

  if (!DefNode->isMachineOpcode())
    return -1;

  unsigned DefClass = get(DefNode->getMachineOpcode()).getSchedClass();
  if (!UseNode->isMachineOpcode())
    return ItinData->getOperandCycle(DefClass, DefIdx);
  unsigned UseClass = get(UseNode->getMachineOpcode()).getSchedClass();
  return ItinData->getOperandLatency(DefClass, DefIdx, UseClass, UseIdx);
}

int TargetInstrInfo::getInstrLatency(const InstrItineraryData *ItinData,
                                     const MachineInstr *MI,
                                     unsigned *PredCost) const {
  if (!ItinData || ItinData->isEmpty())
    return 1;

  return ItinData->getStageLatency(MI->getDesc().getSchedClass());
}

int TargetInstrInfo::getInstrLatency(const InstrItineraryData *ItinData,
                                     SDNode *N) const {
  if (!ItinData || ItinData->isEmpty())
    return 1;

  if (!N->isMachineOpcode())
    return 1;

  return ItinData->getStageLatency(get(N->getMachineOpcode()).getSchedClass());
}

bool TargetInstrInfo::hasLowDefLatency(const InstrItineraryData *ItinData,
                                       const MachineInstr *DefMI,
                                       unsigned DefIdx) const {
  if (!ItinData || ItinData->isEmpty())
    return false;

  unsigned DefClass = DefMI->getDesc().getSchedClass();
  int DefCycle = ItinData->getOperandCycle(DefClass, DefIdx);
  return (DefCycle != -1 && DefCycle <= 1);
}

/// insertNoop - Insert a noop into the instruction stream at the specified
/// point.
void TargetInstrInfo::insertNoop(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI) const {
  llvm_unreachable("Target didn't implement insertNoop!");
}


bool TargetInstrInfo::isUnpredicatedTerminator(const MachineInstr *MI) const {
  const MCInstrDesc &MCID = MI->getDesc();
  if (!MCID.isTerminator()) return false;

  // Conditional branch is a special case.
  if (MCID.isBranch() && !MCID.isBarrier())
    return true;
  if (!MCID.isPredicable())
    return true;
  return !isPredicated(MI);
}


/// Measure the specified inline asm to determine an approximation of its
/// length.
/// Comments (which run till the next SeparatorString or newline) do not
/// count as an instruction.
/// Any other non-whitespace text is considered an instruction, with
/// multiple instructions separated by SeparatorString or newlines.
/// Variable-length instructions are not handled here; this function
/// may be overloaded in the target code to do that.
unsigned TargetInstrInfo::getInlineAsmLength(const char *Str,
                                             const MCAsmInfo &MAI) const {


  // Count the number of instructions in the asm.
  bool atInsnStart = true;
  unsigned Length = 0;
  for (; *Str; ++Str) {
    if (*Str == '\n' || strncmp(Str, MAI.getSeparatorString(),
                                strlen(MAI.getSeparatorString())) == 0)
      atInsnStart = true;
    if (atInsnStart && !std::isspace(*Str)) {
      Length += MAI.getMaxInstLength();
      atInsnStart = false;
    }
    if (atInsnStart && strncmp(Str, MAI.getCommentString(),
                               strlen(MAI.getCommentString())) == 0)
      atInsnStart = false;
  }

  return Length;
}
