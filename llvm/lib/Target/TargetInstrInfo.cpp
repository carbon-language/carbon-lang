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
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  TargetInstrInfo
//===----------------------------------------------------------------------===//

TargetInstrInfo::~TargetInstrInfo() {
}

const TargetRegisterClass*
TargetInstrInfo::getRegClass(const MCInstrDesc &MCID, unsigned OpNum,
                             const TargetRegisterInfo *TRI,
                             const MachineFunction &MF) const {
  if (OpNum >= MCID.getNumOperands())
    return 0;

  short RegClass = MCID.OpInfo[OpNum].RegClass;
  if (MCID.OpInfo[OpNum].isLookupPtrRegClass())
    return TRI->getPointerRegClass(MF, RegClass);

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

/// Return the default expected latency for a def based on it's opcode.
unsigned TargetInstrInfo::defaultDefLatency(const InstrItineraryData *ItinData,
                                            const MachineInstr *DefMI) const {
  if (DefMI->mayLoad())
    return ItinData->Props.LoadLatency;
  if (isHighLatencyDef(DefMI->getOpcode()))
    return ItinData->Props.HighLatency;
  return 1;
}

/// Both DefMI and UseMI must be valid.  By default, call directly to the
/// itinerary. This may be overriden by the target.
int
TargetInstrInfo::getOperandLatency(const InstrItineraryData *ItinData,
                                   const MachineInstr *DefMI, unsigned DefIdx,
                                   const MachineInstr *UseMI,
                                   unsigned UseIdx) const {
  unsigned DefClass = DefMI->getDesc().getSchedClass();
  unsigned UseClass = UseMI->getDesc().getSchedClass();
  return ItinData->getOperandLatency(DefClass, DefIdx, UseClass, UseIdx);
}

/// If we can determine the operand latency from the def only, without itinerary
/// lookup, do so. Otherwise return -1.
static int computeDefOperandLatency(
  const TargetInstrInfo *TII, const InstrItineraryData *ItinData,
  const MachineInstr *DefMI, bool FindMin) {

  // Let the target hook getInstrLatency handle missing itineraries.
  if (!ItinData)
    return TII->getInstrLatency(ItinData, DefMI);

  // Return a latency based on the itinerary properties and defining instruction
  // if possible. Some common subtargets don't require per-operand latency,
  // especially for minimum latencies.
  if (FindMin) {
    // If MinLatency is valid, call getInstrLatency. This uses Stage latency if
    // it exists before defaulting to MinLatency.
    if (ItinData->Props.MinLatency >= 0)
      return TII->getInstrLatency(ItinData, DefMI);

    // If MinLatency is invalid, OperandLatency is interpreted as MinLatency.
    // For empty itineraries, short-cirtuit the check and default to one cycle.
    if (ItinData->isEmpty())
      return 1;
  }
  else if(ItinData->isEmpty())
    return TII->defaultDefLatency(ItinData, DefMI);

  // ...operand lookup required
return -1;
}

/// computeOperandLatency - Compute and return the latency of the given data
/// dependent def and use when the operand indices are already known.
///
/// FindMin may be set to get the minimum vs. expected latency.
unsigned TargetInstrInfo::
computeOperandLatency(const InstrItineraryData *ItinData,
                      const MachineInstr *DefMI, unsigned DefIdx,
                      const MachineInstr *UseMI, unsigned UseIdx,
                      bool FindMin) const {

  int DefLatency = computeDefOperandLatency(this, ItinData, DefMI, FindMin);
  if (DefLatency >= 0)
    return DefLatency;

  assert(ItinData && !ItinData->isEmpty() && "computeDefOperandLatency fail");

  int OperLatency = getOperandLatency(ItinData, DefMI, DefIdx, UseMI, UseIdx);
  if (OperLatency >= 0)
    return OperLatency;

  // No operand latency was found.
  unsigned InstrLatency = getInstrLatency(ItinData, DefMI);

  // Expected latency is the max of the stage latency and itinerary props.
  if (!FindMin)
    InstrLatency = std::max(InstrLatency, defaultDefLatency(ItinData, DefMI));
  return InstrLatency;
}

/// computeOperandLatency - Compute and return the latency of the given data
/// dependent def and use. DefMI must be a valid def. UseMI may be NULL for an
/// unknown use. Depending on the subtarget's itinerary properties, this may or
/// may not need to call getOperandLatency().
///
/// FindMin may be set to get the minimum vs. expected latency. Minimum
/// latency is used for scheduling groups, while expected latency is for
/// instruction cost and critical path.
///
/// For most subtargets, we don't need DefIdx or UseIdx to compute min latency.
/// DefMI must be a valid definition, but UseMI may be NULL for an unknown use.
unsigned TargetInstrInfo::
computeOperandLatency(const InstrItineraryData *ItinData,
                      const TargetRegisterInfo *TRI,
                      const MachineInstr *DefMI, const MachineInstr *UseMI,
                      unsigned Reg, bool FindMin) const {

  int DefLatency = computeDefOperandLatency(this, ItinData, DefMI, FindMin);
  if (DefLatency >= 0)
    return DefLatency;

  assert(ItinData && !ItinData->isEmpty() && "computeDefOperandLatency fail");

  // Find the definition of the register in the defining instruction.
  int DefIdx = DefMI->findRegisterDefOperandIdx(Reg);
  if (DefIdx != -1) {
    const MachineOperand &MO = DefMI->getOperand(DefIdx);
    if (MO.isReg() && MO.isImplicit() &&
        DefIdx >= (int)DefMI->getDesc().getNumOperands()) {
      // This is an implicit def, getOperandLatency() won't return the correct
      // latency. e.g.
      //   %D6<def>, %D7<def> = VLD1q16 %R2<kill>, 0, ..., %Q3<imp-def>
      //   %Q1<def> = VMULv8i16 %Q1<kill>, %Q3<kill>, ...
      // What we want is to compute latency between def of %D6/%D7 and use of
      // %Q3 instead.
      unsigned Op2 = DefMI->findRegisterDefOperandIdx(Reg, false, true, TRI);
      if (DefMI->getOperand(Op2).isReg())
        DefIdx = Op2;
    }
    // For all uses of the register, calculate the maxmimum latency
    int OperLatency = -1;

    // UseMI is null, then it must be a scheduling barrier.
    if (!UseMI) {
      unsigned DefClass = DefMI->getDesc().getSchedClass();
      OperLatency = ItinData->getOperandCycle(DefClass, DefIdx);
    }
    else {
      for (unsigned i = 0, e = UseMI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = UseMI->getOperand(i);
        if (!MO.isReg() || !MO.isUse())
          continue;
        unsigned MOReg = MO.getReg();
        if (MOReg != Reg)
          continue;

        int UseCycle = getOperandLatency(ItinData, DefMI, DefIdx, UseMI, i);
        OperLatency = std::max(OperLatency, UseCycle);
      }
    }
    // If we found an operand latency, we're done.
    if (OperLatency >= 0)
      return OperLatency;
  }
  // No operand latency was found.
  unsigned InstrLatency = getInstrLatency(ItinData, DefMI);

  // Expected latency is the max of the stage latency and itinerary props.
  if (!FindMin)
    InstrLatency = std::max(InstrLatency, defaultDefLatency(ItinData, DefMI));
  return InstrLatency;
}

unsigned TargetInstrInfo::getInstrLatency(const InstrItineraryData *ItinData,
                                          const MachineInstr *MI,
                                          unsigned *PredCost) const {
  // Default to one cycle for no itinerary. However, an "empty" itinerary may
  // still have a MinLatency property, which getStageLatency checks.
  if (!ItinData)
    return MI->mayLoad() ? 2 : 1;

  return ItinData->getStageLatency(MI->getDesc().getSchedClass());
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
