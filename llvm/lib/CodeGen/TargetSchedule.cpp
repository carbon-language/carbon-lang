//===-- llvm/Target/TargetSchedule.cpp - Sched Machine Model ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a wrapper around MCSchedModel that allows the interface
// to benefit from information currently only available in TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool> EnableSchedModel("schedmodel", cl::Hidden, cl::init(true),
  cl::desc("Use TargetSchedModel for latency lookup"));

static cl::opt<bool> EnableSchedItins("scheditins", cl::Hidden, cl::init(true),
  cl::desc("Use InstrItineraryData for latency lookup"));

bool TargetSchedModel::hasInstrSchedModel() const {
  return EnableSchedModel && SchedModel.hasInstrSchedModel();
}

bool TargetSchedModel::hasInstrItineraries() const {
  return EnableSchedItins && !InstrItins.isEmpty();
}

void TargetSchedModel::init(const MCSchedModel &sm,
                            const TargetSubtargetInfo *sti,
                            const TargetInstrInfo *tii) {
  SchedModel = sm;
  STI = sti;
  TII = tii;
  STI->initInstrItins(InstrItins);
}

unsigned TargetSchedModel::getNumMicroOps(MachineInstr *MI) const {
  if (hasInstrItineraries()) {
    int UOps = InstrItins.getNumMicroOps(MI->getDesc().getSchedClass());
    return (UOps >= 0) ? UOps : TII->getNumMicroOps(&InstrItins, MI);
  }
  if (hasInstrSchedModel())
    return resolveSchedClass(MI)->NumMicroOps;

  return 1;
}

/// If we can determine the operand latency from the def only, without machine
/// model or itinerary lookup, do so. Otherwise return -1.
int TargetSchedModel::getDefLatency(const MachineInstr *DefMI,
                                    bool FindMin) const {

  // Return a latency based on the itinerary properties and defining instruction
  // if possible. Some common subtargets don't require per-operand latency,
  // especially for minimum latencies.
  if (FindMin) {
    // If MinLatency is invalid, then use the itinerary for MinLatency. If no
    // itinerary exists either, then use single cycle latency.
    if (SchedModel.MinLatency < 0 && !hasInstrItineraries()) {
      return 1;
    }
    return SchedModel.MinLatency;
  }
  else if (!hasInstrSchedModel() && !hasInstrItineraries()) {
    return TII->defaultDefLatency(&SchedModel, DefMI);
  }
  // ...operand lookup required
  return -1;
}

/// Return the MCSchedClassDesc for this instruction. Some SchedClasses require
/// evaluation of predicates that depend on instruction operands or flags.
const MCSchedClassDesc *TargetSchedModel::
resolveSchedClass(const MachineInstr *MI) const {

  // Get the definition's scheduling class descriptor from this machine model.
  unsigned SchedClass = MI->getDesc().getSchedClass();
  const MCSchedClassDesc *SCDesc = SchedModel.getSchedClassDesc(SchedClass);

#ifndef NDEBUG
  unsigned NIter = 0;
#endif
  while (SCDesc->isVariant()) {
    assert(++NIter < 6 && "Variants are nested deeper than the magic number");

    SchedClass = STI->resolveSchedClass(SchedClass, MI, this);
    SCDesc = SchedModel.getSchedClassDesc(SchedClass);
  }
  return SCDesc;
}

/// Find the def index of this operand. This index maps to the machine model and
/// is independent of use operands. Def operands may be reordered with uses or
/// merged with uses without affecting the def index (e.g. before/after
/// regalloc). However, an instruction's def operands must never be reordered
/// with respect to each other.
static unsigned findDefIdx(const MachineInstr *MI, unsigned DefOperIdx) {
  unsigned DefIdx = 0;
  for (unsigned i = 0; i != DefOperIdx; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef())
      ++DefIdx;
  }
  return DefIdx;
}

/// Find the use index of this operand. This is independent of the instruction's
/// def operands.
///
/// Note that uses are not determined by the operand's isUse property, which
/// is simply the inverse of isDef. Here we consider any readsReg operand to be
/// a "use". The machine model allows an operand to be both a Def and Use.
static unsigned findUseIdx(const MachineInstr *MI, unsigned UseOperIdx) {
  unsigned UseIdx = 0;
  for (unsigned i = 0; i != UseOperIdx; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.readsReg())
      ++UseIdx;
  }
  return UseIdx;
}

// Top-level API for clients that know the operand indices.
unsigned TargetSchedModel::computeOperandLatency(
  const MachineInstr *DefMI, unsigned DefOperIdx,
  const MachineInstr *UseMI, unsigned UseOperIdx,
  bool FindMin) const {

  int DefLatency = getDefLatency(DefMI, FindMin);
  if (DefLatency >= 0)
    return DefLatency;

  if (hasInstrItineraries()) {
    int OperLatency = 0;
    if (UseMI) {
      OperLatency =
        TII->getOperandLatency(&InstrItins, DefMI, DefOperIdx, UseMI, UseOperIdx);
    }
    else {
      unsigned DefClass = DefMI->getDesc().getSchedClass();
      OperLatency = InstrItins.getOperandCycle(DefClass, DefOperIdx);
    }
    if (OperLatency >= 0)
      return OperLatency;

    // No operand latency was found.
    unsigned InstrLatency = TII->getInstrLatency(&InstrItins, DefMI);

    // Expected latency is the max of the stage latency and itinerary props.
    // Rather than directly querying InstrItins stage latency, we call a TII
    // hook to allow subtargets to specialize latency. This hook is only
    // applicable to the InstrItins model. InstrSchedModel should model all
    // special cases without TII hooks.
    if (!FindMin)
      InstrLatency = std::max(InstrLatency,
                              TII->defaultDefLatency(&SchedModel, DefMI));
    return InstrLatency;
  }
  assert(!FindMin && hasInstrSchedModel() &&
         "Expected a SchedModel for this cpu");
  const MCSchedClassDesc *SCDesc = resolveSchedClass(DefMI);
  unsigned DefIdx = findDefIdx(DefMI, DefOperIdx);
  if (DefIdx < SCDesc->NumWriteLatencyEntries) {
    // Lookup the definition's write latency in SubtargetInfo.
    const MCWriteLatencyEntry *WLEntry =
      STI->getWriteLatencyEntry(SCDesc, DefIdx);
    unsigned WriteID = WLEntry->WriteResourceID;
    unsigned Latency = WLEntry->Cycles;
    if (!UseMI)
      return Latency;

    // Lookup the use's latency adjustment in SubtargetInfo.
    const MCSchedClassDesc *UseDesc = resolveSchedClass(UseMI);
    if (UseDesc->NumReadAdvanceEntries == 0)
      return Latency;
    unsigned UseIdx = findUseIdx(UseMI, UseOperIdx);
    return Latency - STI->getReadAdvanceCycles(UseDesc, UseIdx, WriteID);
  }
  // If DefIdx does not exist in the model (e.g. implicit defs), then return
  // unit latency (defaultDefLatency may be too conservative).
#ifndef NDEBUG
  if (SCDesc->isValid() && !DefMI->getOperand(DefOperIdx).isImplicit()
      && !DefMI->getDesc().OpInfo[DefOperIdx].isOptionalDef()) {
    std::string Err;
    raw_string_ostream ss(Err);
    ss << "DefIdx " << DefIdx << " exceeds machine model writes for "
       << *DefMI;
    report_fatal_error(ss.str());
  }
#endif
  return 1;
}

unsigned TargetSchedModel::computeInstrLatency(const MachineInstr *MI) const {
  if (hasInstrItineraries()) {
    // For the itinerary model, fall back to the old subtarget hook.
    return TII->getInstrLatency(&InstrItins, MI);
  }
  if (hasInstrSchedModel()) {
    unsigned Latency = 0;
    const MCSchedClassDesc *SCDesc = resolveSchedClass(MI);
    for (unsigned DefIdx = 0, DefEnd = SCDesc->NumWriteLatencyEntries;
         DefIdx != DefEnd; ++DefIdx) {
      // Lookup the definition's write latency in SubtargetInfo.
      const MCWriteLatencyEntry *WLEntry =
        STI->getWriteLatencyEntry(SCDesc, DefIdx);
      Latency = std::max(Latency, WLEntry->Cycles);
    }
    return Latency;
  }
  return TII->defaultDefLatency(&SchedModel, MI);
}

unsigned TargetSchedModel::
computeOutputLatency(const MachineInstr *DefMI, unsigned DefOperIdx,
                     const MachineInstr *DepMI) const {
  // MinLatency == -1 is for in-order processors that always have unit
  // MinLatency. MinLatency > 0 is for in-order processors with varying min
  // latencies, but since this is not a RAW dep, we always use unit latency.
  if (SchedModel.MinLatency != 0)
    return 1;

  // MinLatency == 0 indicates an out-of-order processor that can dispatch
  // WAW dependencies in the same cycle.

  // Treat predication as a data dependency for out-of-order cpus. In-order
  // cpus do not need to treat predicated writes specially.
  //
  // TODO: The following hack exists because predication passes do not
  // correctly append imp-use operands, and readsReg() strangely returns false
  // for predicated defs.
  unsigned Reg = DefMI->getOperand(DefOperIdx).getReg();
  const MachineFunction &MF = *DefMI->getParent()->getParent();
  const TargetRegisterInfo *TRI = MF.getTarget().getRegisterInfo();
  if (!DepMI->readsRegister(Reg, TRI) && TII->isPredicated(DepMI))
    return computeInstrLatency(DefMI);

  // If we have a per operand scheduling model, check if this def is writing
  // an unbuffered resource. If so, it treated like an in-order cpu.
  if (hasInstrSchedModel()) {
    const MCSchedClassDesc *SCDesc = resolveSchedClass(DefMI);
    for (const MCWriteProcResEntry *PRI = STI->getWriteProcResBegin(SCDesc),
           *PRE = STI->getWriteProcResEnd(SCDesc); PRI != PRE; ++PRI) {
      if (!SchedModel.getProcResource(PRI->ProcResourceIdx)->IsBuffered)
        return 1;
    }
  }
  return 0;
}
