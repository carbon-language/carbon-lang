//===----- SchedulePostRAList.cpp - list scheduler ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a top-down list scheduler, using standard algorithms.
// The basic approach uses a priority queue of available nodes to schedule.
// One at a time, nodes are taken from the priority queue (thus in priority
// order), checked for legality to schedule, and emitted if legal.
//
// Nodes may not be legal to schedule either due to structural hazards (e.g.
// pipeline or resource constraints) or because an input to the instruction has
// not completed execution.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "post-RA-sched"
#include "ExactHazardRecognizer.h"
#include "SimpleHazardRecognizer.h"
#include "ScheduleDAGInstrs.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LatencyPriorityQueue.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtarget.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
#include <map>
#include <set>
using namespace llvm;

STATISTIC(NumNoops, "Number of noops inserted");
STATISTIC(NumStalls, "Number of pipeline stalls");

// Post-RA scheduling is enabled with
// TargetSubtarget.enablePostRAScheduler(). This flag can be used to
// override the target.
static cl::opt<bool>
EnablePostRAScheduler("post-RA-scheduler",
                       cl::desc("Enable scheduling after register allocation"),
                       cl::init(false), cl::Hidden);
static cl::opt<bool>
EnableAntiDepBreaking("break-anti-dependencies",
                      cl::desc("Break post-RA scheduling anti-dependencies"),
                      cl::init(true), cl::Hidden);
static cl::opt<bool>
EnablePostRAHazardAvoidance("avoid-hazards",
                      cl::desc("Enable exact hazard avoidance"),
                      cl::init(true), cl::Hidden);

// If DebugDiv > 0 then only schedule MBB with (ID % DebugDiv) == DebugMod
static cl::opt<int>
DebugDiv("postra-sched-debugdiv",
                      cl::desc("Debug control MBBs that are scheduled"),
                      cl::init(0), cl::Hidden);
static cl::opt<int>
DebugMod("postra-sched-debugmod",
                      cl::desc("Debug control MBBs that are scheduled"),
                      cl::init(0), cl::Hidden);

namespace {
  class VISIBILITY_HIDDEN PostRAScheduler : public MachineFunctionPass {
  public:
    static char ID;
    PostRAScheduler() : MachineFunctionPass(&ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineLoopInfo>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    const char *getPassName() const {
      return "Post RA top-down list latency scheduler";
    }

    bool runOnMachineFunction(MachineFunction &Fn);
  };
  char PostRAScheduler::ID = 0;

  class VISIBILITY_HIDDEN SchedulePostRATDList : public ScheduleDAGInstrs {
    /// AvailableQueue - The priority queue to use for the available SUnits.
    ///
    LatencyPriorityQueue AvailableQueue;
  
    /// PendingQueue - This contains all of the instructions whose operands have
    /// been issued, but their results are not ready yet (due to the latency of
    /// the operation).  Once the operands becomes available, the instruction is
    /// added to the AvailableQueue.
    std::vector<SUnit*> PendingQueue;

    /// Topo - A topological ordering for SUnits.
    ScheduleDAGTopologicalSort Topo;

    /// AllocatableSet - The set of allocatable registers.
    /// We'll be ignoring anti-dependencies on non-allocatable registers,
    /// because they may not be safe to break.
    const BitVector AllocatableSet;

    /// HazardRec - The hazard recognizer to use.
    ScheduleHazardRecognizer *HazardRec;

    /// Classes - For live regs that are only used in one register class in a
    /// live range, the register class. If the register is not live, the
    /// corresponding value is null. If the register is live but used in
    /// multiple register classes, the corresponding value is -1 casted to a
    /// pointer.
    const TargetRegisterClass *
      Classes[TargetRegisterInfo::FirstVirtualRegister];

    /// RegRegs - Map registers to all their references within a live range.
    std::multimap<unsigned, MachineOperand *> RegRefs;

    /// KillIndices - The index of the most recent kill (proceding bottom-up),
    /// or ~0u if the register is not live.
    unsigned KillIndices[TargetRegisterInfo::FirstVirtualRegister];

    /// DefIndices - The index of the most recent complete def (proceding bottom
    /// up), or ~0u if the register is live.
    unsigned DefIndices[TargetRegisterInfo::FirstVirtualRegister];

    /// KeepRegs - A set of registers which are live and cannot be changed to
    /// break anti-dependencies.
    SmallSet<unsigned, 4> KeepRegs;

  public:
    SchedulePostRATDList(MachineFunction &MF,
                         const MachineLoopInfo &MLI,
                         const MachineDominatorTree &MDT,
                         ScheduleHazardRecognizer *HR)
      : ScheduleDAGInstrs(MF, MLI, MDT), Topo(SUnits),
        AllocatableSet(TRI->getAllocatableSet(MF)),
        HazardRec(HR) {}

    ~SchedulePostRATDList() {
      delete HazardRec;
    }

    /// StartBlock - Initialize register live-range state for scheduling in
    /// this block.
    ///
    void StartBlock(MachineBasicBlock *BB);

    /// Schedule - Schedule the instruction range using list scheduling.
    ///
    void Schedule();
    
    /// FixupKills - Fix register kill flags that have been made
    /// invalid due to scheduling
    ///
    void FixupKills(MachineBasicBlock *MBB);

    /// Observe - Update liveness information to account for the current
    /// instruction, which will not be scheduled.
    ///
    void Observe(MachineInstr *MI, unsigned Count);

    /// FinishBlock - Clean up register live-range state.
    ///
    void FinishBlock();

  private:
    void PrescanInstruction(MachineInstr *MI);
    void ScanInstruction(MachineInstr *MI, unsigned Count);
    void ReleaseSucc(SUnit *SU, SDep *SuccEdge);
    void ReleaseSuccessors(SUnit *SU);
    void ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle);
    void ListScheduleTopDown();
    bool BreakAntiDependencies();
    unsigned findSuitableFreeRegister(unsigned AntiDepReg,
                                      unsigned LastNewReg,
                                      const TargetRegisterClass *);
    void StartBlockForKills(MachineBasicBlock *BB);
    
    // ToggleKillFlag - Toggle a register operand kill flag. Other
    // adjustments may be made to the instruction if necessary. Return
    // true if the operand has been deleted, false if not.
    bool ToggleKillFlag(MachineInstr *MI, MachineOperand &MO);
  };
}

/// isSchedulingBoundary - Test if the given instruction should be
/// considered a scheduling boundary. This primarily includes labels
/// and terminators.
///
static bool isSchedulingBoundary(const MachineInstr *MI,
                                 const MachineFunction &MF) {
  // Terminators and labels can't be scheduled around.
  if (MI->getDesc().isTerminator() || MI->isLabel())
    return true;

  // Don't attempt to schedule around any instruction that modifies
  // a stack-oriented pointer, as it's unlikely to be profitable. This
  // saves compile time, because it doesn't require every single
  // stack slot reference to depend on the instruction that does the
  // modification.
  const TargetLowering &TLI = *MF.getTarget().getTargetLowering();
  if (MI->modifiesRegister(TLI.getStackPointerRegisterToSaveRestore()))
    return true;

  return false;
}

bool PostRAScheduler::runOnMachineFunction(MachineFunction &Fn) {
  // Check for explicit enable/disable of post-ra scheduling.
  if (EnablePostRAScheduler.getPosition() > 0) {
    if (!EnablePostRAScheduler)
      return true;
  } else {
    // Check that post-RA scheduling is enabled for this function
    const TargetSubtarget &ST = Fn.getTarget().getSubtarget<TargetSubtarget>();
    if (!ST.enablePostRAScheduler())
      return true;
  }

  DEBUG(errs() << "PostRAScheduler\n");

  const MachineLoopInfo &MLI = getAnalysis<MachineLoopInfo>();
  const MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();
  const InstrItineraryData &InstrItins = Fn.getTarget().getInstrItineraryData();
  ScheduleHazardRecognizer *HR = EnablePostRAHazardAvoidance ?
    (ScheduleHazardRecognizer *)new ExactHazardRecognizer(InstrItins) :
    (ScheduleHazardRecognizer *)new SimpleHazardRecognizer();

  SchedulePostRATDList Scheduler(Fn, MLI, MDT, HR);

  // Loop over all of the basic blocks
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
#ifndef NDEBUG
    // If DebugDiv > 0 then only schedule MBB with (ID % DebugDiv) == DebugMod
    if (DebugDiv > 0) {
      static int bbcnt = 0;
      if (bbcnt++ % DebugDiv != DebugMod)
        continue;
      errs() << "*** DEBUG scheduling " << Fn.getFunction()->getNameStr() <<
        ":MBB ID#" << MBB->getNumber() << " ***\n";
    }
#endif

    // Initialize register live-range state for scheduling in this block.
    Scheduler.StartBlock(MBB);

    // Schedule each sequence of instructions not interrupted by a label
    // or anything else that effectively needs to shut down scheduling.
    MachineBasicBlock::iterator Current = MBB->end();
    unsigned Count = MBB->size(), CurrentCount = Count;
    for (MachineBasicBlock::iterator I = Current; I != MBB->begin(); ) {
      MachineInstr *MI = prior(I);
      if (isSchedulingBoundary(MI, Fn)) {
        Scheduler.Run(MBB, I, Current, CurrentCount);
        Scheduler.EmitSchedule(0);
        Current = MI;
        CurrentCount = Count - 1;
        Scheduler.Observe(MI, CurrentCount);
      }
      I = MI;
      --Count;
    }
    assert(Count == 0 && "Instruction count mismatch!");
    assert((MBB->begin() == Current || CurrentCount != 0) &&
           "Instruction count mismatch!");
    Scheduler.Run(MBB, MBB->begin(), Current, CurrentCount);
    Scheduler.EmitSchedule(0);

    // Clean up register live-range state.
    Scheduler.FinishBlock();

    // Update register kills
    Scheduler.FixupKills(MBB);
  }

  return true;
}
  
/// StartBlock - Initialize register live-range state for scheduling in
/// this block.
///
void SchedulePostRATDList::StartBlock(MachineBasicBlock *BB) {
  // Call the superclass.
  ScheduleDAGInstrs::StartBlock(BB);

  // Reset the hazard recognizer.
  HazardRec->Reset();

  // Clear out the register class data.
  std::fill(Classes, array_endof(Classes),
            static_cast<const TargetRegisterClass *>(0));

  // Initialize the indices to indicate that no registers are live.
  std::fill(KillIndices, array_endof(KillIndices), ~0u);
  std::fill(DefIndices, array_endof(DefIndices), BB->size());

  // Clear "do not change" set.
  KeepRegs.clear();

  bool IsReturnBlock = (!BB->empty() && BB->back().getDesc().isReturn());

  // Determine the live-out physregs for this block.
  if (IsReturnBlock) {
    // In a return block, examine the function live-out regs.
    for (MachineRegisterInfo::liveout_iterator I = MRI.liveout_begin(),
         E = MRI.liveout_end(); I != E; ++I) {
      unsigned Reg = *I;
      Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
      KillIndices[Reg] = BB->size();
      DefIndices[Reg] = ~0u;
      // Repeat, for all aliases.
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        unsigned AliasReg = *Alias;
        Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
        KillIndices[AliasReg] = BB->size();
        DefIndices[AliasReg] = ~0u;
      }
    }
  } else {
    // In a non-return block, examine the live-in regs of all successors.
    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
         SE = BB->succ_end(); SI != SE; ++SI)
      for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
           E = (*SI)->livein_end(); I != E; ++I) {
        unsigned Reg = *I;
        Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
        KillIndices[Reg] = BB->size();
        DefIndices[Reg] = ~0u;
        // Repeat, for all aliases.
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          unsigned AliasReg = *Alias;
          Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
          KillIndices[AliasReg] = BB->size();
          DefIndices[AliasReg] = ~0u;
        }
      }
  }

  // Mark live-out callee-saved registers. In a return block this is
  // all callee-saved registers. In non-return this is any
  // callee-saved register that is not saved in the prolog.
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  BitVector Pristine = MFI->getPristineRegs(BB);
  for (const unsigned *I = TRI->getCalleeSavedRegs(); *I; ++I) {
    unsigned Reg = *I;
    if (!IsReturnBlock && !Pristine.test(Reg)) continue;
    Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
    KillIndices[Reg] = BB->size();
    DefIndices[Reg] = ~0u;
    // Repeat, for all aliases.
    for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
      unsigned AliasReg = *Alias;
      Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
      KillIndices[AliasReg] = BB->size();
      DefIndices[AliasReg] = ~0u;
    }
  }
}

/// Schedule - Schedule the instruction range using list scheduling.
///
void SchedulePostRATDList::Schedule() {
  DEBUG(errs() << "********** List Scheduling **********\n");
  
  // Build the scheduling graph.
  BuildSchedGraph();

  if (EnableAntiDepBreaking) {
    if (BreakAntiDependencies()) {
      // We made changes. Update the dependency graph.
      // Theoretically we could update the graph in place:
      // When a live range is changed to use a different register, remove
      // the def's anti-dependence *and* output-dependence edges due to
      // that register, and add new anti-dependence and output-dependence
      // edges based on the next live range of the register.
      SUnits.clear();
      EntrySU = SUnit();
      ExitSU = SUnit();
      BuildSchedGraph();
    }
  }

  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(this));

  AvailableQueue.initNodes(SUnits);

  ListScheduleTopDown();
  
  AvailableQueue.releaseState();
}

/// Observe - Update liveness information to account for the current
/// instruction, which will not be scheduled.
///
void SchedulePostRATDList::Observe(MachineInstr *MI, unsigned Count) {
  assert(Count < InsertPosIndex && "Instruction index out of expected range!");

  // Any register which was defined within the previous scheduling region
  // may have been rescheduled and its lifetime may overlap with registers
  // in ways not reflected in our current liveness state. For each such
  // register, adjust the liveness state to be conservatively correct.
  for (unsigned Reg = 0; Reg != TargetRegisterInfo::FirstVirtualRegister; ++Reg)
    if (DefIndices[Reg] < InsertPosIndex && DefIndices[Reg] >= Count) {
      assert(KillIndices[Reg] == ~0u && "Clobbered register is live!");
      // Mark this register to be non-renamable.
      Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
      // Move the def index to the end of the previous region, to reflect
      // that the def could theoretically have been scheduled at the end.
      DefIndices[Reg] = InsertPosIndex;
    }

  PrescanInstruction(MI);
  ScanInstruction(MI, Count);
}

/// FinishBlock - Clean up register live-range state.
///
void SchedulePostRATDList::FinishBlock() {
  RegRefs.clear();

  // Call the superclass.
  ScheduleDAGInstrs::FinishBlock();
}

/// CriticalPathStep - Return the next SUnit after SU on the bottom-up
/// critical path.
static SDep *CriticalPathStep(SUnit *SU) {
  SDep *Next = 0;
  unsigned NextDepth = 0;
  // Find the predecessor edge with the greatest depth.
  for (SUnit::pred_iterator P = SU->Preds.begin(), PE = SU->Preds.end();
       P != PE; ++P) {
    SUnit *PredSU = P->getSUnit();
    unsigned PredLatency = P->getLatency();
    unsigned PredTotalLatency = PredSU->getDepth() + PredLatency;
    // In the case of a latency tie, prefer an anti-dependency edge over
    // other types of edges.
    if (NextDepth < PredTotalLatency ||
        (NextDepth == PredTotalLatency && P->getKind() == SDep::Anti)) {
      NextDepth = PredTotalLatency;
      Next = &*P;
    }
  }
  return Next;
}

void SchedulePostRATDList::PrescanInstruction(MachineInstr *MI) {
  // Scan the register operands for this instruction and update
  // Classes and RegRefs.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    const TargetRegisterClass *NewRC = 0;
    
    if (i < MI->getDesc().getNumOperands())
      NewRC = MI->getDesc().OpInfo[i].getRegClass(TRI);

    // For now, only allow the register to be changed if its register
    // class is consistent across all uses.
    if (!Classes[Reg] && NewRC)
      Classes[Reg] = NewRC;
    else if (!NewRC || Classes[Reg] != NewRC)
      Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);

    // Now check for aliases.
    for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
      // If an alias of the reg is used during the live range, give up.
      // Note that this allows us to skip checking if AntiDepReg
      // overlaps with any of the aliases, among other things.
      unsigned AliasReg = *Alias;
      if (Classes[AliasReg]) {
        Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
        Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
      }
    }

    // If we're still willing to consider this register, note the reference.
    if (Classes[Reg] != reinterpret_cast<TargetRegisterClass *>(-1))
      RegRefs.insert(std::make_pair(Reg, &MO));

    // It's not safe to change register allocation for source operands of
    // that have special allocation requirements.
    if (MO.isUse() && MI->getDesc().hasExtraSrcRegAllocReq()) {
      if (KeepRegs.insert(Reg)) {
        for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
             *Subreg; ++Subreg)
          KeepRegs.insert(*Subreg);
      }
    }
  }
}

void SchedulePostRATDList::ScanInstruction(MachineInstr *MI,
                                           unsigned Count) {
  // Update liveness.
  // Proceding upwards, registers that are defed but not used in this
  // instruction are now dead.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    if (!MO.isDef()) continue;
    // Ignore two-addr defs.
    if (MI->isRegTiedToUseOperand(i)) continue;

    DefIndices[Reg] = Count;
    KillIndices[Reg] = ~0u;
    assert(((KillIndices[Reg] == ~0u) !=
            (DefIndices[Reg] == ~0u)) &&
           "Kill and Def maps aren't consistent for Reg!");
    KeepRegs.erase(Reg);
    Classes[Reg] = 0;
    RegRefs.erase(Reg);
    // Repeat, for all subregs.
    for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
         *Subreg; ++Subreg) {
      unsigned SubregReg = *Subreg;
      DefIndices[SubregReg] = Count;
      KillIndices[SubregReg] = ~0u;
      KeepRegs.erase(SubregReg);
      Classes[SubregReg] = 0;
      RegRefs.erase(SubregReg);
    }
    // Conservatively mark super-registers as unusable.
    for (const unsigned *Super = TRI->getSuperRegisters(Reg);
         *Super; ++Super) {
      unsigned SuperReg = *Super;
      Classes[SuperReg] = reinterpret_cast<TargetRegisterClass *>(-1);
    }
  }
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    if (!MO.isUse()) continue;

    const TargetRegisterClass *NewRC = 0;
    if (i < MI->getDesc().getNumOperands())
      NewRC = MI->getDesc().OpInfo[i].getRegClass(TRI);

    // For now, only allow the register to be changed if its register
    // class is consistent across all uses.
    if (!Classes[Reg] && NewRC)
      Classes[Reg] = NewRC;
    else if (!NewRC || Classes[Reg] != NewRC)
      Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);

    RegRefs.insert(std::make_pair(Reg, &MO));

    // It wasn't previously live but now it is, this is a kill.
    if (KillIndices[Reg] == ~0u) {
      KillIndices[Reg] = Count;
      DefIndices[Reg] = ~0u;
          assert(((KillIndices[Reg] == ~0u) !=
                  (DefIndices[Reg] == ~0u)) &&
               "Kill and Def maps aren't consistent for Reg!");
    }
    // Repeat, for all aliases.
    for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
      unsigned AliasReg = *Alias;
      if (KillIndices[AliasReg] == ~0u) {
        KillIndices[AliasReg] = Count;
        DefIndices[AliasReg] = ~0u;
      }
    }
  }
}

unsigned
SchedulePostRATDList::findSuitableFreeRegister(unsigned AntiDepReg,
                                               unsigned LastNewReg,
                                               const TargetRegisterClass *RC) {
  for (TargetRegisterClass::iterator R = RC->allocation_order_begin(MF),
       RE = RC->allocation_order_end(MF); R != RE; ++R) {
    unsigned NewReg = *R;
    // Don't replace a register with itself.
    if (NewReg == AntiDepReg) continue;
    // Don't replace a register with one that was recently used to repair
    // an anti-dependence with this AntiDepReg, because that would
    // re-introduce that anti-dependence.
    if (NewReg == LastNewReg) continue;
    // If NewReg is dead and NewReg's most recent def is not before
    // AntiDepReg's kill, it's safe to replace AntiDepReg with NewReg.
    assert(((KillIndices[AntiDepReg] == ~0u) != (DefIndices[AntiDepReg] == ~0u)) &&
           "Kill and Def maps aren't consistent for AntiDepReg!");
    assert(((KillIndices[NewReg] == ~0u) != (DefIndices[NewReg] == ~0u)) &&
           "Kill and Def maps aren't consistent for NewReg!");
    if (KillIndices[NewReg] != ~0u ||
        Classes[NewReg] == reinterpret_cast<TargetRegisterClass *>(-1) ||
        KillIndices[AntiDepReg] > DefIndices[NewReg])
      continue;
    return NewReg;
  }

  // No registers are free and available!
  return 0;
}

/// BreakAntiDependencies - Identifiy anti-dependencies along the critical path
/// of the ScheduleDAG and break them by renaming registers.
///
bool SchedulePostRATDList::BreakAntiDependencies() {
  // The code below assumes that there is at least one instruction,
  // so just duck out immediately if the block is empty.
  if (SUnits.empty()) return false;

  // Find the node at the bottom of the critical path.
  SUnit *Max = 0;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[i];
    if (!Max || SU->getDepth() + SU->Latency > Max->getDepth() + Max->Latency)
      Max = SU;
  }

  DEBUG(errs() << "Critical path has total latency "
        << (Max->getDepth() + Max->Latency) << "\n");

  // Track progress along the critical path through the SUnit graph as we walk
  // the instructions.
  SUnit *CriticalPathSU = Max;
  MachineInstr *CriticalPathMI = CriticalPathSU->getInstr();

  // Consider this pattern:
  //   A = ...
  //   ... = A
  //   A = ...
  //   ... = A
  //   A = ...
  //   ... = A
  //   A = ...
  //   ... = A
  // There are three anti-dependencies here, and without special care,
  // we'd break all of them using the same register:
  //   A = ...
  //   ... = A
  //   B = ...
  //   ... = B
  //   B = ...
  //   ... = B
  //   B = ...
  //   ... = B
  // because at each anti-dependence, B is the first register that
  // isn't A which is free.  This re-introduces anti-dependencies
  // at all but one of the original anti-dependencies that we were
  // trying to break.  To avoid this, keep track of the most recent
  // register that each register was replaced with, avoid
  // using it to repair an anti-dependence on the same register.
  // This lets us produce this:
  //   A = ...
  //   ... = A
  //   B = ...
  //   ... = B
  //   C = ...
  //   ... = C
  //   B = ...
  //   ... = B
  // This still has an anti-dependence on B, but at least it isn't on the
  // original critical path.
  //
  // TODO: If we tracked more than one register here, we could potentially
  // fix that remaining critical edge too. This is a little more involved,
  // because unlike the most recent register, less recent registers should
  // still be considered, though only if no other registers are available.
  unsigned LastNewReg[TargetRegisterInfo::FirstVirtualRegister] = {};

  // Attempt to break anti-dependence edges on the critical path. Walk the
  // instructions from the bottom up, tracking information about liveness
  // as we go to help determine which registers are available.
  bool Changed = false;
  unsigned Count = InsertPosIndex - 1;
  for (MachineBasicBlock::iterator I = InsertPos, E = Begin;
       I != E; --Count) {
    MachineInstr *MI = --I;

    // Check if this instruction has a dependence on the critical path that
    // is an anti-dependence that we may be able to break. If it is, set
    // AntiDepReg to the non-zero register associated with the anti-dependence.
    //
    // We limit our attention to the critical path as a heuristic to avoid
    // breaking anti-dependence edges that aren't going to significantly
    // impact the overall schedule. There are a limited number of registers
    // and we want to save them for the important edges.
    // 
    // TODO: Instructions with multiple defs could have multiple
    // anti-dependencies. The current code here only knows how to break one
    // edge per instruction. Note that we'd have to be able to break all of
    // the anti-dependencies in an instruction in order to be effective.
    unsigned AntiDepReg = 0;
    if (MI == CriticalPathMI) {
      if (SDep *Edge = CriticalPathStep(CriticalPathSU)) {
        SUnit *NextSU = Edge->getSUnit();

        // Only consider anti-dependence edges.
        if (Edge->getKind() == SDep::Anti) {
          AntiDepReg = Edge->getReg();
          assert(AntiDepReg != 0 && "Anti-dependence on reg0?");
          if (!AllocatableSet.test(AntiDepReg))
            // Don't break anti-dependencies on non-allocatable registers.
            AntiDepReg = 0;
          else if (KeepRegs.count(AntiDepReg))
            // Don't break anti-dependencies if an use down below requires
            // this exact register.
            AntiDepReg = 0;
          else {
            // If the SUnit has other dependencies on the SUnit that it
            // anti-depends on, don't bother breaking the anti-dependency
            // since those edges would prevent such units from being
            // scheduled past each other regardless.
            //
            // Also, if there are dependencies on other SUnits with the
            // same register as the anti-dependency, don't attempt to
            // break it.
            for (SUnit::pred_iterator P = CriticalPathSU->Preds.begin(),
                 PE = CriticalPathSU->Preds.end(); P != PE; ++P)
              if (P->getSUnit() == NextSU ?
                    (P->getKind() != SDep::Anti || P->getReg() != AntiDepReg) :
                    (P->getKind() == SDep::Data && P->getReg() == AntiDepReg)) {
                AntiDepReg = 0;
                break;
              }
          }
        }
        CriticalPathSU = NextSU;
        CriticalPathMI = CriticalPathSU->getInstr();
      } else {
        // We've reached the end of the critical path.
        CriticalPathSU = 0;
        CriticalPathMI = 0;
      }
    }

    PrescanInstruction(MI);

    if (MI->getDesc().hasExtraDefRegAllocReq())
      // If this instruction's defs have special allocation requirement, don't
      // break this anti-dependency.
      AntiDepReg = 0;
    else if (AntiDepReg) {
      // If this instruction has a use of AntiDepReg, breaking it
      // is invalid.
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg()) continue;
        unsigned Reg = MO.getReg();
        if (Reg == 0) continue;
        if (MO.isUse() && AntiDepReg == Reg) {
          AntiDepReg = 0;
          break;
        }
      }
    }

    // Determine AntiDepReg's register class, if it is live and is
    // consistently used within a single class.
    const TargetRegisterClass *RC = AntiDepReg != 0 ? Classes[AntiDepReg] : 0;
    assert((AntiDepReg == 0 || RC != NULL) &&
           "Register should be live if it's causing an anti-dependence!");
    if (RC == reinterpret_cast<TargetRegisterClass *>(-1))
      AntiDepReg = 0;

    // Look for a suitable register to use to break the anti-depenence.
    //
    // TODO: Instead of picking the first free register, consider which might
    // be the best.
    if (AntiDepReg != 0) {
      if (unsigned NewReg = findSuitableFreeRegister(AntiDepReg,
                                                     LastNewReg[AntiDepReg],
                                                     RC)) {
        DEBUG(errs() << "Breaking anti-dependence edge on "
              << TRI->getName(AntiDepReg)
              << " with " << RegRefs.count(AntiDepReg) << " references"
              << " using " << TRI->getName(NewReg) << "!\n");

        // Update the references to the old register to refer to the new
        // register.
        std::pair<std::multimap<unsigned, MachineOperand *>::iterator,
                  std::multimap<unsigned, MachineOperand *>::iterator>
           Range = RegRefs.equal_range(AntiDepReg);
        for (std::multimap<unsigned, MachineOperand *>::iterator
             Q = Range.first, QE = Range.second; Q != QE; ++Q)
          Q->second->setReg(NewReg);

        // We just went back in time and modified history; the
        // liveness information for the anti-depenence reg is now
        // inconsistent. Set the state as if it were dead.
        Classes[NewReg] = Classes[AntiDepReg];
        DefIndices[NewReg] = DefIndices[AntiDepReg];
        KillIndices[NewReg] = KillIndices[AntiDepReg];
        assert(((KillIndices[NewReg] == ~0u) !=
                (DefIndices[NewReg] == ~0u)) &&
             "Kill and Def maps aren't consistent for NewReg!");

        Classes[AntiDepReg] = 0;
        DefIndices[AntiDepReg] = KillIndices[AntiDepReg];
        KillIndices[AntiDepReg] = ~0u;
        assert(((KillIndices[AntiDepReg] == ~0u) !=
                (DefIndices[AntiDepReg] == ~0u)) &&
             "Kill and Def maps aren't consistent for AntiDepReg!");

        RegRefs.erase(AntiDepReg);
        Changed = true;
        LastNewReg[AntiDepReg] = NewReg;
      }
    }

    ScanInstruction(MI, Count);
  }

  return Changed;
}

/// StartBlockForKills - Initialize register live-range state for updating kills
///
void SchedulePostRATDList::StartBlockForKills(MachineBasicBlock *BB) {
  // Initialize the indices to indicate that no registers are live.
  std::fill(KillIndices, array_endof(KillIndices), ~0u);

  // Determine the live-out physregs for this block.
  if (!BB->empty() && BB->back().getDesc().isReturn()) {
    // In a return block, examine the function live-out regs.
    for (MachineRegisterInfo::liveout_iterator I = MRI.liveout_begin(),
           E = MRI.liveout_end(); I != E; ++I) {
      unsigned Reg = *I;
      KillIndices[Reg] = BB->size();
      // Repeat, for all subregs.
      for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
           *Subreg; ++Subreg) {
        KillIndices[*Subreg] = BB->size();
      }
    }
  }
  else {
    // In a non-return block, examine the live-in regs of all successors.
    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
           SE = BB->succ_end(); SI != SE; ++SI) {
      for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
             E = (*SI)->livein_end(); I != E; ++I) {
        unsigned Reg = *I;
        KillIndices[Reg] = BB->size();
        // Repeat, for all subregs.
        for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
             *Subreg; ++Subreg) {
          KillIndices[*Subreg] = BB->size();
        }
      }
    }
  }
}

bool SchedulePostRATDList::ToggleKillFlag(MachineInstr *MI,
                                          MachineOperand &MO) {
  // Setting kill flag...
  if (!MO.isKill()) {
    MO.setIsKill(true);
    return false;
  }
  
  // If MO itself is live, clear the kill flag...
  if (KillIndices[MO.getReg()] != ~0u) {
    MO.setIsKill(false);
    return false;
  }

  // If any subreg of MO is live, then create an imp-def for that
  // subreg and keep MO marked as killed.
  bool AllDead = true;
  const unsigned SuperReg = MO.getReg();
  for (const unsigned *Subreg = TRI->getSubRegisters(SuperReg);
       *Subreg; ++Subreg) {
    if (KillIndices[*Subreg] != ~0u) {
      MI->addOperand(MachineOperand::CreateReg(*Subreg,
                                               true  /*IsDef*/,
                                               true  /*IsImp*/,
                                               false /*IsKill*/,
                                               false /*IsDead*/));
      AllDead = false;
    }
  }

  MO.setIsKill(AllDead);
  return false;
}

/// FixupKills - Fix the register kill flags, they may have been made
/// incorrect by instruction reordering.
///
void SchedulePostRATDList::FixupKills(MachineBasicBlock *MBB) {
  DEBUG(errs() << "Fixup kills for BB ID#" << MBB->getNumber() << '\n');

  std::set<unsigned> killedRegs;
  BitVector ReservedRegs = TRI->getReservedRegs(MF);

  StartBlockForKills(MBB);
  
  // Examine block from end to start...
  unsigned Count = MBB->size();
  for (MachineBasicBlock::iterator I = MBB->end(), E = MBB->begin();
       I != E; --Count) {
    MachineInstr *MI = --I;

    // Update liveness.  Registers that are defed but not used in this
    // instruction are now dead. Mark register and all subregs as they
    // are completely defined.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;
      if (!MO.isDef()) continue;
      // Ignore two-addr defs.
      if (MI->isRegTiedToUseOperand(i)) continue;
      
      KillIndices[Reg] = ~0u;
      
      // Repeat for all subregs.
      for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
           *Subreg; ++Subreg) {
        KillIndices[*Subreg] = ~0u;
      }
    }

    // Examine all used registers and set/clear kill flag. When a
    // register is used multiple times we only set the kill flag on
    // the first use.
    killedRegs.clear();
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isUse()) continue;
      unsigned Reg = MO.getReg();
      if ((Reg == 0) || ReservedRegs.test(Reg)) continue;

      bool kill = false;
      if (killedRegs.find(Reg) == killedRegs.end()) {
        kill = true;
        // A register is not killed if any subregs are live...
        for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
             *Subreg; ++Subreg) {
          if (KillIndices[*Subreg] != ~0u) {
            kill = false;
            break;
          }
        }

        // If subreg is not live, then register is killed if it became
        // live in this instruction
        if (kill)
          kill = (KillIndices[Reg] == ~0u);
      }
      
      if (MO.isKill() != kill) {
        bool removed = ToggleKillFlag(MI, MO);
        if (removed) {
          DEBUG(errs() << "Fixed <removed> in ");
        } else {
          DEBUG(errs() << "Fixed " << MO << " in ");
        }
        DEBUG(MI->dump());
      }
      
      killedRegs.insert(Reg);
    }
    
    // Mark any used register (that is not using undef) and subregs as
    // now live...
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isUse() || MO.isUndef()) continue;
      unsigned Reg = MO.getReg();
      if ((Reg == 0) || ReservedRegs.test(Reg)) continue;

      KillIndices[Reg] = Count;
      
      for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
           *Subreg; ++Subreg) {
        KillIndices[*Subreg] = Count;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the PendingQueue if the count reaches zero. Also update its cycle bound.
void SchedulePostRATDList::ReleaseSucc(SUnit *SU, SDep *SuccEdge) {
  SUnit *SuccSU = SuccEdge->getSUnit();

#ifndef NDEBUG
  if (SuccSU->NumPredsLeft == 0) {
    errs() << "*** Scheduling failed! ***\n";
    SuccSU->dump(this);
    errs() << " has been released too many times!\n";
    llvm_unreachable(0);
  }
#endif
  --SuccSU->NumPredsLeft;

  // Compute how many cycles it will be before this actually becomes
  // available.  This is the max of the start time of all predecessors plus
  // their latencies.
  SuccSU->setDepthToAtLeast(SU->getDepth() + SuccEdge->getLatency());
  
  // If all the node's predecessors are scheduled, this node is ready
  // to be scheduled. Ignore the special ExitSU node.
  if (SuccSU->NumPredsLeft == 0 && SuccSU != &ExitSU)
    PendingQueue.push_back(SuccSU);
}

/// ReleaseSuccessors - Call ReleaseSucc on each of SU's successors.
void SchedulePostRATDList::ReleaseSuccessors(SUnit *SU) {
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I)
    ReleaseSucc(SU, &*I);
}

/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void SchedulePostRATDList::ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle) {
  DEBUG(errs() << "*** Scheduling [" << CurCycle << "]: ");
  DEBUG(SU->dump(this));
  
  Sequence.push_back(SU);
  assert(CurCycle >= SU->getDepth() && "Node scheduled above its depth!");
  SU->setDepthToAtLeast(CurCycle);

  ReleaseSuccessors(SU);
  SU->isScheduled = true;
  AvailableQueue.ScheduledNode(SU);
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void SchedulePostRATDList::ListScheduleTopDown() {
  unsigned CurCycle = 0;

  // Release any successors of the special Entry node.
  ReleaseSuccessors(&EntrySU);

  // All leaves to Available queue.
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    // It is available if it has no predecessors.
    if (SUnits[i].Preds.empty()) {
      AvailableQueue.push(&SUnits[i]);
      SUnits[i].isAvailable = true;
    }
  }

  // In any cycle where we can't schedule any instructions, we must
  // stall or emit a noop, depending on the target.
  bool CycleHasInsts = false;

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  std::vector<SUnit*> NotReady;
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue.empty() || !PendingQueue.empty()) {
    // Check to see if any of the pending instructions are ready to issue.  If
    // so, add them to the available queue.
    unsigned MinDepth = ~0u;
    for (unsigned i = 0, e = PendingQueue.size(); i != e; ++i) {
      if (PendingQueue[i]->getDepth() <= CurCycle) {
        AvailableQueue.push(PendingQueue[i]);
        PendingQueue[i]->isAvailable = true;
        PendingQueue[i] = PendingQueue.back();
        PendingQueue.pop_back();
        --i; --e;
      } else if (PendingQueue[i]->getDepth() < MinDepth)
        MinDepth = PendingQueue[i]->getDepth();
    }

    DEBUG(errs() << "\n*** Examining Available\n";
          LatencyPriorityQueue q = AvailableQueue;
          while (!q.empty()) {
            SUnit *su = q.pop();
            errs() << "Height " << su->getHeight() << ": ";
            su->dump(this);
          });

    SUnit *FoundSUnit = 0;

    bool HasNoopHazards = false;
    while (!AvailableQueue.empty()) {
      SUnit *CurSUnit = AvailableQueue.pop();

      ScheduleHazardRecognizer::HazardType HT =
        HazardRec->getHazardType(CurSUnit);
      if (HT == ScheduleHazardRecognizer::NoHazard) {
        FoundSUnit = CurSUnit;
        break;
      }

      // Remember if this is a noop hazard.
      HasNoopHazards |= HT == ScheduleHazardRecognizer::NoopHazard;

      NotReady.push_back(CurSUnit);
    }

    // Add the nodes that aren't ready back onto the available list.
    if (!NotReady.empty()) {
      AvailableQueue.push_all(NotReady);
      NotReady.clear();
    }

    // If we found a node to schedule, do it now.
    if (FoundSUnit) {
      ScheduleNodeTopDown(FoundSUnit, CurCycle);
      HazardRec->EmitInstruction(FoundSUnit);
      CycleHasInsts = true;

      // If we are using the target-specific hazards, then don't
      // advance the cycle time just because we schedule a node. If
      // the target allows it we can schedule multiple nodes in the
      // same cycle.
      if (!EnablePostRAHazardAvoidance) {
        if (FoundSUnit->Latency)  // Don't increment CurCycle for pseudo-ops!
          ++CurCycle;
      }
    } else {
      if (CycleHasInsts) {
        DEBUG(errs() << "*** Finished cycle " << CurCycle << '\n');
        HazardRec->AdvanceCycle();
      } else if (!HasNoopHazards) {
        // Otherwise, we have a pipeline stall, but no other problem,
        // just advance the current cycle and try again.
        DEBUG(errs() << "*** Stall in cycle " << CurCycle << '\n');
        HazardRec->AdvanceCycle();
        ++NumStalls;
      } else {
        // Otherwise, we have no instructions to issue and we have instructions
        // that will fault if we don't do this right.  This is the case for
        // processors without pipeline interlocks and other cases.
        DEBUG(errs() << "*** Emitting noop in cycle " << CurCycle << '\n');
        HazardRec->EmitNoop();
        Sequence.push_back(0);   // NULL here means noop
        ++NumNoops;
      }

      ++CurCycle;
      CycleHasInsts = false;
    }
  }

#ifndef NDEBUG
  VerifySchedule(/*isBottomUp=*/false);
#endif
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createPostRAScheduler() {
  return new PostRAScheduler();
}
