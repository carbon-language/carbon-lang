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
#include "AntiDepBreaker.h"
#include "AggressiveAntiDepBreaker.h"
#include "CriticalAntiDepBreaker.h"
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
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtarget.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Statistic.h"
#include <map>
#include <set>
using namespace llvm;

STATISTIC(NumNoops, "Number of noops inserted");
STATISTIC(NumStalls, "Number of pipeline stalls");
STATISTIC(NumFixedAnti, "Number of fixed anti-dependencies");

// Post-RA scheduling is enabled with
// TargetSubtarget.enablePostRAScheduler(). This flag can be used to
// override the target.
static cl::opt<bool>
EnablePostRAScheduler("post-RA-scheduler",
                       cl::desc("Enable scheduling after register allocation"),
                       cl::init(false), cl::Hidden);
static cl::opt<std::string>
EnableAntiDepBreaking("break-anti-dependencies",
                      cl::desc("Break post-RA scheduling anti-dependencies: "
                               "\"critical\", \"all\", or \"none\""),
                      cl::init("none"), cl::Hidden);
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

AntiDepBreaker::~AntiDepBreaker() { }

namespace {
  class PostRAScheduler : public MachineFunctionPass {
    AliasAnalysis *AA;
    CodeGenOpt::Level OptLevel;

  public:
    static char ID;
    PostRAScheduler(CodeGenOpt::Level ol) :
      MachineFunctionPass(&ID), OptLevel(ol) {}

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<AliasAnalysis>();
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

  class SchedulePostRATDList : public ScheduleDAGInstrs {
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

    /// HazardRec - The hazard recognizer to use.
    ScheduleHazardRecognizer *HazardRec;

    /// AntiDepBreak - Anti-dependence breaking object, or NULL if none
    AntiDepBreaker *AntiDepBreak;

    /// AA - AliasAnalysis for making memory reference queries.
    AliasAnalysis *AA;

    /// KillIndices - The index of the most recent kill (proceding bottom-up),
    /// or ~0u if the register is not live.
    unsigned KillIndices[TargetRegisterInfo::FirstVirtualRegister];

  public:
    SchedulePostRATDList(MachineFunction &MF,
                         const MachineLoopInfo &MLI,
                         const MachineDominatorTree &MDT,
                         ScheduleHazardRecognizer *HR,
                         AntiDepBreaker *ADB,
                         AliasAnalysis *aa)
      : ScheduleDAGInstrs(MF, MLI, MDT), Topo(SUnits),
      HazardRec(HR), AntiDepBreak(ADB), AA(aa) {}

    ~SchedulePostRATDList() {
    }

    /// StartBlock - Initialize register live-range state for scheduling in
    /// this block.
    ///
    void StartBlock(MachineBasicBlock *BB);

    /// Schedule - Schedule the instruction range using list scheduling.
    ///
    void Schedule();
    
    /// Observe - Update liveness information to account for the current
    /// instruction, which will not be scheduled.
    ///
    void Observe(MachineInstr *MI, unsigned Count);

    /// FinishBlock - Clean up register live-range state.
    ///
    void FinishBlock();

    /// FixupKills - Fix register kill flags that have been made
    /// invalid due to scheduling
    ///
    void FixupKills(MachineBasicBlock *MBB);

  private:
    void ReleaseSucc(SUnit *SU, SDep *SuccEdge);
    void ReleaseSuccessors(SUnit *SU);
    void ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle);
    void ListScheduleTopDown();
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
  AA = &getAnalysis<AliasAnalysis>();

  // Check for explicit enable/disable of post-ra scheduling.
  TargetSubtarget::AntiDepBreakMode AntiDepMode = TargetSubtarget::ANTIDEP_NONE;
  SmallVector<TargetRegisterClass*, 4> CriticalPathRCs;
  if (EnablePostRAScheduler.getPosition() > 0) {
    if (!EnablePostRAScheduler)
      return false;
  } else {
    // Check that post-RA scheduling is enabled for this target.
    const TargetSubtarget &ST = Fn.getTarget().getSubtarget<TargetSubtarget>();
    if (!ST.enablePostRAScheduler(OptLevel, AntiDepMode, CriticalPathRCs))
      return false;
  }

  // Check for antidep breaking override...
  if (EnableAntiDepBreaking.getPosition() > 0) {
    AntiDepMode = (EnableAntiDepBreaking == "all") ? TargetSubtarget::ANTIDEP_ALL :
      (EnableAntiDepBreaking == "critical") ? TargetSubtarget::ANTIDEP_CRITICAL :
      TargetSubtarget::ANTIDEP_NONE;
  }

  DEBUG(dbgs() << "PostRAScheduler\n");

  const MachineLoopInfo &MLI = getAnalysis<MachineLoopInfo>();
  const MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();
  const InstrItineraryData &InstrItins = Fn.getTarget().getInstrItineraryData();
  ScheduleHazardRecognizer *HR = EnablePostRAHazardAvoidance ?
    (ScheduleHazardRecognizer *)new ExactHazardRecognizer(InstrItins) :
    (ScheduleHazardRecognizer *)new SimpleHazardRecognizer();
  AntiDepBreaker *ADB = 
    ((AntiDepMode == TargetSubtarget::ANTIDEP_ALL) ?
     (AntiDepBreaker *)new AggressiveAntiDepBreaker(Fn, CriticalPathRCs) :
     ((AntiDepMode == TargetSubtarget::ANTIDEP_CRITICAL) ? 
      (AntiDepBreaker *)new CriticalAntiDepBreaker(Fn) : NULL));

  SchedulePostRATDList Scheduler(Fn, MLI, MDT, HR, ADB, AA);

  // Loop over all of the basic blocks
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
#ifndef NDEBUG
    // If DebugDiv > 0 then only schedule MBB with (ID % DebugDiv) == DebugMod
    if (DebugDiv > 0) {
      static int bbcnt = 0;
      if (bbcnt++ % DebugDiv != DebugMod)
        continue;
      dbgs() << "*** DEBUG scheduling " << Fn.getFunction()->getNameStr() <<
        ":BB#" << MBB->getNumber() << " ***\n";
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

  delete HR;
  delete ADB;

  return true;
}
  
/// StartBlock - Initialize register live-range state for scheduling in
/// this block.
///
void SchedulePostRATDList::StartBlock(MachineBasicBlock *BB) {
  // Call the superclass.
  ScheduleDAGInstrs::StartBlock(BB);

  // Reset the hazard recognizer and anti-dep breaker.
  HazardRec->Reset();
  if (AntiDepBreak != NULL)
    AntiDepBreak->StartBlock(BB);
}

/// Schedule - Schedule the instruction range using list scheduling.
///
void SchedulePostRATDList::Schedule() {
  // Build the scheduling graph.
  BuildSchedGraph(AA);

  if (AntiDepBreak != NULL) {
    unsigned Broken = 
      AntiDepBreak->BreakAntiDependencies(SUnits, Begin, InsertPos,
                                          InsertPosIndex);
    
    if (Broken != 0) {
      // We made changes. Update the dependency graph.
      // Theoretically we could update the graph in place:
      // When a live range is changed to use a different register, remove
      // the def's anti-dependence *and* output-dependence edges due to
      // that register, and add new anti-dependence and output-dependence
      // edges based on the next live range of the register.
      SUnits.clear();
      Sequence.clear();
      EntrySU = SUnit();
      ExitSU = SUnit();
      BuildSchedGraph(AA);
      
      NumFixedAnti += Broken;
    }
  }

  DEBUG(dbgs() << "********** List Scheduling **********\n");
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
  if (AntiDepBreak != NULL)
    AntiDepBreak->Observe(MI, Count, InsertPosIndex);
}

/// FinishBlock - Clean up register live-range state.
///
void SchedulePostRATDList::FinishBlock() {
  if (AntiDepBreak != NULL)
    AntiDepBreak->FinishBlock();

  // Call the superclass.
  ScheduleDAGInstrs::FinishBlock();
}

/// StartBlockForKills - Initialize register live-range state for updating kills
///
void SchedulePostRATDList::StartBlockForKills(MachineBasicBlock *BB) {
  // Initialize the indices to indicate that no registers are live.
  for (unsigned i = 0; i < TRI->getNumRegs(); ++i)
    KillIndices[i] = ~0u;

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
  MO.setIsKill(false);
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

  if(AllDead)
    MO.setIsKill(true);
  return false;
}

/// FixupKills - Fix the register kill flags, they may have been made
/// incorrect by instruction reordering.
///
void SchedulePostRATDList::FixupKills(MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Fixup kills for BB#" << MBB->getNumber() << '\n');

  std::set<unsigned> killedRegs;
  BitVector ReservedRegs = TRI->getReservedRegs(MF);

  StartBlockForKills(MBB);
  
  // Examine block from end to start...
  unsigned Count = MBB->size();
  for (MachineBasicBlock::iterator I = MBB->end(), E = MBB->begin();
       I != E; --Count) {
    MachineInstr *MI = --I;
    if (MI->isDebugValue())
      continue;

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
        DEBUG(dbgs() << "Fixing " << MO << " in ");
        // Warning: ToggleKillFlag may invalidate MO.
        ToggleKillFlag(MI, MO);
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
    dbgs() << "*** Scheduling failed! ***\n";
    SuccSU->dump(this);
    dbgs() << " has been released too many times!\n";
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
       I != E; ++I) {
    ReleaseSucc(SU, &*I);
  }
}

/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void SchedulePostRATDList::ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle) {
  DEBUG(dbgs() << "*** Scheduling [" << CurCycle << "]: ");
  DEBUG(SU->dump(this));
  
  Sequence.push_back(SU);
  assert(CurCycle >= SU->getDepth() && 
         "Node scheduled above its depth!");
  SU->setDepthToAtLeast(CurCycle);

  ReleaseSuccessors(SU);
  SU->isScheduled = true;
  AvailableQueue.ScheduledNode(SU);
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void SchedulePostRATDList::ListScheduleTopDown() {
  unsigned CurCycle = 0;
  
  // We're scheduling top-down but we're visiting the regions in
  // bottom-up order, so we don't know the hazards at the start of a
  // region. So assume no hazards (this should usually be ok as most
  // blocks are a single region).
  HazardRec->Reset();

  // Release any successors of the special Entry node.
  ReleaseSuccessors(&EntrySU);

  // Add all leaves to Available queue.
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    // It is available if it has no predecessors.
    bool available = SUnits[i].Preds.empty();
    if (available) {
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

    DEBUG(dbgs() << "\n*** Examining Available\n";
          LatencyPriorityQueue q = AvailableQueue;
          while (!q.empty()) {
            SUnit *su = q.pop();
            dbgs() << "Height " << su->getHeight() << ": ";
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

    // If we found a node to schedule...
    if (FoundSUnit) {
      // ... schedule the node...
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
        DEBUG(dbgs() << "*** Finished cycle " << CurCycle << '\n');
        HazardRec->AdvanceCycle();
      } else if (!HasNoopHazards) {
        // Otherwise, we have a pipeline stall, but no other problem,
        // just advance the current cycle and try again.
        DEBUG(dbgs() << "*** Stall in cycle " << CurCycle << '\n');
        HazardRec->AdvanceCycle();
        ++NumStalls;
      } else {
        // Otherwise, we have no instructions to issue and we have instructions
        // that will fault if we don't do this right.  This is the case for
        // processors without pipeline interlocks and other cases.
        DEBUG(dbgs() << "*** Emitting noop in cycle " << CurCycle << '\n');
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

FunctionPass *llvm::createPostRAScheduler(CodeGenOpt::Level OptLevel) {
  return new PostRAScheduler(OptLevel);
}
