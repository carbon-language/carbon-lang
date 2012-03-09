//===- MachineScheduler.cpp - Machine Instruction Scheduler ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// MachineScheduler schedules machine instructions after phi elimination. It
// preserves LiveIntervals so it can be invoked before register allocation.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "misched"

#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/OwningPtr.h"

#include <queue>

using namespace llvm;

#ifndef NDEBUG
static cl::opt<bool> ViewMISchedDAGs("view-misched-dags", cl::Hidden,
  cl::desc("Pop up a window to show MISched dags after they are processed"));
#else
static bool ViewMISchedDAGs = false;
#endif // NDEBUG

//===----------------------------------------------------------------------===//
// Machine Instruction Scheduling Pass and Registry
//===----------------------------------------------------------------------===//

namespace {
/// MachineScheduler runs after coalescing and before register allocation.
class MachineScheduler : public MachineSchedContext,
                         public MachineFunctionPass {
public:
  MachineScheduler();

  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  virtual void releaseMemory() {}

  virtual bool runOnMachineFunction(MachineFunction&);

  virtual void print(raw_ostream &O, const Module* = 0) const;

  static char ID; // Class identification, replacement for typeinfo
};
} // namespace

char MachineScheduler::ID = 0;

char &llvm::MachineSchedulerID = MachineScheduler::ID;

INITIALIZE_PASS_BEGIN(MachineScheduler, "misched",
                      "Machine Instruction Scheduler", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(MachineScheduler, "misched",
                    "Machine Instruction Scheduler", false, false)

MachineScheduler::MachineScheduler()
: MachineFunctionPass(ID) {
  initializeMachineSchedulerPass(*PassRegistry::getPassRegistry());
}

void MachineScheduler::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequiredID(MachineDominatorsID);
  AU.addRequired<MachineLoopInfo>();
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<TargetPassConfig>();
  AU.addRequired<SlotIndexes>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

MachinePassRegistry MachineSchedRegistry::Registry;

/// A dummy default scheduler factory indicates whether the scheduler
/// is overridden on the command line.
static ScheduleDAGInstrs *useDefaultMachineSched(MachineSchedContext *C) {
  return 0;
}

/// MachineSchedOpt allows command line selection of the scheduler.
static cl::opt<MachineSchedRegistry::ScheduleDAGCtor, false,
               RegisterPassParser<MachineSchedRegistry> >
MachineSchedOpt("misched",
                cl::init(&useDefaultMachineSched), cl::Hidden,
                cl::desc("Machine instruction scheduler to use"));

static MachineSchedRegistry
SchedDefaultRegistry("default", "Use the target's default scheduler choice.",
                     useDefaultMachineSched);

/// Forward declare the common machine scheduler. This will be used as the
/// default scheduler if the target does not set a default.
static ScheduleDAGInstrs *createCommonMachineSched(MachineSchedContext *C);

bool MachineScheduler::runOnMachineFunction(MachineFunction &mf) {
  // Initialize the context of the pass.
  MF = &mf;
  MLI = &getAnalysis<MachineLoopInfo>();
  MDT = &getAnalysis<MachineDominatorTree>();
  PassConfig = &getAnalysis<TargetPassConfig>();
  AA = &getAnalysis<AliasAnalysis>();

  LIS = &getAnalysis<LiveIntervals>();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();

  // Select the scheduler, or set the default.
  MachineSchedRegistry::ScheduleDAGCtor Ctor = MachineSchedOpt;
  if (Ctor == useDefaultMachineSched) {
    // Get the default scheduler set by the target.
    Ctor = MachineSchedRegistry::getDefault();
    if (!Ctor) {
      Ctor = createCommonMachineSched;
      MachineSchedRegistry::setDefault(Ctor);
    }
  }
  // Instantiate the selected scheduler.
  OwningPtr<ScheduleDAGInstrs> Scheduler(Ctor(this));

  // Visit all machine basic blocks.
  for (MachineFunction::iterator MBB = MF->begin(), MBBEnd = MF->end();
       MBB != MBBEnd; ++MBB) {

    // Break the block into scheduling regions [I, RegionEnd), and schedule each
    // region as soon as it is discovered.
    unsigned RemainingCount = MBB->size();
    for(MachineBasicBlock::iterator RegionEnd = MBB->end();
        RegionEnd != MBB->begin();) {
      Scheduler->startBlock(MBB);
      // The next region starts above the previous region. Look backward in the
      // instruction stream until we find the nearest boundary.
      MachineBasicBlock::iterator I = RegionEnd;
      for(;I != MBB->begin(); --I, --RemainingCount) {
        if (TII->isSchedulingBoundary(llvm::prior(I), MBB, *MF))
          break;
      }
      // Notify the scheduler of the region, even if we may skip scheduling
      // it. Perhaps it still needs to be bundled.
      Scheduler->enterRegion(MBB, I, RegionEnd, RemainingCount);

      // Skip empty scheduling regions (0 or 1 schedulable instructions).
      if (I == RegionEnd || I == llvm::prior(RegionEnd)) {
        RegionEnd = llvm::prior(RegionEnd);
        if (I != RegionEnd)
          --RemainingCount;
        // Close the current region. Bundle the terminator if needed.
        Scheduler->exitRegion();
        continue;
      }
      DEBUG(dbgs() << "MachineScheduling " << MF->getFunction()->getName()
            << ":BB#" << MBB->getNumber() << "\n  From: " << *I << "    To: ";
            if (RegionEnd != MBB->end()) dbgs() << *RegionEnd;
            else dbgs() << "End";
            dbgs() << " Remaining: " << RemainingCount << "\n");

      // Schedule a region: possibly reorder instructions.
      Scheduler->schedule();

      // Close the current region.
      Scheduler->exitRegion();

      // Scheduling has invalidated the current iterator 'I'. Ask the
      // scheduler for the top of it's scheduled region.
      RegionEnd = Scheduler->begin();
    }
    assert(RemainingCount == 0 && "Instruction count mismatch!");
    Scheduler->finishBlock();
  }
  return true;
}

void MachineScheduler::print(raw_ostream &O, const Module* m) const {
  // unimplemented
}

//===----------------------------------------------------------------------===//
// ScheduleTopeDownLive - Base class for basic top-down scheduling with
// LiveIntervals preservation.
// ===----------------------------------------------------------------------===//

namespace {
/// ScheduleTopDownLive is an implementation of ScheduleDAGInstrs that schedules
/// machine instructions while updating LiveIntervals.
class ScheduleTopDownLive : public ScheduleDAGInstrs {
  AliasAnalysis *AA;
public:
  ScheduleTopDownLive(MachineSchedContext *C):
    ScheduleDAGInstrs(*C->MF, *C->MLI, *C->MDT, /*IsPostRA=*/false, C->LIS),
    AA(C->AA) {}

  /// ScheduleDAGInstrs interface.
  void schedule();

  /// Interface implemented by the selected top-down liveinterval scheduler.
  ///
  /// Pick the next node to schedule, or return NULL.
  virtual SUnit *pickNode() = 0;

  /// When all preceeding dependencies have been resolved, free this node for
  /// scheduling.
  virtual void releaseNode(SUnit *SU) = 0;

protected:
  void releaseSucc(SUnit *SU, SDep *SuccEdge);
  void releaseSuccessors(SUnit *SU);
};
} // namespace

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. When
/// NumPredsLeft reaches zero, release the successor node.
void ScheduleTopDownLive::releaseSucc(SUnit *SU, SDep *SuccEdge) {
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
  if (SuccSU->NumPredsLeft == 0 && SuccSU != &ExitSU)
    releaseNode(SuccSU);
}

/// releaseSuccessors - Call releaseSucc on each of SU's successors.
void ScheduleTopDownLive::releaseSuccessors(SUnit *SU) {
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    releaseSucc(SU, &*I);
  }
}

/// schedule - This is called back from ScheduleDAGInstrs::Run() when it's
/// time to do some work.
void ScheduleTopDownLive::schedule() {
  buildSchedGraph(AA);

  DEBUG(dbgs() << "********** MI Scheduling **********\n");
  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(this));

  if (ViewMISchedDAGs) viewGraph();

  // Release any successors of the special Entry node. It is currently unused,
  // but we keep up appearances.
  releaseSuccessors(&EntrySU);

  // Release all DAG roots for scheduling.
  for (std::vector<SUnit>::iterator I = SUnits.begin(), E = SUnits.end();
       I != E; ++I) {
    // A SUnit is ready to schedule if it has no predecessors.
    if (I->Preds.empty())
      releaseNode(&(*I));
  }

  MachineBasicBlock::iterator InsertPos = RegionBegin;
  while (SUnit *SU = pickNode()) {
    DEBUG(dbgs() << "*** Scheduling Instruction:\n"; SU->dump(this));

    // Move the instruction to its new location in the instruction stream.
    MachineInstr *MI = SU->getInstr();
    if (&*InsertPos == MI)
      ++InsertPos;
    else {
      BB->splice(InsertPos, BB, MI);
      LIS->handleMove(MI);
      if (RegionBegin == InsertPos)
        RegionBegin = MI;
    }

    // Release dependent instructions for scheduling.
    releaseSuccessors(SU);
  }
}

//===----------------------------------------------------------------------===//
// Placeholder for the default machine instruction scheduler.
//===----------------------------------------------------------------------===//

namespace {
class CommonMachineScheduler : public ScheduleDAGInstrs {
  AliasAnalysis *AA;
public:
  CommonMachineScheduler(MachineSchedContext *C):
    ScheduleDAGInstrs(*C->MF, *C->MLI, *C->MDT, /*IsPostRA=*/false, C->LIS),
    AA(C->AA) {}

  /// schedule - This is called back from ScheduleDAGInstrs::Run() when it's
  /// time to do some work.
  void schedule();
};
} // namespace

/// The common machine scheduler will be used as the default scheduler if the
/// target does not set a default.
static ScheduleDAGInstrs *createCommonMachineSched(MachineSchedContext *C) {
  return new CommonMachineScheduler(C);
}
static MachineSchedRegistry
SchedCommonRegistry("common", "Use the target's default scheduler choice.",
                     createCommonMachineSched);

/// Schedule - This is called back from ScheduleDAGInstrs::Run() when it's
/// time to do some work.
void CommonMachineScheduler::schedule() {
  buildSchedGraph(AA);

  DEBUG(dbgs() << "********** MI Scheduling **********\n");
  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(this));

  // TODO: Put interesting things here.
  //
  // When this is fully implemented, it will become a subclass of
  // ScheduleTopDownLive. So this driver will disappear.
}

//===----------------------------------------------------------------------===//
// Machine Instruction Shuffler for Correctness Testing
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
namespace {
// Nodes with a higher number have higher priority. This way we attempt to
// schedule the latest instructions earliest.
//
// TODO: Relies on the property of the BuildSchedGraph that results in SUnits
// being ordered in sequence top-down.
struct ShuffleSUnitOrder {
  bool operator()(SUnit *A, SUnit *B) const {
    return A->NodeNum < B->NodeNum;
  }
};

/// Reorder instructions as much as possible.
class InstructionShuffler : public ScheduleTopDownLive {
  std::priority_queue<SUnit*, std::vector<SUnit*>, ShuffleSUnitOrder> Queue;
public:
  InstructionShuffler(MachineSchedContext *C):
    ScheduleTopDownLive(C) {}

  /// ScheduleTopDownLive Interface

  virtual SUnit *pickNode() {
    if (Queue.empty()) return NULL;
    SUnit *SU = Queue.top();
    Queue.pop();
    return SU;
  }

  virtual void releaseNode(SUnit *SU) {
    Queue.push(SU);
  }
};
} // namespace

static ScheduleDAGInstrs *createInstructionShuffler(MachineSchedContext *C) {
  return new InstructionShuffler(C);
}
static MachineSchedRegistry ShufflerRegistry("shuffle",
                                             "Shuffle machine instructions",
                                             createInstructionShuffler);
#endif // !NDEBUG
