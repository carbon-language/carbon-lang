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

#include "ScheduleDAGInstrs.h"
#include "LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/OwningPtr.h"

#include <queue>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Machine Instruction Scheduling Pass and Registry
//===----------------------------------------------------------------------===//

namespace {
/// MachineScheduler runs after coalescing and before register allocation.
class MachineScheduler : public MachineFunctionPass {
public:
  MachineFunction *MF;
  const TargetInstrInfo *TII;
  const MachineLoopInfo *MLI;
  const MachineDominatorTree *MDT;

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
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariables)
INITIALIZE_PASS_DEPENDENCY(StrongPHIElimination)
INITIALIZE_PASS_DEPENDENCY(RegisterCoalescer)
INITIALIZE_PASS_END(MachineScheduler, "misched",
                    "Machine Instruction Scheduler", false, false)

MachineScheduler::MachineScheduler()
: MachineFunctionPass(ID), MF(0), MLI(0), MDT(0) {
  initializeMachineSchedulerPass(*PassRegistry::getPassRegistry());
}

void MachineScheduler::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequiredID(MachineDominatorsID);
  AU.addRequired<MachineLoopInfo>();
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addRequired<SlotIndexes>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  AU.addRequired<LiveDebugVariables>();
  AU.addPreserved<LiveDebugVariables>();
  if (StrongPHIElim) {
    AU.addRequiredID(StrongPHIEliminationID);
    AU.addPreservedID(StrongPHIEliminationID);
  }
  AU.addRequiredID(RegisterCoalescerPassID);
  AU.addPreservedID(RegisterCoalescerPassID);
  MachineFunctionPass::getAnalysisUsage(AU);
}

namespace {
/// MachineSchedRegistry provides a selection of available machine instruction
/// schedulers.
class MachineSchedRegistry : public MachinePassRegistryNode {
public:
  typedef ScheduleDAGInstrs *(*ScheduleDAGCtor)(MachineScheduler *);

  // RegisterPassParser requires a (misnamed) FunctionPassCtor type.
  typedef ScheduleDAGCtor FunctionPassCtor;

  static MachinePassRegistry Registry;

  MachineSchedRegistry(const char *N, const char *D, ScheduleDAGCtor C)
    : MachinePassRegistryNode(N, D, (MachinePassCtor)C) {
    Registry.Add(this);
  }
  ~MachineSchedRegistry() { Registry.Remove(this); }

  // Accessors.
  //
  MachineSchedRegistry *getNext() const {
    return (MachineSchedRegistry *)MachinePassRegistryNode::getNext();
  }
  static MachineSchedRegistry *getList() {
    return (MachineSchedRegistry *)Registry.getList();
  }
  static ScheduleDAGCtor getDefault() {
    return (ScheduleDAGCtor)Registry.getDefault();
  }
  static void setDefault(ScheduleDAGCtor C) {
    Registry.setDefault((MachinePassCtor)C);
  }
  static void setListener(MachinePassRegistryListener *L) {
    Registry.setListener(L);
  }
};
} // namespace

MachinePassRegistry MachineSchedRegistry::Registry;

static ScheduleDAGInstrs *createDefaultMachineSched(MachineScheduler *P);

/// MachineSchedOpt allows command line selection of the scheduler.
static cl::opt<MachineSchedRegistry::ScheduleDAGCtor, false,
               RegisterPassParser<MachineSchedRegistry> >
MachineSchedOpt("misched",
                cl::init(&createDefaultMachineSched), cl::Hidden,
                cl::desc("Machine instruction scheduler to use"));

//===----------------------------------------------------------------------===//
// Machine Instruction Scheduling Common Implementation
//===----------------------------------------------------------------------===//

namespace {
/// MachineScheduler is an implementation of ScheduleDAGInstrs that schedules
/// machine instructions while updating LiveIntervals.
class ScheduleTopDownLive : public ScheduleDAGInstrs {
protected:
  MachineScheduler *Pass;
public:
  ScheduleTopDownLive(MachineScheduler *P):
    ScheduleDAGInstrs(*P->MF, *P->MLI, *P->MDT, /*IsPostRA=*/false), Pass(P) {}

  /// ScheduleDAGInstrs callback.
  void Schedule();

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

/// Schedule - This is called back from ScheduleDAGInstrs::Run() when it's
/// time to do some work.
void ScheduleTopDownLive::Schedule() {
  BuildSchedGraph(&Pass->getAnalysis<AliasAnalysis>());

  DEBUG(dbgs() << "********** MI Scheduling **********\n");
  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(this));

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

  InsertPos = Begin;
  while (SUnit *SU = pickNode()) {
    DEBUG(dbgs() << "*** Scheduling Instruction:\n"; SU->dump(this));

    // Move the instruction to its new location in the instruction stream.
    MachineInstr *MI = SU->getInstr();
    if (&*InsertPos == MI)
      ++InsertPos;
    else {
      BB->splice(InsertPos, BB, MI);
      if (Begin == InsertPos)
        Begin = MI;
    }

    // TODO: Update live intervals.

    // Release dependent instructions for scheduling.
    releaseSuccessors(SU);
  }
}

bool MachineScheduler::runOnMachineFunction(MachineFunction &mf) {
  // Initialize the context of the pass.
  MF = &mf;
  MLI = &getAnalysis<MachineLoopInfo>();
  MDT = &getAnalysis<MachineDominatorTree>();
  TII = MF->getTarget().getInstrInfo();

  // Select the scheduler, or set the default.
  MachineSchedRegistry::ScheduleDAGCtor Ctor =
    MachineSchedRegistry::getDefault();
  if (!Ctor) {
    Ctor = MachineSchedOpt;
    MachineSchedRegistry::setDefault(Ctor);
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
      // The next region starts above the previous region. Look backward in the
      // instruction stream until we find the nearest boundary.
      MachineBasicBlock::iterator I = RegionEnd;
      for(;I != MBB->begin(); --I, --RemainingCount) {
        if (TII->isSchedulingBoundary(llvm::prior(I), MBB, *MF))
          break;
      }
      if (I == RegionEnd) {
        // Skip empty scheduling regions.
        RegionEnd = llvm::prior(RegionEnd);
        --RemainingCount;
        continue;
      }
      // Skip regions with one instruction.
      if (I == llvm::prior(RegionEnd)) {
        RegionEnd = llvm::prior(RegionEnd);
        continue;
      }
      DEBUG(dbgs() << "MachineScheduling " << MF->getFunction()->getName()
            << ":BB#" << MBB->getNumber() << "\n  From: " << *I << "    To: "
            << *RegionEnd << " Remaining: " << RemainingCount << "\n");

      // Inform ScheduleDAGInstrs of the region being scheduled. It calls back
      // to our Schedule() method.
      Scheduler->Run(MBB, I, RegionEnd, MBB->size());
      RegionEnd = Scheduler->Begin;
    }
    assert(RemainingCount == 0 && "Instruction count mismatch!");
  }
  return true;
}

void MachineScheduler::print(raw_ostream &O, const Module* m) const {
  // unimplemented
}

//===----------------------------------------------------------------------===//
// Placeholder for extending the machine instruction scheduler.
//===----------------------------------------------------------------------===//

namespace {
class DefaultMachineScheduler : public ScheduleDAGInstrs {
  MachineScheduler *Pass;
public:
  DefaultMachineScheduler(MachineScheduler *P):
    ScheduleDAGInstrs(*P->MF, *P->MLI, *P->MDT, /*IsPostRA=*/false), Pass(P) {}

  /// Schedule - This is called back from ScheduleDAGInstrs::Run() when it's
  /// time to do some work.
  void Schedule();
};
} // namespace

static ScheduleDAGInstrs *createDefaultMachineSched(MachineScheduler *P) {
  return new DefaultMachineScheduler(P);
}
static MachineSchedRegistry
SchedDefaultRegistry("default", "Activate the scheduler pass, "
                     "but don't reorder instructions",
                     createDefaultMachineSched);


/// Schedule - This is called back from ScheduleDAGInstrs::Run() when it's
/// time to do some work.
void DefaultMachineScheduler::Schedule() {
  BuildSchedGraph(&Pass->getAnalysis<AliasAnalysis>());

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
// Nodes with a higher number have lower priority. This way we attempt to
// schedule the latest instructions earliest.
//
// TODO: Relies on the property of the BuildSchedGraph that results in SUnits
// being ordered in sequence bottom-up. This will be formalized, probably be
// constructing SUnits in a prepass.
struct ShuffleSUnitOrder {
  bool operator()(SUnit *A, SUnit *B) const {
    return A->NodeNum > B->NodeNum;
  }
};

/// Reorder instructions as much as possible.
class InstructionShuffler : public ScheduleTopDownLive {
  std::priority_queue<SUnit*, std::vector<SUnit*>, ShuffleSUnitOrder> Queue;
public:
  InstructionShuffler(MachineScheduler *P):
    ScheduleTopDownLive(P) {}

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

static ScheduleDAGInstrs *createInstructionShuffler(MachineScheduler *P) {
  return new InstructionShuffler(P);
}
static MachineSchedRegistry ShufflerRegistry("shuffle",
                                             "Shuffle machine instructions",
                                             createInstructionShuffler);
#endif // !NDEBUG
