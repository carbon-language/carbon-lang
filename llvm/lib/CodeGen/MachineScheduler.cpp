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

#include "RegisterClassInfo.h"
#include "RegisterPressure.h"
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
#include "llvm/ADT/PriorityQueue.h"

#include <queue>

using namespace llvm;

static cl::opt<bool> ForceTopDown("misched-topdown", cl::Hidden,
                                  cl::desc("Force top-down list scheduling"));
static cl::opt<bool> ForceBottomUp("misched-bottomup", cl::Hidden,
                                  cl::desc("Force bottom-up list scheduling"));

#ifndef NDEBUG
static cl::opt<bool> ViewMISchedDAGs("view-misched-dags", cl::Hidden,
  cl::desc("Pop up a window to show MISched dags after they are processed"));

static cl::opt<unsigned> MISchedCutoff("misched-cutoff", cl::Hidden,
  cl::desc("Stop scheduling after N instructions"), cl::init(~0U));
#else
static bool ViewMISchedDAGs = false;
#endif // NDEBUG

//===----------------------------------------------------------------------===//
// Machine Instruction Scheduling Pass and Registry
//===----------------------------------------------------------------------===//

MachineSchedContext::MachineSchedContext():
    MF(0), MLI(0), MDT(0), PassConfig(0), AA(0), LIS(0) {
  RegClassInfo = new RegisterClassInfo();
}

MachineSchedContext::~MachineSchedContext() {
  delete RegClassInfo;
}

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
DefaultSchedRegistry("default", "Use the target's default scheduler choice.",
                     useDefaultMachineSched);

/// Forward declare the standard machine scheduler. This will be used as the
/// default scheduler if the target does not set a default.
static ScheduleDAGInstrs *createConvergingSched(MachineSchedContext *C);


/// Decrement this iterator until reaching the top or a non-debug instr.
static MachineBasicBlock::iterator
priorNonDebug(MachineBasicBlock::iterator I, MachineBasicBlock::iterator Beg) {
  assert(I != Beg && "reached the top of the region, cannot decrement");
  while (--I != Beg) {
    if (!I->isDebugValue())
      break;
  }
  return I;
}

/// If this iterator is a debug value, increment until reaching the End or a
/// non-debug instruction.
static MachineBasicBlock::iterator
nextIfDebug(MachineBasicBlock::iterator I, MachineBasicBlock::iterator End) {
  while(I != End) {
    if (!I->isDebugValue())
      break;
  }
  return I;
}

/// Top-level MachineScheduler pass driver.
///
/// Visit blocks in function order. Divide each block into scheduling regions
/// and visit them bottom-up. Visiting regions bottom-up is not required, but is
/// consistent with the DAG builder, which traverses the interior of the
/// scheduling regions bottom-up.
///
/// This design avoids exposing scheduling boundaries to the DAG builder,
/// simplifying the DAG builder's support for "special" target instructions.
/// At the same time the design allows target schedulers to operate across
/// scheduling boundaries, for example to bundle the boudary instructions
/// without reordering them. This creates complexity, because the target
/// scheduler must update the RegionBegin and RegionEnd positions cached by
/// ScheduleDAGInstrs whenever adding or removing instructions. A much simpler
/// design would be to split blocks at scheduling boundaries, but LLVM has a
/// general bias against block splitting purely for implementation simplicity.
bool MachineScheduler::runOnMachineFunction(MachineFunction &mf) {
  // Initialize the context of the pass.
  MF = &mf;
  MLI = &getAnalysis<MachineLoopInfo>();
  MDT = &getAnalysis<MachineDominatorTree>();
  PassConfig = &getAnalysis<TargetPassConfig>();
  AA = &getAnalysis<AliasAnalysis>();

  LIS = &getAnalysis<LiveIntervals>();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();

  RegClassInfo->runOnMachineFunction(*MF);

  // Select the scheduler, or set the default.
  MachineSchedRegistry::ScheduleDAGCtor Ctor = MachineSchedOpt;
  if (Ctor == useDefaultMachineSched) {
    // Get the default scheduler set by the target.
    Ctor = MachineSchedRegistry::getDefault();
    if (!Ctor) {
      Ctor = createConvergingSched;
      MachineSchedRegistry::setDefault(Ctor);
    }
  }
  // Instantiate the selected scheduler.
  OwningPtr<ScheduleDAGInstrs> Scheduler(Ctor(this));

  // Visit all machine basic blocks.
  //
  // TODO: Visit blocks in global postorder or postorder within the bottom-up
  // loop tree. Then we can optionally compute global RegPressure.
  for (MachineFunction::iterator MBB = MF->begin(), MBBEnd = MF->end();
       MBB != MBBEnd; ++MBB) {

    Scheduler->startBlock(MBB);

    // Break the block into scheduling regions [I, RegionEnd), and schedule each
    // region as soon as it is discovered. RegionEnd points the the scheduling
    // boundary at the bottom of the region. The DAG does not include RegionEnd,
    // but the region does (i.e. the next RegionEnd is above the previous
    // RegionBegin). If the current block has no terminator then RegionEnd ==
    // MBB->end() for the bottom region.
    //
    // The Scheduler may insert instructions during either schedule() or
    // exitRegion(), even for empty regions. So the local iterators 'I' and
    // 'RegionEnd' are invalid across these calls.
    unsigned RemainingCount = MBB->size();
    for(MachineBasicBlock::iterator RegionEnd = MBB->end();
        RegionEnd != MBB->begin(); RegionEnd = Scheduler->begin()) {

      // Avoid decrementing RegionEnd for blocks with no terminator.
      if (RegionEnd != MBB->end()
          || TII->isSchedulingBoundary(llvm::prior(RegionEnd), MBB, *MF)) {
        --RegionEnd;
        // Count the boundary instruction.
        --RemainingCount;
      }

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
        // Close the current region. Bundle the terminator if needed.
        // This invalidates 'RegionEnd' and 'I'.
        Scheduler->exitRegion();
        continue;
      }
      DEBUG(dbgs() << "MachineScheduling " << MF->getFunction()->getName()
            << ":BB#" << MBB->getNumber() << "\n  From: " << *I << "    To: ";
            if (RegionEnd != MBB->end()) dbgs() << *RegionEnd;
            else dbgs() << "End";
            dbgs() << " Remaining: " << RemainingCount << "\n");

      // Schedule a region: possibly reorder instructions.
      // This invalidates 'RegionEnd' and 'I'.
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
  Scheduler->finalizeSchedule();
  DEBUG(LIS->print(dbgs()));
  return true;
}

void MachineScheduler::print(raw_ostream &O, const Module* m) const {
  // unimplemented
}

//===----------------------------------------------------------------------===//
// MachineSchedStrategy - Interface to a machine scheduling algorithm.
//===----------------------------------------------------------------------===//

namespace {
class ScheduleDAGMI;

/// MachineSchedStrategy - Interface used by ScheduleDAGMI to drive the selected
/// scheduling algorithm.
///
/// If this works well and targets wish to reuse ScheduleDAGMI, we may expose it
/// in ScheduleDAGInstrs.h
class MachineSchedStrategy {
public:
  virtual ~MachineSchedStrategy() {}

  /// Initialize the strategy after building the DAG for a new region.
  virtual void initialize(ScheduleDAGMI *DAG) = 0;

  /// Pick the next node to schedule, or return NULL. Set IsTopNode to true to
  /// schedule the node at the top of the unscheduled region. Otherwise it will
  /// be scheduled at the bottom.
  virtual SUnit *pickNode(bool &IsTopNode) = 0;

  /// When all predecessor dependencies have been resolved, free this node for
  /// top-down scheduling.
  virtual void releaseTopNode(SUnit *SU) = 0;
  /// When all successor dependencies have been resolved, free this node for
  /// bottom-up scheduling.
  virtual void releaseBottomNode(SUnit *SU) = 0;
};
} // namespace

//===----------------------------------------------------------------------===//
// ScheduleDAGMI - Base class for MachineInstr scheduling with LiveIntervals
// preservation.
//===----------------------------------------------------------------------===//

namespace {
/// ScheduleDAGMI is an implementation of ScheduleDAGInstrs that schedules
/// machine instructions while updating LiveIntervals.
class ScheduleDAGMI : public ScheduleDAGInstrs {
  AliasAnalysis *AA;
  RegisterClassInfo *RegClassInfo;
  MachineSchedStrategy *SchedImpl;

  MachineBasicBlock::iterator LiveRegionEnd;

  // Register pressure in this region computed by buildSchedGraph.
  IntervalPressure RegPressure;
  RegPressureTracker RPTracker;

  /// The top of the unscheduled zone.
  MachineBasicBlock::iterator CurrentTop;
  IntervalPressure TopPressure;
  RegPressureTracker TopRPTracker;

  /// The bottom of the unscheduled zone.
  MachineBasicBlock::iterator CurrentBottom;
  IntervalPressure BotPressure;
  RegPressureTracker BotRPTracker;

  /// The number of instructions scheduled so far. Used to cut off the
  /// scheduler at the point determined by misched-cutoff.
  unsigned NumInstrsScheduled;
public:
  ScheduleDAGMI(MachineSchedContext *C, MachineSchedStrategy *S):
    ScheduleDAGInstrs(*C->MF, *C->MLI, *C->MDT, /*IsPostRA=*/false, C->LIS),
    AA(C->AA), RegClassInfo(C->RegClassInfo), SchedImpl(S),
    RPTracker(RegPressure), CurrentTop(), TopRPTracker(TopPressure),
    CurrentBottom(), BotRPTracker(BotPressure), NumInstrsScheduled(0) {}

  ~ScheduleDAGMI() {
    delete SchedImpl;
  }

  MachineBasicBlock::iterator top() const { return CurrentTop; }
  MachineBasicBlock::iterator bottom() const { return CurrentBottom; }

  /// Get current register pressure for the top scheduled instructions.
  const IntervalPressure &getTopPressure() const { return TopPressure; }

  /// Get current register pressure for the bottom scheduled instructions.
  const IntervalPressure &getBotPressure() const { return BotPressure; }

  /// Get register pressure for the entire scheduling region before scheduling.
  const IntervalPressure &getRegPressure() const { return RegPressure; }

  /// Implement the ScheduleDAGInstrs interface for handling the next scheduling
  /// region. This covers all instructions in a block, while schedule() may only
  /// cover a subset.
  void enterRegion(MachineBasicBlock *bb,
                   MachineBasicBlock::iterator begin,
                   MachineBasicBlock::iterator end,
                   unsigned endcount);

  /// Implement ScheduleDAGInstrs interface for scheduling a sequence of
  /// reorderable instructions.
  void schedule();

protected:
  void initRegPressure();

  void moveInstruction(MachineInstr *MI, MachineBasicBlock::iterator InsertPos);
  bool checkSchedLimit();

  void releaseSucc(SUnit *SU, SDep *SuccEdge);
  void releaseSuccessors(SUnit *SU);
  void releasePred(SUnit *SU, SDep *PredEdge);
  void releasePredecessors(SUnit *SU);

  void placeDebugValues();
};
} // namespace

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. When
/// NumPredsLeft reaches zero, release the successor node.
void ScheduleDAGMI::releaseSucc(SUnit *SU, SDep *SuccEdge) {
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
    SchedImpl->releaseTopNode(SuccSU);
}

/// releaseSuccessors - Call releaseSucc on each of SU's successors.
void ScheduleDAGMI::releaseSuccessors(SUnit *SU) {
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    releaseSucc(SU, &*I);
  }
}

/// ReleasePred - Decrement the NumSuccsLeft count of a predecessor. When
/// NumSuccsLeft reaches zero, release the predecessor node.
void ScheduleDAGMI::releasePred(SUnit *SU, SDep *PredEdge) {
  SUnit *PredSU = PredEdge->getSUnit();

#ifndef NDEBUG
  if (PredSU->NumSuccsLeft == 0) {
    dbgs() << "*** Scheduling failed! ***\n";
    PredSU->dump(this);
    dbgs() << " has been released too many times!\n";
    llvm_unreachable(0);
  }
#endif
  --PredSU->NumSuccsLeft;
  if (PredSU->NumSuccsLeft == 0 && PredSU != &EntrySU)
    SchedImpl->releaseBottomNode(PredSU);
}

/// releasePredecessors - Call releasePred on each of SU's predecessors.
void ScheduleDAGMI::releasePredecessors(SUnit *SU) {
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    releasePred(SU, &*I);
  }
}

void ScheduleDAGMI::moveInstruction(MachineInstr *MI,
                                    MachineBasicBlock::iterator InsertPos) {
  // Fix RegionBegin if the first instruction moves down.
  if (&*RegionBegin == MI)
    RegionBegin = llvm::next(RegionBegin);
  BB->splice(InsertPos, BB, MI);
  LIS->handleMove(MI);
  // Fix RegionBegin if another instruction moves above the first instruction.
  if (RegionBegin == InsertPos)
    RegionBegin = MI;
  // Fix TopRPTracker if we move something above CurrentTop.
  if (CurrentTop == InsertPos)
    TopRPTracker.setPos(MI);
}

bool ScheduleDAGMI::checkSchedLimit() {
#ifndef NDEBUG
  if (NumInstrsScheduled == MISchedCutoff && MISchedCutoff != ~0U) {
    CurrentTop = CurrentBottom;
    return false;
  }
  ++NumInstrsScheduled;
#endif
  return true;
}

/// enterRegion - Called back from MachineScheduler::runOnMachineFunction after
/// crossing a scheduling boundary. [begin, end) includes all instructions in
/// the region, including the boundary itself and single-instruction regions
/// that don't get scheduled.
void ScheduleDAGMI::enterRegion(MachineBasicBlock *bb,
                                MachineBasicBlock::iterator begin,
                                MachineBasicBlock::iterator end,
                                unsigned endcount)
{
  ScheduleDAGInstrs::enterRegion(bb, begin, end, endcount);

  // For convenience remember the end of the liveness region.
  LiveRegionEnd =
    (RegionEnd == bb->end()) ? RegionEnd : llvm::next(RegionEnd);
}

// Setup the register pressure trackers for the top scheduled top and bottom
// scheduled regions.
void ScheduleDAGMI::initRegPressure() {
  TopRPTracker.init(&MF, RegClassInfo, LIS, BB, RegionBegin);
  BotRPTracker.init(&MF, RegClassInfo, LIS, BB, LiveRegionEnd);

  // Close the RPTracker to finalize live ins.
  RPTracker.closeRegion();

  // Initialize the live ins and live outs.
  TopRPTracker.addLiveRegs(RPTracker.getPressure().LiveInRegs);
  BotRPTracker.addLiveRegs(RPTracker.getPressure().LiveOutRegs);

  // Close one end of the tracker so we can call
  // getMaxUpward/DownwardPressureDelta before advancing across any
  // instructions. This converts currently live regs into live ins/outs.
  TopRPTracker.closeTop();
  BotRPTracker.closeBottom();

  // Account for liveness generated by the region boundary.
  if (LiveRegionEnd != RegionEnd)
    BotRPTracker.recede();

  assert(BotRPTracker.getPos() == RegionEnd && "Can't find the region bottom");
}

/// schedule - Called back from MachineScheduler::runOnMachineFunction
/// after setting up the current scheduling region. [RegionBegin, RegionEnd)
/// only includes instructions that have DAG nodes, not scheduling boundaries.
void ScheduleDAGMI::schedule() {
  // Initialize the register pressure tracker used by buildSchedGraph.
  RPTracker.init(&MF, RegClassInfo, LIS, BB, LiveRegionEnd);

  // Account for liveness generate by the region boundary.
  if (LiveRegionEnd != RegionEnd)
    RPTracker.recede();

  // Build the DAG, and compute current register pressure.
  buildSchedGraph(AA, &RPTracker);

  // Initialize top/bottom trackers after computing region pressure.
  initRegPressure();

  DEBUG(dbgs() << "********** MI Scheduling **********\n");
  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(this));

  if (ViewMISchedDAGs) viewGraph();

  SchedImpl->initialize(this);

  // Release edges from the special Entry node or to the special Exit node.
  releaseSuccessors(&EntrySU);
  releasePredecessors(&ExitSU);

  // Release all DAG roots for scheduling.
  for (std::vector<SUnit>::iterator I = SUnits.begin(), E = SUnits.end();
       I != E; ++I) {
    // A SUnit is ready to top schedule if it has no predecessors.
    if (I->Preds.empty())
      SchedImpl->releaseTopNode(&(*I));
    // A SUnit is ready to bottom schedule if it has no successors.
    if (I->Succs.empty())
      SchedImpl->releaseBottomNode(&(*I));
  }

  CurrentTop = nextIfDebug(RegionBegin, RegionEnd);
  CurrentBottom = RegionEnd;
  bool IsTopNode = false;
  while (SUnit *SU = SchedImpl->pickNode(IsTopNode)) {
    DEBUG(dbgs() << "*** " << (IsTopNode ? "Top" : "Bottom")
          << " Scheduling Instruction:\n"; SU->dump(this));
    if (!checkSchedLimit())
      break;

    // Move the instruction to its new location in the instruction stream.
    MachineInstr *MI = SU->getInstr();

    if (IsTopNode) {
      assert(SU->isTopReady() && "node still has unscheduled dependencies");
      if (&*CurrentTop == MI)
        CurrentTop = nextIfDebug(++CurrentTop, CurrentBottom);
      else
        moveInstruction(MI, CurrentTop);

      // Update top scheduled pressure.
      TopRPTracker.advance();
      assert(TopRPTracker.getPos() == CurrentTop && "out of sync");

      // Release dependent instructions for scheduling.
      releaseSuccessors(SU);
    }
    else {
      assert(SU->isBottomReady() && "node still has unscheduled dependencies");
      MachineBasicBlock::iterator priorII =
        priorNonDebug(CurrentBottom, CurrentTop);
      if (&*priorII == MI)
        CurrentBottom = priorII;
      else {
        if (&*CurrentTop == MI)
          CurrentTop = nextIfDebug(++CurrentTop, CurrentBottom);
        moveInstruction(MI, CurrentBottom);
        CurrentBottom = MI;
      }
      // Update bottom scheduled pressure.
      BotRPTracker.recede();
      assert(BotRPTracker.getPos() == CurrentBottom && "out of sync");

      // Release dependent instructions for scheduling.
      releasePredecessors(SU);
    }
    SU->isScheduled = true;
  }
  assert(CurrentTop == CurrentBottom && "Nonempty unscheduled zone.");

  placeDebugValues();
}

/// Reinsert any remaining debug_values, just like the PostRA scheduler.
void ScheduleDAGMI::placeDebugValues() {
  // If first instruction was a DBG_VALUE then put it back.
  if (FirstDbgValue) {
    BB->splice(RegionBegin, BB, FirstDbgValue);
    RegionBegin = FirstDbgValue;
  }

  for (std::vector<std::pair<MachineInstr *, MachineInstr *> >::iterator
         DI = DbgValues.end(), DE = DbgValues.begin(); DI != DE; --DI) {
    std::pair<MachineInstr *, MachineInstr *> P = *prior(DI);
    MachineInstr *DbgValue = P.first;
    MachineBasicBlock::iterator OrigPrevMI = P.second;
    BB->splice(++OrigPrevMI, BB, DbgValue);
    if (OrigPrevMI == llvm::prior(RegionEnd))
      RegionEnd = DbgValue;
  }
  DbgValues.clear();
  FirstDbgValue = NULL;
}

//===----------------------------------------------------------------------===//
// ConvergingScheduler - Implementation of the standard MachineSchedStrategy.
//===----------------------------------------------------------------------===//

namespace {
/// ConvergingScheduler shrinks the unscheduled zone using heuristics to balance
/// the schedule.
class ConvergingScheduler : public MachineSchedStrategy {
  ScheduleDAGMI *DAG;

  unsigned NumTopReady;
  unsigned NumBottomReady;

public:
  virtual void initialize(ScheduleDAGMI *dag) {
    DAG = dag;

    assert((!ForceTopDown || !ForceBottomUp) &&
           "-misched-topdown incompatible with -misched-bottomup");
  }

  virtual SUnit *pickNode(bool &IsTopNode) {
    if (DAG->top() == DAG->bottom())
      return NULL;

    // As an initial placeholder heuristic, schedule in the direction that has
    // the fewest choices.
    SUnit *SU;
    if (ForceTopDown || (!ForceBottomUp && NumTopReady <= NumBottomReady)) {
      SU = DAG->getSUnit(DAG->top());
      IsTopNode = true;
    }
    else {
      SU = DAG->getSUnit(priorNonDebug(DAG->bottom(), DAG->top()));
      IsTopNode = false;
    }
    if (SU->isTopReady()) {
      assert(NumTopReady > 0 && "bad ready count");
      --NumTopReady;
    }
    if (SU->isBottomReady()) {
      assert(NumBottomReady > 0 && "bad ready count");
      --NumBottomReady;
    }
    return SU;
  }

  virtual void releaseTopNode(SUnit *SU) {
    ++NumTopReady;
  }
  virtual void releaseBottomNode(SUnit *SU) {
    ++NumBottomReady;
  }
};
} // namespace

/// Create the standard converging machine scheduler. This will be used as the
/// default scheduler if the target does not set a default.
static ScheduleDAGInstrs *createConvergingSched(MachineSchedContext *C) {
  assert((!ForceTopDown || !ForceBottomUp) &&
         "-misched-topdown incompatible with -misched-bottomup");
  return new ScheduleDAGMI(C, new ConvergingScheduler());
}
static MachineSchedRegistry
ConvergingSchedRegistry("converge", "Standard converging scheduler.",
                        createConvergingSched);

//===----------------------------------------------------------------------===//
// Machine Instruction Shuffler for Correctness Testing
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
namespace {
/// Apply a less-than relation on the node order, which corresponds to the
/// instruction order prior to scheduling. IsReverse implements greater-than.
template<bool IsReverse>
struct SUnitOrder {
  bool operator()(SUnit *A, SUnit *B) const {
    if (IsReverse)
      return A->NodeNum > B->NodeNum;
    else
      return A->NodeNum < B->NodeNum;
  }
};

/// Reorder instructions as much as possible.
class InstructionShuffler : public MachineSchedStrategy {
  bool IsAlternating;
  bool IsTopDown;

  // Using a less-than relation (SUnitOrder<false>) for the TopQ priority
  // gives nodes with a higher number higher priority causing the latest
  // instructions to be scheduled first.
  PriorityQueue<SUnit*, std::vector<SUnit*>, SUnitOrder<false> >
    TopQ;
  // When scheduling bottom-up, use greater-than as the queue priority.
  PriorityQueue<SUnit*, std::vector<SUnit*>, SUnitOrder<true> >
    BottomQ;
public:
  InstructionShuffler(bool alternate, bool topdown)
    : IsAlternating(alternate), IsTopDown(topdown) {}

  virtual void initialize(ScheduleDAGMI *) {
    TopQ.clear();
    BottomQ.clear();
  }

  /// Implement MachineSchedStrategy interface.
  /// -----------------------------------------

  virtual SUnit *pickNode(bool &IsTopNode) {
    SUnit *SU;
    if (IsTopDown) {
      do {
        if (TopQ.empty()) return NULL;
        SU = TopQ.top();
        TopQ.pop();
      } while (SU->isScheduled);
      IsTopNode = true;
    }
    else {
      do {
        if (BottomQ.empty()) return NULL;
        SU = BottomQ.top();
        BottomQ.pop();
      } while (SU->isScheduled);
      IsTopNode = false;
    }
    if (IsAlternating)
      IsTopDown = !IsTopDown;
    return SU;
  }

  virtual void releaseTopNode(SUnit *SU) {
    TopQ.push(SU);
  }
  virtual void releaseBottomNode(SUnit *SU) {
    BottomQ.push(SU);
  }
};
} // namespace

static ScheduleDAGInstrs *createInstructionShuffler(MachineSchedContext *C) {
  bool Alternate = !ForceTopDown && !ForceBottomUp;
  bool TopDown = !ForceBottomUp;
  assert((TopDown || !ForceTopDown) &&
         "-misched-topdown incompatible with -misched-bottomup");
  return new ScheduleDAGMI(C, new InstructionShuffler(Alternate, TopDown));
}
static MachineSchedRegistry ShufflerRegistry(
  "shuffle", "Shuffle machine instructions alternating directions",
  createInstructionShuffler);
#endif // !NDEBUG
