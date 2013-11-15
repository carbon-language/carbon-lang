//==- MachineScheduler.h - MachineInstr Scheduling Pass ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides an interface for customizing the standard MachineScheduler
// pass. Note that the entire pass may be replaced as follows:
//
// <Target>TargetMachine::createPassConfig(PassManagerBase &PM) {
//   PM.substitutePass(&MachineSchedulerID, &CustomSchedulerPassID);
//   ...}
//
// The MachineScheduler pass is only responsible for choosing the regions to be
// scheduled. Targets can override the DAG builder and scheduler without
// replacing the pass as follows:
//
// ScheduleDAGInstrs *<Target>PassConfig::
// createMachineScheduler(MachineSchedContext *C) {
//   return new CustomMachineScheduler(C);
// }
//
// The default scheduler, ScheduleDAGMI, builds the DAG and drives list
// scheduling while updating the instruction stream, register pressure, and live
// intervals. Most targets don't need to override the DAG builder and list
// schedulier, but subtargets that require custom scheduling heuristics may
// plugin an alternate MachineSchedStrategy. The strategy is responsible for
// selecting the highest priority node from the list:
//
// ScheduleDAGInstrs *<Target>PassConfig::
// createMachineScheduler(MachineSchedContext *C) {
//   return new ScheduleDAGMI(C, CustomStrategy(C));
// }
//
// The DAG builder can also be customized in a sense by adding DAG mutations
// that will run after DAG building and before list scheduling. DAG mutations
// can adjust dependencies based on target-specific knowledge or add weak edges
// to aid heuristics:
//
// ScheduleDAGInstrs *<Target>PassConfig::
// createMachineScheduler(MachineSchedContext *C) {
//   ScheduleDAGMI *DAG = new ScheduleDAGMI(C, CustomStrategy(C));
//   DAG->addMutation(new CustomDependencies(DAG->TII, DAG->TRI));
//   return DAG;
// }
//
// A target that supports alternative schedulers can use the
// MachineSchedRegistry to allow command line selection. This can be done by
// implementing the following boilerplate:
//
// static ScheduleDAGInstrs *createCustomMachineSched(MachineSchedContext *C) {
//  return new CustomMachineScheduler(C);
// }
// static MachineSchedRegistry
// SchedCustomRegistry("custom", "Run my target's custom scheduler",
//                     createCustomMachineSched);
//
//
// Finally, subtargets that don't need to implement custom heuristics but would
// like to configure the GenericScheduler's policy for a given scheduler region,
// including scheduling direction and register pressure tracking policy, can do
// this:
//
// void <SubTarget>Subtarget::
// overrideSchedPolicy(MachineSchedPolicy &Policy,
//                     MachineInstr *begin,
//                     MachineInstr *end,
//                     unsigned NumRegionInstrs) const {
//   Policy.<Flag> = true;
// }
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINESCHEDULER_H
#define LLVM_CODEGEN_MACHINESCHEDULER_H

#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

namespace llvm {

extern cl::opt<bool> ForceTopDown;
extern cl::opt<bool> ForceBottomUp;

class AliasAnalysis;
class LiveIntervals;
class MachineDominatorTree;
class MachineLoopInfo;
class RegisterClassInfo;
class ScheduleDAGInstrs;
class SchedDFSResult;

/// MachineSchedContext provides enough context from the MachineScheduler pass
/// for the target to instantiate a scheduler.
struct MachineSchedContext {
  MachineFunction *MF;
  const MachineLoopInfo *MLI;
  const MachineDominatorTree *MDT;
  const TargetPassConfig *PassConfig;
  AliasAnalysis *AA;
  LiveIntervals *LIS;

  RegisterClassInfo *RegClassInfo;

  MachineSchedContext();
  virtual ~MachineSchedContext();
};

/// MachineSchedRegistry provides a selection of available machine instruction
/// schedulers.
class MachineSchedRegistry : public MachinePassRegistryNode {
public:
  typedef ScheduleDAGInstrs *(*ScheduleDAGCtor)(MachineSchedContext *);

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
  static void setListener(MachinePassRegistryListener *L) {
    Registry.setListener(L);
  }
};

class ScheduleDAGMI;

/// Define a generic scheduling policy for targets that don't provide their own
/// MachineSchedStrategy. This can be overriden for each scheduling region
/// before building the DAG.
struct MachineSchedPolicy {
  // Allow the scheduler to disable register pressure tracking.
  bool ShouldTrackPressure;

  // Allow the scheduler to force top-down or bottom-up scheduling. If neither
  // is true, the scheduler runs in both directions and converges.
  bool OnlyTopDown;
  bool OnlyBottomUp;

  MachineSchedPolicy():
    ShouldTrackPressure(false), OnlyTopDown(false), OnlyBottomUp(false) {}
};

/// MachineSchedStrategy - Interface to the scheduling algorithm used by
/// ScheduleDAGMI.
///
/// Initialization sequence:
///   initPolicy -> shouldTrackPressure -> initialize(DAG) -> registerRoots
class MachineSchedStrategy {
  virtual void anchor();
public:
  virtual ~MachineSchedStrategy() {}

  /// Optionally override the per-region scheduling policy.
  virtual void initPolicy(MachineBasicBlock::iterator Begin,
                          MachineBasicBlock::iterator End,
                          unsigned NumRegionInstrs) {}

  /// Check if pressure tracking is needed before building the DAG and
  /// initializing this strategy. Called after initPolicy.
  virtual bool shouldTrackPressure() const { return true; }

  /// Initialize the strategy after building the DAG for a new region.
  virtual void initialize(ScheduleDAGMI *DAG) = 0;

  /// Notify this strategy that all roots have been released (including those
  /// that depend on EntrySU or ExitSU).
  virtual void registerRoots() {}

  /// Pick the next node to schedule, or return NULL. Set IsTopNode to true to
  /// schedule the node at the top of the unscheduled region. Otherwise it will
  /// be scheduled at the bottom.
  virtual SUnit *pickNode(bool &IsTopNode) = 0;

  /// \brief Scheduler callback to notify that a new subtree is scheduled.
  virtual void scheduleTree(unsigned SubtreeID) {}

  /// Notify MachineSchedStrategy that ScheduleDAGMI has scheduled an
  /// instruction and updated scheduled/remaining flags in the DAG nodes.
  virtual void schedNode(SUnit *SU, bool IsTopNode) = 0;

  /// When all predecessor dependencies have been resolved, free this node for
  /// top-down scheduling.
  virtual void releaseTopNode(SUnit *SU) = 0;
  /// When all successor dependencies have been resolved, free this node for
  /// bottom-up scheduling.
  virtual void releaseBottomNode(SUnit *SU) = 0;
};

/// ReadyQueue encapsulates vector of "ready" SUnits with basic convenience
/// methods for pushing and removing nodes. ReadyQueue's are uniquely identified
/// by an ID. SUnit::NodeQueueId is a mask of the ReadyQueues the SUnit is in.
///
/// This is a convenience class that may be used by implementations of
/// MachineSchedStrategy.
class ReadyQueue {
  unsigned ID;
  std::string Name;
  std::vector<SUnit*> Queue;

public:
  ReadyQueue(unsigned id, const Twine &name): ID(id), Name(name.str()) {}

  unsigned getID() const { return ID; }

  StringRef getName() const { return Name; }

  // SU is in this queue if it's NodeQueueID is a superset of this ID.
  bool isInQueue(SUnit *SU) const { return (SU->NodeQueueId & ID); }

  bool empty() const { return Queue.empty(); }

  void clear() { Queue.clear(); }

  unsigned size() const { return Queue.size(); }

  typedef std::vector<SUnit*>::iterator iterator;

  iterator begin() { return Queue.begin(); }

  iterator end() { return Queue.end(); }

  ArrayRef<SUnit*> elements() { return Queue; }

  iterator find(SUnit *SU) {
    return std::find(Queue.begin(), Queue.end(), SU);
  }

  void push(SUnit *SU) {
    Queue.push_back(SU);
    SU->NodeQueueId |= ID;
  }

  iterator remove(iterator I) {
    (*I)->NodeQueueId &= ~ID;
    *I = Queue.back();
    unsigned idx = I - Queue.begin();
    Queue.pop_back();
    return Queue.begin() + idx;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump();
#endif
};

/// Mutate the DAG as a postpass after normal DAG building.
class ScheduleDAGMutation {
  virtual void anchor();
public:
  virtual ~ScheduleDAGMutation() {}

  virtual void apply(ScheduleDAGMI *DAG) = 0;
};

/// ScheduleDAGMI is an implementation of ScheduleDAGInstrs that schedules
/// machine instructions while updating LiveIntervals and tracking regpressure.
class ScheduleDAGMI : public ScheduleDAGInstrs {
protected:
  AliasAnalysis *AA;
  RegisterClassInfo *RegClassInfo;
  MachineSchedStrategy *SchedImpl;

  /// Information about DAG subtrees. If DFSResult is NULL, then SchedulerTrees
  /// will be empty.
  SchedDFSResult *DFSResult;
  BitVector ScheduledTrees;

  /// Topo - A topological ordering for SUnits which permits fast IsReachable
  /// and similar queries.
  ScheduleDAGTopologicalSort Topo;

  /// Ordered list of DAG postprocessing steps.
  std::vector<ScheduleDAGMutation*> Mutations;

  MachineBasicBlock::iterator LiveRegionEnd;

  // Map each SU to its summary of pressure changes. This array is updated for
  // liveness during bottom-up scheduling. Top-down scheduling may proceed but
  // has no affect on the pressure diffs.
  PressureDiffs SUPressureDiffs;

  /// Register pressure in this region computed by initRegPressure.
  bool ShouldTrackPressure;
  IntervalPressure RegPressure;
  RegPressureTracker RPTracker;

  /// List of pressure sets that exceed the target's pressure limit before
  /// scheduling, listed in increasing set ID order. Each pressure set is paired
  /// with its max pressure in the currently scheduled regions.
  std::vector<PressureChange> RegionCriticalPSets;

  /// The top of the unscheduled zone.
  MachineBasicBlock::iterator CurrentTop;
  IntervalPressure TopPressure;
  RegPressureTracker TopRPTracker;

  /// The bottom of the unscheduled zone.
  MachineBasicBlock::iterator CurrentBottom;
  IntervalPressure BotPressure;
  RegPressureTracker BotRPTracker;

  /// Record the next node in a scheduled cluster.
  const SUnit *NextClusterPred;
  const SUnit *NextClusterSucc;

#ifndef NDEBUG
  /// The number of instructions scheduled so far. Used to cut off the
  /// scheduler at the point determined by misched-cutoff.
  unsigned NumInstrsScheduled;
#endif

public:
  ScheduleDAGMI(MachineSchedContext *C, MachineSchedStrategy *S):
    ScheduleDAGInstrs(*C->MF, *C->MLI, *C->MDT, /*IsPostRA=*/false, C->LIS),
    AA(C->AA), RegClassInfo(C->RegClassInfo), SchedImpl(S), DFSResult(0),
    Topo(SUnits, &ExitSU), ShouldTrackPressure(false),
    RPTracker(RegPressure), CurrentTop(), TopRPTracker(TopPressure),
    CurrentBottom(), BotRPTracker(BotPressure),
    NextClusterPred(NULL), NextClusterSucc(NULL) {
#ifndef NDEBUG
    NumInstrsScheduled = 0;
#endif
  }

  virtual ~ScheduleDAGMI();

  /// \brief Return true if register pressure tracking is enabled.
  bool isTrackingPressure() const { return ShouldTrackPressure; }

  /// Add a postprocessing step to the DAG builder.
  /// Mutations are applied in the order that they are added after normal DAG
  /// building and before MachineSchedStrategy initialization.
  ///
  /// ScheduleDAGMI takes ownership of the Mutation object.
  void addMutation(ScheduleDAGMutation *Mutation) {
    Mutations.push_back(Mutation);
  }

  /// \brief True if an edge can be added from PredSU to SuccSU without creating
  /// a cycle.
  bool canAddEdge(SUnit *SuccSU, SUnit *PredSU);

  /// \brief Add a DAG edge to the given SU with the given predecessor
  /// dependence data.
  ///
  /// \returns true if the edge may be added without creating a cycle OR if an
  /// equivalent edge already existed (false indicates failure).
  bool addEdge(SUnit *SuccSU, const SDep &PredDep);

  MachineBasicBlock::iterator top() const { return CurrentTop; }
  MachineBasicBlock::iterator bottom() const { return CurrentBottom; }

  /// Implement the ScheduleDAGInstrs interface for handling the next scheduling
  /// region. This covers all instructions in a block, while schedule() may only
  /// cover a subset.
  void enterRegion(MachineBasicBlock *bb,
                   MachineBasicBlock::iterator begin,
                   MachineBasicBlock::iterator end,
                   unsigned regioninstrs) LLVM_OVERRIDE;

  /// Implement ScheduleDAGInstrs interface for scheduling a sequence of
  /// reorderable instructions.
  virtual void schedule();

  /// Change the position of an instruction within the basic block and update
  /// live ranges and region boundary iterators.
  void moveInstruction(MachineInstr *MI, MachineBasicBlock::iterator InsertPos);

  /// Get current register pressure for the top scheduled instructions.
  const IntervalPressure &getTopPressure() const { return TopPressure; }
  const RegPressureTracker &getTopRPTracker() const { return TopRPTracker; }

  /// Get current register pressure for the bottom scheduled instructions.
  const IntervalPressure &getBotPressure() const { return BotPressure; }
  const RegPressureTracker &getBotRPTracker() const { return BotRPTracker; }

  /// Get register pressure for the entire scheduling region before scheduling.
  const IntervalPressure &getRegPressure() const { return RegPressure; }

  const std::vector<PressureChange> &getRegionCriticalPSets() const {
    return RegionCriticalPSets;
  }

  PressureDiff &getPressureDiff(const SUnit *SU) {
    return SUPressureDiffs[SU->NodeNum];
  }

  const SUnit *getNextClusterPred() const { return NextClusterPred; }

  const SUnit *getNextClusterSucc() const { return NextClusterSucc; }

  /// Compute a DFSResult after DAG building is complete, and before any
  /// queue comparisons.
  void computeDFSResult();

  /// Return a non-null DFS result if the scheduling strategy initialized it.
  const SchedDFSResult *getDFSResult() const { return DFSResult; }

  BitVector &getScheduledTrees() { return ScheduledTrees; }

  /// Compute the cyclic critical path through the DAG.
  unsigned computeCyclicCriticalPath();

  void viewGraph(const Twine &Name, const Twine &Title) LLVM_OVERRIDE;
  void viewGraph() LLVM_OVERRIDE;

protected:
  // Top-Level entry points for the schedule() driver...

  /// Call ScheduleDAGInstrs::buildSchedGraph with register pressure tracking
  /// enabled. This sets up three trackers. RPTracker will cover the entire DAG
  /// region, TopTracker and BottomTracker will be initialized to the top and
  /// bottom of the DAG region without covereing any unscheduled instruction.
  void buildDAGWithRegPressure();

  /// Apply each ScheduleDAGMutation step in order. This allows different
  /// instances of ScheduleDAGMI to perform custom DAG postprocessing.
  void postprocessDAG();

  /// Release ExitSU predecessors and setup scheduler queues.
  void initQueues(ArrayRef<SUnit*> TopRoots, ArrayRef<SUnit*> BotRoots);

  /// Move an instruction and update register pressure.
  void scheduleMI(SUnit *SU, bool IsTopNode);

  /// Update scheduler DAG and queues after scheduling an instruction.
  void updateQueues(SUnit *SU, bool IsTopNode);

  /// Reinsert debug_values recorded in ScheduleDAGInstrs::DbgValues.
  void placeDebugValues();

  /// \brief dump the scheduled Sequence.
  void dumpSchedule() const;

  // Lesser helpers...

  void initRegPressure();

  void updatePressureDiffs(ArrayRef<unsigned> LiveUses);

  void updateScheduledPressure(const SUnit *SU,
                               const std::vector<unsigned> &NewMaxPressure);

  bool checkSchedLimit();

  void findRootsAndBiasEdges(SmallVectorImpl<SUnit*> &TopRoots,
                             SmallVectorImpl<SUnit*> &BotRoots);

  void releaseSucc(SUnit *SU, SDep *SuccEdge);
  void releaseSuccessors(SUnit *SU);
  void releasePred(SUnit *SU, SDep *PredEdge);
  void releasePredecessors(SUnit *SU);
};

} // namespace llvm

#endif
