//==- MachineScheduler.h - MachineInstr Scheduling Pass ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a MachineSchedRegistry for registering alternative machine
// schedulers. A Target may provide an alternative scheduler implementation by
// implementing the following boilerplate:
//
// static ScheduleDAGInstrs *createCustomMachineSched(MachineSchedContext *C) {
//  return new CustomMachineScheduler(C);
// }
// static MachineSchedRegistry
// SchedCustomRegistry("custom", "Run my target's custom scheduler",
//                     createCustomMachineSched);
//
// Inside <Target>PassConfig:
//   enablePass(&MachineSchedulerID);
//   MachineSchedRegistry::setDefault(createCustomMachineSched);
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
  static ScheduleDAGCtor getDefault() {
    return (ScheduleDAGCtor)Registry.getDefault();
  }
  static void setDefault(ScheduleDAGCtor C) {
    Registry.setDefault((MachinePassCtor)C);
  }
  static void setDefault(StringRef Name) {
    Registry.setDefault(Name);
  }
  static void setListener(MachinePassRegistryListener *L) {
    Registry.setListener(L);
  }
};

class ScheduleDAGMI;

/// MachineSchedStrategy - Interface to the scheduling algorithm used by
/// ScheduleDAGMI.
class MachineSchedStrategy {
public:
  virtual ~MachineSchedStrategy() {}

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

  /// Register pressure in this region computed by buildSchedGraph.
  IntervalPressure RegPressure;
  RegPressureTracker RPTracker;

  /// List of pressure sets that exceed the target's pressure limit before
  /// scheduling, listed in increasing set ID order. Each pressure set is paired
  /// with its max pressure in the currently scheduled regions.
  std::vector<PressureElement> RegionCriticalPSets;

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
    Topo(SUnits, &ExitSU), RPTracker(RegPressure), CurrentTop(),
    TopRPTracker(TopPressure), CurrentBottom(), BotRPTracker(BotPressure),
    NextClusterPred(NULL), NextClusterSucc(NULL) {
#ifndef NDEBUG
    NumInstrsScheduled = 0;
#endif
  }

  virtual ~ScheduleDAGMI();

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
                   unsigned endcount);


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

  const std::vector<PressureElement> &getRegionCriticalPSets() const {
    return RegionCriticalPSets;
  }

  const SUnit *getNextClusterPred() const { return NextClusterPred; }

  const SUnit *getNextClusterSucc() const { return NextClusterSucc; }

  /// Compute a DFSResult after DAG building is complete, and before any
  /// queue comparisons.
  void computeDFSResult();

  /// Return a non-null DFS result if the scheduling strategy initialized it.
  const SchedDFSResult *getDFSResult() const { return DFSResult; }

  BitVector &getScheduledTrees() { return ScheduledTrees; }

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

  void updateScheduledPressure(const std::vector<unsigned> &NewMaxPressure);

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
