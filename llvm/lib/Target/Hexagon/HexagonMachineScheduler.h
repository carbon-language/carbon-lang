//===-- HexagonMachineScheduler.h - Custom Hexagon MI scheduler.      ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Custom Hexagon MI scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONASMPRINTER_H
#define HEXAGONASMPRINTER_H

#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ResourcePriorityQueue.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/PriorityQueue.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// MachineSchedStrategy - Interface to a machine scheduling algorithm.
//===----------------------------------------------------------------------===//

namespace llvm {
class VLIWMachineScheduler;

/// MachineSchedStrategy - Interface used by VLIWMachineScheduler to drive
/// the selected scheduling algorithm.
///
/// TODO: Move this to ScheduleDAGInstrs.h
class MachineSchedStrategy {
public:
  virtual ~MachineSchedStrategy() {}

  /// Initialize the strategy after building the DAG for a new region.
  virtual void initialize(VLIWMachineScheduler *DAG) = 0;

  /// Pick the next node to schedule, or return NULL. Set IsTopNode to true to
  /// schedule the node at the top of the unscheduled region. Otherwise it will
  /// be scheduled at the bottom.
  virtual SUnit *pickNode(bool &IsTopNode) = 0;

  /// Notify MachineSchedStrategy that VLIWMachineScheduler has
  /// scheduled a node.
  virtual void schedNode(SUnit *SU, bool IsTopNode) = 0;

  /// When all predecessor dependencies have been resolved, free this node for
  /// top-down scheduling.
  virtual void releaseTopNode(SUnit *SU) = 0;
  /// When all successor dependencies have been resolved, free this node for
  /// bottom-up scheduling.
  virtual void releaseBottomNode(SUnit *SU) = 0;
};

//===----------------------------------------------------------------------===//
// ConvergingVLIWScheduler - Implementation of the standard
// MachineSchedStrategy.
//===----------------------------------------------------------------------===//

/// ReadyQueue encapsulates vector of "ready" SUnits with basic convenience
/// methods for pushing and removing nodes. ReadyQueue's are uniquely identified
/// by an ID. SUnit::NodeQueueId is a mask of the ReadyQueues the SUnit is in.
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

  unsigned size() const { return Queue.size(); }

  typedef std::vector<SUnit*>::iterator iterator;

  iterator begin() { return Queue.begin(); }

  iterator end() { return Queue.end(); }

  iterator find(SUnit *SU) {
    return std::find(Queue.begin(), Queue.end(), SU);
  }

  void push(SUnit *SU) {
    Queue.push_back(SU);
    SU->NodeQueueId |= ID;
  }

  void remove(iterator I) {
    (*I)->NodeQueueId &= ~ID;
    *I = Queue.back();
    Queue.pop_back();
  }

  void dump() {
    dbgs() << Name << ": ";
    for (unsigned i = 0, e = Queue.size(); i < e; ++i)
      dbgs() << Queue[i]->NodeNum << " ";
    dbgs() << "\n";
  }
};

class VLIWResourceModel {
  /// ResourcesModel - Represents VLIW state.
  /// Not limited to VLIW targets per say, but assumes
  /// definition of DFA by a target.
  DFAPacketizer *ResourcesModel;

  const InstrItineraryData *InstrItins;

  /// Local packet/bundle model. Purely
  /// internal to the MI schedulre at the time.
  std::vector<SUnit*> Packet;

  /// Total packets created.
  unsigned TotalPackets;

public:
  VLIWResourceModel(MachineSchedContext *C, const InstrItineraryData *IID) :
    InstrItins(IID), TotalPackets(0) {
    const TargetMachine &TM = C->MF->getTarget();
    ResourcesModel = TM.getInstrInfo()->CreateTargetScheduleState(&TM,NULL);

    // This hard requirement could be relaxed,
    // but for now do not let it proceed.
    assert(ResourcesModel && "Unimplemented CreateTargetScheduleState.");

    Packet.resize(InstrItins->SchedModel->IssueWidth);
    Packet.clear();
    ResourcesModel->clearResources();
  }

  VLIWResourceModel(const TargetMachine &TM) :
    InstrItins(TM.getInstrItineraryData()), TotalPackets(0) {
    ResourcesModel = TM.getInstrInfo()->CreateTargetScheduleState(&TM,NULL);

    // This hard requirement could be relaxed,
    // but for now do not let it proceed.
    assert(ResourcesModel && "Unimplemented CreateTargetScheduleState.");

    Packet.resize(InstrItins->SchedModel->IssueWidth);
    Packet.clear();
    ResourcesModel->clearResources();
  }

  ~VLIWResourceModel() {
    delete ResourcesModel;
  }

  void resetPacketState() {
    Packet.clear();
  }

  void resetDFA() {
    ResourcesModel->clearResources();
  }

  void reset() {
    Packet.clear();
    ResourcesModel->clearResources();
  }

  bool isResourceAvailable(SUnit *SU);
  bool reserveResources(SUnit *SU);
  unsigned getTotalPackets() const { return TotalPackets; }
};

class VLIWMachineScheduler : public ScheduleDAGInstrs {
  /// AA - AliasAnalysis for making memory reference queries.
  AliasAnalysis *AA;

  RegisterClassInfo *RegClassInfo;
  MachineSchedStrategy *SchedImpl;

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

#ifndef NDEBUG
  /// The number of instructions scheduled so far. Used to cut off the
  /// scheduler at the point determined by misched-cutoff.
  unsigned NumInstrsScheduled;
#endif

  /// Total packets in the region.
  unsigned TotalPackets;

  const MachineLoopInfo *MLI;
public:
  VLIWMachineScheduler(MachineSchedContext *C, MachineSchedStrategy *S):
    ScheduleDAGInstrs(*C->MF, *C->MLI, *C->MDT, /*IsPostRA=*/false, C->LIS),
    AA(C->AA), RegClassInfo(C->RegClassInfo), SchedImpl(S),
    RPTracker(RegPressure), CurrentTop(), TopRPTracker(TopPressure),
    CurrentBottom(), BotRPTracker(BotPressure), MLI(C->MLI) {
#ifndef NDEBUG
    NumInstrsScheduled = 0;
#endif
    TotalPackets = 0;
  }

  virtual ~VLIWMachineScheduler() {
    delete SchedImpl;
  }

  MachineBasicBlock::iterator top() const { return CurrentTop; }
  MachineBasicBlock::iterator bottom() const { return CurrentBottom; }

  /// Implement the ScheduleDAGInstrs interface for handling the next scheduling
  /// region. This covers all instructions in a block, while schedule() may only
  /// cover a subset.
  void enterRegion(MachineBasicBlock *bb,
                   MachineBasicBlock::iterator begin,
                   MachineBasicBlock::iterator end,
                   unsigned endcount);

  /// Schedule - This is called back from ScheduleDAGInstrs::Run() when it's
  /// time to do some work.
  void schedule();

  unsigned CurCycle;

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

  /// getIssueWidth - Return the max instructions per scheduling group.
  unsigned getIssueWidth() const {
    return (InstrItins && InstrItins->SchedModel)
      ? InstrItins->SchedModel->IssueWidth : 1;
  }

  /// getNumMicroOps - Return the number of issue slots required for this MI.
  unsigned getNumMicroOps(MachineInstr *MI) const {
    return 1;
    //if (!InstrItins) return 1;
    //int UOps = InstrItins->getNumMicroOps(MI->getDesc().getSchedClass());
    //return (UOps >= 0) ? UOps : TII->getNumMicroOps(InstrItins, MI);
  }

private:
  void scheduleNodeTopDown(SUnit *SU);
  void listScheduleTopDown();

  void initRegPressure();
  void updateScheduledPressure(std::vector<unsigned> NewMaxPressure);

  void moveInstruction(MachineInstr *MI, MachineBasicBlock::iterator InsertPos);
  bool checkSchedLimit();

  void releaseRoots();

  void releaseSucc(SUnit *SU, SDep *SuccEdge);
  void releaseSuccessors(SUnit *SU);
  void releasePred(SUnit *SU, SDep *PredEdge);
  void releasePredecessors(SUnit *SU);

  void placeDebugValues();
};

/// ConvergingVLIWScheduler shrinks the unscheduled zone using heuristics
/// to balance the schedule.
class ConvergingVLIWScheduler : public MachineSchedStrategy {

  /// Store the state used by ConvergingVLIWScheduler heuristics, required
  ///  for the lifetime of one invocation of pickNode().
  struct SchedCandidate {
    // The best SUnit candidate.
    SUnit *SU;

    // Register pressure values for the best candidate.
    RegPressureDelta RPDelta;

    // Best scheduling cost.
    int SCost;

    SchedCandidate(): SU(NULL), SCost(0) {}
  };
  /// Represent the type of SchedCandidate found within a single queue.
  enum CandResult {
    NoCand, NodeOrder, SingleExcess, SingleCritical, SingleMax, MultiPressure,
    BestCost};

  /// Each Scheduling boundary is associated with ready queues. It tracks the
  /// current cycle in whichever direction at has moved, and maintains the state
  /// of "hazards" and other interlocks at the current cycle.
  struct SchedBoundary {
    VLIWMachineScheduler *DAG;

    ReadyQueue Available;
    ReadyQueue Pending;
    bool CheckPending;

    ScheduleHazardRecognizer *HazardRec;
    VLIWResourceModel *ResourceModel;

    unsigned CurrCycle;
    unsigned IssueCount;

    /// MinReadyCycle - Cycle of the soonest available instruction.
    unsigned MinReadyCycle;

    // Remember the greatest min operand latency.
    unsigned MaxMinLatency;

    /// Pending queues extend the ready queues with the same ID and the
    /// PendingFlag set.
    SchedBoundary(unsigned ID, const Twine &Name):
      DAG(0), Available(ID, Name+".A"),
      Pending(ID << ConvergingVLIWScheduler::LogMaxQID, Name+".P"),
      CheckPending(false), HazardRec(0), ResourceModel(0),
      CurrCycle(0), IssueCount(0),
      MinReadyCycle(UINT_MAX), MaxMinLatency(0) {}

    ~SchedBoundary() {
      delete ResourceModel;
      delete HazardRec;
    }

    bool isTop() const {
      return Available.getID() == ConvergingVLIWScheduler::TopQID;
    }

    bool checkHazard(SUnit *SU);

    void releaseNode(SUnit *SU, unsigned ReadyCycle);

    void bumpCycle();

    void bumpNode(SUnit *SU);

    void releasePending();

    void removeReady(SUnit *SU);

    SUnit *pickOnlyChoice();
  };

  VLIWMachineScheduler *DAG;
  const TargetRegisterInfo *TRI;

  // State of the top and bottom scheduled instruction boundaries.
  SchedBoundary Top;
  SchedBoundary Bot;

public:
  /// SUnit::NodeQueueId: 0 (none), 1 (top), 2 (bot), 3 (both)
  enum {
    TopQID = 1,
    BotQID = 2,
    LogMaxQID = 2
  };

  ConvergingVLIWScheduler():
    DAG(0), TRI(0), Top(TopQID, "TopQ"), Bot(BotQID, "BotQ") {}

  virtual void initialize(VLIWMachineScheduler *dag);

  virtual SUnit *pickNode(bool &IsTopNode);

  virtual void schedNode(SUnit *SU, bool IsTopNode);

  virtual void releaseTopNode(SUnit *SU);

  virtual void releaseBottomNode(SUnit *SU);

protected:
  SUnit *pickNodeBidrectional(bool &IsTopNode);

  int SchedulingCost(ReadyQueue &Q,
                     SUnit *SU, SchedCandidate &Candidate,
                     RegPressureDelta &Delta, bool verbose);

  CandResult pickNodeFromQueue(ReadyQueue &Q,
                               const RegPressureTracker &RPTracker,
                               SchedCandidate &Candidate);
#ifndef NDEBUG
  void traceCandidate(const char *Label, const ReadyQueue &Q, SUnit *SU,
                      PressureElement P = PressureElement());
#endif
};

} // namespace


#endif
