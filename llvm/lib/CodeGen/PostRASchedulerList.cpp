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
#include "llvm/Analysis/AliasAnalysis.h"
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
                      cl::init("critical"), cl::Hidden);
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

  class VISIBILITY_HIDDEN SchedulePostRATDList : public ScheduleDAGInstrs {
    /// RegisterReference - Information about a register reference
    /// within a liverange
    typedef struct {
      /// Operand - The registers operand
      MachineOperand *Operand;
      /// RC - The register class
      const TargetRegisterClass *RC;
    } RegisterReference;

    /// AvailableQueue - The priority queue to use for the available SUnits.
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

    /// AA - AliasAnalysis for making memory reference queries.
    AliasAnalysis *AA;

    /// AllocatableSet - The set of allocatable registers.
    /// We'll be ignoring anti-dependencies on non-allocatable registers,
    /// because they may not be safe to break.
    const BitVector AllocatableSet;

    /// GroupNodes - Implements a disjoint-union data structure to
    /// form register groups. A node is represented by an index into
    /// the vector. A node can "point to" itself to indicate that it
    /// is the parent of a group, or point to another node to indicate
    /// that it is a member of the same group as that node.
    std::vector<unsigned> GroupNodes;

    /// GroupNodeIndices - For each register, the index of the GroupNode
    /// currently representing the group that the register belongs to.
    /// Register 0 is always represented by the 0 group, a group
    /// composed of registers that are not eligible for anti-aliasing.
    unsigned GroupNodeIndices[TargetRegisterInfo::FirstVirtualRegister];

    /// RegRegs - Map registers to all their references within a live range.
    std::multimap<unsigned, RegisterReference> RegRefs;

    /// KillIndices - The index of the most recent kill (proceding
    /// bottom-up), or ~0u if no kill of the register has been
    /// seen. The register is live if this index != ~0u and DefIndices
    /// == ~0u.
    unsigned KillIndices[TargetRegisterInfo::FirstVirtualRegister];

    /// DefIndices - The index of the most recent complete def (proceding bottom
    /// up), or ~0u if the register is live.
    unsigned DefIndices[TargetRegisterInfo::FirstVirtualRegister];

  public:
    SchedulePostRATDList(MachineFunction &MF,
                         const MachineLoopInfo &MLI,
                         const MachineDominatorTree &MDT,
                         ScheduleHazardRecognizer *HR,
                         AliasAnalysis *aa)
      : ScheduleDAGInstrs(MF, MLI, MDT), Topo(SUnits),
      HazardRec(HR), AA(aa),
      AllocatableSet(TRI->getAllocatableSet(MF)),
      GroupNodes(TargetRegisterInfo::FirstVirtualRegister, 0) {}

    ~SchedulePostRATDList() {
      delete HazardRec;
    }

    /// StartBlock - Initialize register live-range state for scheduling in
    /// this block.
    ///
    void StartBlock(MachineBasicBlock *BB);

    /// FinishBlock - Clean up register live-range state.
    ///
    void FinishBlock();

    /// Observe - Update liveness information to account for the current
    /// instruction, which will not be scheduled.
    ///
    void Observe(MachineInstr *MI, unsigned Count);

    /// Schedule - Schedule the instruction range using list scheduling.
    ///
    void Schedule();
    
    /// FixupKills - Fix register kill flags that have been made
    /// invalid due to scheduling
    ///
    void FixupKills(MachineBasicBlock *MBB);

  private:
    /// IsLive - Return true if Reg is live
    bool IsLive(unsigned Reg);

    void PrescanInstruction(MachineInstr *MI, unsigned Count);
    void ScanInstruction(MachineInstr *MI, unsigned Count);
    bool BreakAntiDependencies(bool CriticalPathOnly);
    unsigned FindSuitableFreeRegister(unsigned AntiDepReg,
                                      unsigned LastNewReg);

    void ReleaseSucc(SUnit *SU, SDep *SuccEdge);
    void ReleaseSuccessors(SUnit *SU);
    void ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle);
    void ListScheduleTopDown();

    void StartBlockForKills(MachineBasicBlock *BB);
    
    // ToggleKillFlag - Toggle a register operand kill flag. Other
    // adjustments may be made to the instruction if necessary. Return
    // true if the operand has been deleted, false if not.
    bool ToggleKillFlag(MachineInstr *MI, MachineOperand &MO);
    
    // GetGroup - Get the group for a register. The returned value is
    // the index of the GroupNode representing the group.
    unsigned GetGroup(unsigned Reg);
    
    // GetGroupRegs - Return a vector of the registers belonging to a
    // group.
    void GetGroupRegs(unsigned Group, std::vector<unsigned> &Regs);

    // UnionGroups - Union Reg1's and Reg2's groups to form a new
    // group. Return the index of the GroupNode representing the
    // group.
    unsigned UnionGroups(unsigned Reg1, unsigned Reg2);

    // LeaveGroup - Remove a register from its current group and place
    // it alone in its own group. Return the index of the GroupNode
    // representing the registers new group.
    unsigned LeaveGroup(unsigned Reg);
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
  if (EnablePostRAScheduler.getPosition() > 0) {
    if (!EnablePostRAScheduler)
      return false;
  } else {
    // Check that post-RA scheduling is enabled for this target.
    const TargetSubtarget &ST = Fn.getTarget().getSubtarget<TargetSubtarget>();
    if (!ST.enablePostRAScheduler(OptLevel))
      return false;
  }

  DEBUG(errs() << "PostRAScheduler\n");

  const MachineLoopInfo &MLI = getAnalysis<MachineLoopInfo>();
  const MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();
  const InstrItineraryData &InstrItins = Fn.getTarget().getInstrItineraryData();
  ScheduleHazardRecognizer *HR = EnablePostRAHazardAvoidance ?
    (ScheduleHazardRecognizer *)new ExactHazardRecognizer(InstrItins) :
    (ScheduleHazardRecognizer *)new SimpleHazardRecognizer();

  SchedulePostRATDList Scheduler(Fn, MLI, MDT, HR, AA);

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

unsigned SchedulePostRATDList::GetGroup(unsigned Reg)
{
  unsigned Node = GroupNodeIndices[Reg];
  while (GroupNodes[Node] != Node)
    Node = GroupNodes[Node];

  return Node;
}

void SchedulePostRATDList::GetGroupRegs(unsigned Group, std::vector<unsigned> &Regs)
{
  for (unsigned Reg = 0; Reg != TargetRegisterInfo::FirstVirtualRegister; ++Reg) {
    if (GetGroup(Reg) == Group)
      Regs.push_back(Reg);
  }
}

unsigned SchedulePostRATDList::UnionGroups(unsigned Reg1, unsigned Reg2)
{
  assert(GroupNodes[0] == 0 && "GroupNode 0 not parent!");
  assert(GroupNodeIndices[0] == 0 && "Reg 0 not in Group 0!");
  
  // find group for each register
  unsigned Group1 = GetGroup(Reg1);
  unsigned Group2 = GetGroup(Reg2);
  
  // if either group is 0, then that must become the parent
  unsigned Parent = (Group1 == 0) ? Group1 : Group2;
  unsigned Other = (Parent == Group1) ? Group2 : Group1;
  GroupNodes.at(Other) = Parent;
  return Parent;
}
  
unsigned SchedulePostRATDList::LeaveGroup(unsigned Reg)
{
  // Create a new GroupNode for Reg. Reg's existing GroupNode must
  // stay as is because there could be other GroupNodes referring to
  // it.
  unsigned idx = GroupNodes.size();
  GroupNodes.push_back(idx);
  GroupNodeIndices[Reg] = idx;
  return idx;
}

bool SchedulePostRATDList::IsLive(unsigned Reg)
{
  // KillIndex must be defined and DefIndex not defined for a register
  // to be live.
  return((KillIndices[Reg] != ~0u) && (DefIndices[Reg] == ~0u));
}

/// StartBlock - Initialize register live-range state for scheduling in
/// this block.
///
void SchedulePostRATDList::StartBlock(MachineBasicBlock *BB) {
  // Call the superclass.
  ScheduleDAGInstrs::StartBlock(BB);

  // Reset the hazard recognizer.
  HazardRec->Reset();

  // Initialize all registers to be in their own group. Initially we
  // assign the register to the same-indexed GroupNode.
  for (unsigned i = 0; i < TargetRegisterInfo::FirstVirtualRegister; ++i)
    GroupNodeIndices[i] = i;

  // Initialize the indices to indicate that no registers are live.
  std::fill(KillIndices, array_endof(KillIndices), ~0u);
  std::fill(DefIndices, array_endof(DefIndices), BB->size());

  bool IsReturnBlock = (!BB->empty() && BB->back().getDesc().isReturn());

  // Determine the live-out physregs for this block.
  if (IsReturnBlock) {
    // In a return block, examine the function live-out regs.
    for (MachineRegisterInfo::liveout_iterator I = MRI.liveout_begin(),
         E = MRI.liveout_end(); I != E; ++I) {
      unsigned Reg = *I;
      UnionGroups(Reg, 0);
      KillIndices[Reg] = BB->size();
      DefIndices[Reg] = ~0u;
      // Repeat, for all aliases.
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        unsigned AliasReg = *Alias;
        UnionGroups(AliasReg, 0);
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
        UnionGroups(Reg, 0);
        KillIndices[Reg] = BB->size();
        DefIndices[Reg] = ~0u;
        // Repeat, for all aliases.
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          unsigned AliasReg = *Alias;
          UnionGroups(AliasReg, 0);
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
    UnionGroups(Reg, 0);
    KillIndices[Reg] = BB->size();
    DefIndices[Reg] = ~0u;
    // Repeat, for all aliases.
    for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
      unsigned AliasReg = *Alias;
      UnionGroups(AliasReg, 0);
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
  BuildSchedGraph(AA);

  if (EnableAntiDepBreaking != "none") {
    if (BreakAntiDependencies((EnableAntiDepBreaking == "all") ? false : true)) {
      // We made changes. Update the dependency graph.
      // Theoretically we could update the graph in place:
      // When a live range is changed to use a different register, remove
      // the def's anti-dependence *and* output-dependence edges due to
      // that register, and add new anti-dependence and output-dependence
      // edges based on the next live range of the register.
      SUnits.clear();
      EntrySU = SUnit();
      ExitSU = SUnit();
      BuildSchedGraph(AA);
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

  DEBUG(errs() << "Observe: ");
  DEBUG(MI->dump());

  for (unsigned Reg = 0; Reg != TargetRegisterInfo::FirstVirtualRegister; ++Reg) {
    // If Reg is current live, then mark that it can't be renamed as
    // we don't know the extent of its live-range anymore (now that it
    // has been scheduled). If it is not live but was defined in the
    // previous schedule region, then set its def index to the most
    // conservative location (i.e. the beginning of the previous
    // schedule region).
    if (IsLive(Reg)) {
      DEBUG(if (GetGroup(Reg) != 0)
              errs() << " " << TRI->getName(Reg) << "=g" << 
                GetGroup(Reg) << "->g0(region live-out)");
      UnionGroups(Reg, 0);
    } else if ((DefIndices[Reg] < InsertPosIndex) && (DefIndices[Reg] >= Count)) {
      DefIndices[Reg] = Count;
    }
  }

  PrescanInstruction(MI, Count);
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

/// AntiDepPathStep - Return SUnit that SU has an anti-dependence on.
static SDep *AntiDepPathStep(SUnit *SU) {
  for (SUnit::pred_iterator P = SU->Preds.begin(), PE = SU->Preds.end();
       P != PE; ++P) {
    if (P->getKind() == SDep::Anti) {
      return &*P;
    }
  }
  return 0;
}

void SchedulePostRATDList::PrescanInstruction(MachineInstr *MI, unsigned Count) {
  // Scan the register defs for this instruction and update
  // live-ranges, groups and RegRefs.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isDef()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    // Ignore two-addr defs for liveness...
    if (MI->isRegTiedToUseOperand(i)) continue;

    // Update Def for Reg and subregs.
    DefIndices[Reg] = Count;
    for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
         *Subreg; ++Subreg) {
      unsigned SubregReg = *Subreg;
      DefIndices[SubregReg] = Count;
    }
  }

  DEBUG(errs() << "\tGroups:");
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isDef()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;

    DEBUG(errs() << " " << TRI->getName(Reg) << "=g" << GetGroup(Reg)); 

    // If MI's defs have special allocation requirement, don't allow
    // any def registers to be changed. Also assume all registers
    // defined in a call must not be changed (ABI).
    if (MI->getDesc().isCall() || MI->getDesc().hasExtraDefRegAllocReq()) {
      DEBUG(if (GetGroup(Reg) != 0) errs() << "->g0(alloc-req)");
      UnionGroups(Reg, 0);
    }

    // Any subregisters that are live at this point are defined here,
    // so group those subregisters with Reg.
    for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
         *Subreg; ++Subreg) {
      unsigned SubregReg = *Subreg;
      if (IsLive(SubregReg)) {
        UnionGroups(Reg, SubregReg);
        DEBUG(errs() << "->g" << GetGroup(Reg) << "(via " << 
              TRI->getName(SubregReg) << ")");
      }
    }
    
    // Note register reference...
    const TargetRegisterClass *RC = NULL;
    if (i < MI->getDesc().getNumOperands())
      RC = MI->getDesc().OpInfo[i].getRegClass(TRI);
    RegisterReference RR = { &MO, RC };
    RegRefs.insert(std::make_pair(Reg, RR));
  }

  DEBUG(errs() << '\n');
}

void SchedulePostRATDList::ScanInstruction(MachineInstr *MI,
                                           unsigned Count) {
  // Scan the register uses for this instruction and update
  // live-ranges, groups and RegRefs.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    
    // It wasn't previously live but now it is, this is a kill. Forget
    // the previous live-range information and start a new live-range
    // for the register.
    if (!IsLive(Reg)) {
      KillIndices[Reg] = Count;
      DefIndices[Reg] = ~0u;
      RegRefs.erase(Reg);
      LeaveGroup(Reg);
    }
    // Repeat, for subregisters.
    for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
         *Subreg; ++Subreg) {
      unsigned SubregReg = *Subreg;
      if (!IsLive(SubregReg)) {
        KillIndices[SubregReg] = Count;
        DefIndices[SubregReg] = ~0u;
        RegRefs.erase(SubregReg);
        LeaveGroup(SubregReg);
      }
    }

    // Note register reference...
    const TargetRegisterClass *RC = NULL;
    if (i < MI->getDesc().getNumOperands())
      RC = MI->getDesc().OpInfo[i].getRegClass(TRI);
    RegisterReference RR = { &MO, RC };
    RegRefs.insert(std::make_pair(Reg, RR));
  }
  
  // Form a group of all defs and uses of a KILL instruction to ensure
  // that all registers are renamed as a group.
  if (MI->getOpcode() == TargetInstrInfo::KILL) {
    unsigned FirstReg = 0;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;
      
      if (FirstReg != 0)
        UnionGroups(FirstReg, Reg);
      FirstReg = Reg;
    }

    DEBUG(if (FirstReg != 0) errs() << "\tKill Group: g" << 
                               GetGroup(FirstReg) << '\n'); 
  }
}

unsigned SchedulePostRATDList::FindSuitableFreeRegister(unsigned AntiDepReg,
                                                        unsigned LastNewReg) {
  // Collect all registers in the same group as AntiDepReg. These all
  // need to be renamed together if we are to break the
  // anti-dependence.
  std::vector<unsigned> Regs;
  GetGroupRegs(GetGroup(AntiDepReg), Regs);

  DEBUG(errs() << "\tRename Register Group:");
  DEBUG(for (unsigned i = 0, e = Regs.size(); i != e; ++i)
          DEBUG(errs() << " " << TRI->getName(Regs[i])));
  DEBUG(errs() << "\n");

  // If there is a single register that needs to be renamed then we
  // can do it ourselves.
  if (Regs.size() == 1) {
    assert(Regs[0] == AntiDepReg && "Register group does not contain register!");

    // Check all references that need rewriting. Gather up all the
    // register classes for the register references.
    const TargetRegisterClass *FirstRC = NULL;
    std::set<const TargetRegisterClass *> RCs;
    std::pair<std::multimap<unsigned, RegisterReference>::iterator,
      std::multimap<unsigned, RegisterReference>::iterator>
      Range = RegRefs.equal_range(AntiDepReg);
    for (std::multimap<unsigned, RegisterReference>::iterator
           Q = Range.first, QE = Range.second; Q != QE; ++Q) {
      const TargetRegisterClass *RC = Q->second.RC;
      if (RC == NULL) continue;
      if (FirstRC == NULL)
        FirstRC = RC;
      else if (FirstRC != RC)
        RCs.insert(RC);
    }
    
    if (FirstRC == NULL)
      return 0;

    DEBUG(errs() << "\tChecking Regclasses: " << FirstRC->getName());
    DEBUG(for (std::set<const TargetRegisterClass *>::iterator S = 
                 RCs.begin(), E = RCs.end(); S != E; ++S)
            errs() << " " << (*S)->getName());
    DEBUG(errs() << '\n');

    // Using the allocation order for one of the register classes,
    // find the first register that belongs to all the register
    // classes that is available over the liverange of the register.
    DEBUG(errs() << "\tFind Register:");
    for (TargetRegisterClass::iterator R = FirstRC->allocation_order_begin(MF),
           RE = FirstRC->allocation_order_end(MF); R != RE; ++R) {
      unsigned NewReg = *R;
      
      // Don't replace a register with itself.
      if (NewReg == AntiDepReg) continue;

      DEBUG(errs() << " " << TRI->getName(NewReg));
      
      // Make sure NewReg is in all required register classes.
      for (std::set<const TargetRegisterClass *>::iterator S = 
             RCs.begin(), E = RCs.end(); S != E; ++S) {
        const TargetRegisterClass *RC = *S;
        if (!RC->contains(NewReg)) {
          DEBUG(errs() << "(not in " << RC->getName() << ")");
          NewReg = 0;
          break;
        }
      }

      // If NewReg is dead and NewReg's most recent def is not before
      // AntiDepReg's kill, it's safe to replace AntiDepReg with
      // NewReg. We must also check all subregisters of NewReg.
      if (IsLive(NewReg) || (KillIndices[AntiDepReg] > DefIndices[NewReg])) {
        DEBUG(errs() << "(live)");
        continue;
      }
      {
        bool found = false;
        for (const unsigned *Subreg = TRI->getSubRegisters(NewReg);
             *Subreg; ++Subreg) {
          unsigned SubregReg = *Subreg;
          if (IsLive(SubregReg) || (KillIndices[AntiDepReg] > DefIndices[SubregReg])) {
            DEBUG(errs() << "(subreg " << TRI->getName(SubregReg) << " live)");
            found = true;
          }
        }
        if (found)
          continue;
      }
      
      if (NewReg != 0) { 
        DEBUG(errs() << '\n');
        return NewReg;
      }
    }

    DEBUG(errs() << '\n');
  }

  // No registers are free and available!
  return 0;
}

/// BreakAntiDependencies - Identifiy anti-dependencies along the critical path
/// of the ScheduleDAG and break them by renaming registers.
///
bool SchedulePostRATDList::BreakAntiDependencies(bool CriticalPathOnly) {
  // The code below assumes that there is at least one instruction,
  // so just duck out immediately if the block is empty.
  if (SUnits.empty()) return false;

  // If breaking anti-dependencies only along the critical path, track
  // progress along the critical path through the SUnit graph as we
  // walk the instructions.
  SUnit *CriticalPathSU = 0;
  MachineInstr *CriticalPathMI = 0;
  
  // If breaking all anti-dependencies need a map from MI to SUnit.
  std::map<MachineInstr *, SUnit *> MISUnitMap;

  // Find the node at the bottom of the critical path.
  if (CriticalPathOnly) {
    SUnit *Max = 0;
    for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
      SUnit *SU = &SUnits[i];
      if (!Max || SU->getDepth() + SU->Latency > Max->getDepth() + Max->Latency)
        Max = SU;
    }

    DEBUG(errs() << "Critical path has total latency "
          << (Max->getDepth() + Max->Latency) << "\n");
    CriticalPathSU = Max;
    CriticalPathMI = CriticalPathSU->getInstr();
  } else {
    for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
      SUnit *SU = &SUnits[i];
      MISUnitMap.insert(std::pair<MachineInstr *, SUnit *>(SU->getInstr(), SU));
    }
    DEBUG(errs() << "Breaking all anti-dependencies\n");
  }

#ifndef NDEBUG 
  {
    DEBUG(errs() << "Available regs:");
    for (unsigned Reg = 0; Reg < TRI->getNumRegs(); ++Reg) {
      if (!IsLive(Reg))
        DEBUG(errs() << " " << TRI->getName(Reg));
    }
    DEBUG(errs() << '\n');
  }
  std::string dbgStr;
#endif

  // TODO: If we tracked more than one register here, we could potentially
  // fix that remaining critical edge too. This is a little more involved,
  // because unlike the most recent register, less recent registers should
  // still be considered, though only if no other registers are available.
  unsigned LastNewReg[TargetRegisterInfo::FirstVirtualRegister] = {};

  // Attempt to break anti-dependence edges. Walk the instructions
  // from the bottom up, tracking information about liveness as we go
  // to help determine which registers are available.
  bool Changed = false;
  unsigned Count = InsertPosIndex - 1;
  for (MachineBasicBlock::iterator I = InsertPos, E = Begin;
       I != E; --Count) {
    MachineInstr *MI = --I;

    DEBUG(errs() << "Anti: ");
    DEBUG(MI->dump());

    // Process the defs in MI...
    PrescanInstruction(MI, Count);

    // Check if this instruction has an anti-dependence that we may be
    // able to break. If it is, set AntiDepReg to the non-zero
    // register associated with the anti-dependence.
    //
    unsigned AntiDepReg = 0;
  
    // Limiting our attention to the critical path is a heuristic to avoid
    // breaking anti-dependence edges that aren't going to significantly
    // impact the overall schedule. There are a limited number of registers
    // and we want to save them for the important edges.
    // 
    // We can also break all anti-dependencies because they can
    // occur along the non-critical path but are still detrimental for
    // scheduling.
    // 
    // TODO: Instructions with multiple defs could have multiple
    // anti-dependencies. The current code here only knows how to break one
    // edge per instruction. Note that we'd have to be able to break all of
    // the anti-dependencies in an instruction in order to be effective.
    if (!CriticalPathOnly || (MI == CriticalPathMI)) {
      DEBUG(dbgStr.clear());

      SUnit *PathSU;
      SDep *Edge;
      if (CriticalPathOnly) {
        PathSU = CriticalPathSU;
        Edge = CriticalPathStep(PathSU);
      } else {
        PathSU = MISUnitMap[MI];
        Edge = (PathSU) ? AntiDepPathStep(PathSU) : 0;
      }
      
      if (Edge) {
        SUnit *NextSU = Edge->getSUnit();

        // Only consider anti-dependence edges, and ignore KILL
        // instructions (they form a group in ScanInstruction but
        // don't cause any anti-dependence breaking themselves)
        if ((Edge->getKind() == SDep::Anti) &&
            (MI->getOpcode() != TargetInstrInfo::KILL)) {
          AntiDepReg = Edge->getReg();
          DEBUG(dbgStr += "\tAntidep reg: ");
          DEBUG(dbgStr += TRI->getName(AntiDepReg));
          assert(AntiDepReg != 0 && "Anti-dependence on reg0?");
          if (!AllocatableSet.test(AntiDepReg)) {
            // Don't break anti-dependencies on non-allocatable registers.
            DEBUG(dbgStr += " (non-allocatable)");
            AntiDepReg = 0;
          } else {
            int OpIdx = MI->findRegisterDefOperandIdx(AntiDepReg);
            assert(OpIdx != -1 && "Can't find index for defined register operand");
            if (MI->isRegTiedToUseOperand(OpIdx)) {
              // If the anti-dep register is tied to a use, then don't try to
              // change it. It will be changed along with the use if required
              // to break an earlier antidep.
              DEBUG(dbgStr += " (tied-to-use)");
              AntiDepReg = 0;
            } else {
              // If the SUnit has other dependencies on the SUnit that
              // it anti-depends on, don't bother breaking the
              // anti-dependency since those edges would prevent such
              // units from being scheduled past each other
              // regardless.
              //
              // Also, if there are dependencies on other SUnits with
              // the same register as the anti-dependency, don't
              // attempt to break it.
              for (SUnit::pred_iterator P = PathSU->Preds.begin(),
                     PE = PathSU->Preds.end(); P != PE; ++P) {
                if (P->getSUnit() == NextSU ?
                    (P->getKind() != SDep::Anti || P->getReg() != AntiDepReg) :
                    (P->getKind() == SDep::Data && P->getReg() == AntiDepReg)) {
                  DEBUG(dbgStr += " (real dependency)");
                  AntiDepReg = 0;
                  break;
                }
              }
            }
          }
        }
        
        if (CriticalPathOnly) {
          CriticalPathSU = NextSU;
          CriticalPathMI = CriticalPathSU->getInstr();
        }
      } else {
        // We've reached the end of the critical path.
        CriticalPathSU = 0;
        CriticalPathMI = 0;
      }
    }

    // Determine AntiDepReg's register group.
    const unsigned GroupIndex = AntiDepReg != 0 ? GetGroup(AntiDepReg) : 0;
    if (GroupIndex == 0) {
      DEBUG(if (AntiDepReg != 0) dbgStr += " (zero group)");
      AntiDepReg = 0;
    }

    DEBUG(if (!dbgStr.empty()) errs() << dbgStr << '\n');

    // Look for a suitable register to use to break the anti-dependence.
    //
    // TODO: Instead of picking the first free register, consider which might
    // be the best.
    if (AntiDepReg != 0) {
      if (unsigned NewReg = FindSuitableFreeRegister(AntiDepReg,
                                                     LastNewReg[AntiDepReg])) {
        DEBUG(errs() << "\tBreaking anti-dependence edge on "
              << TRI->getName(AntiDepReg)
              << " with " << RegRefs.count(AntiDepReg) << " references"
              << " using " << TRI->getName(NewReg) << "!\n");

        // Update the references to the old register to refer to the new
        // register.
        std::pair<std::multimap<unsigned, RegisterReference>::iterator,
                  std::multimap<unsigned, RegisterReference>::iterator>
           Range = RegRefs.equal_range(AntiDepReg);
        for (std::multimap<unsigned, RegisterReference>::iterator
             Q = Range.first, QE = Range.second; Q != QE; ++Q)
          Q->second.Operand->setReg(NewReg);

        // We just went back in time and modified history; the
        // liveness information for the anti-dependence reg is now
        // inconsistent. Set the state as if it were dead.
        // FIXME forall in group
        UnionGroups(NewReg, 0);
        RegRefs.erase(NewReg);
        DefIndices[NewReg] = DefIndices[AntiDepReg];
        KillIndices[NewReg] = KillIndices[AntiDepReg];

        // FIXME forall in group
        UnionGroups(AntiDepReg, 0);
        RegRefs.erase(AntiDepReg);
        DefIndices[AntiDepReg] = KillIndices[AntiDepReg];
        KillIndices[AntiDepReg] = ~0u;
        assert(((KillIndices[AntiDepReg] == ~0u) !=
                (DefIndices[AntiDepReg] == ~0u)) &&
             "Kill and Def maps aren't consistent for AntiDepReg!");

        Changed = true;
        LastNewReg[AntiDepReg] = NewReg;
        ++NumFixedAnti;
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

  if (AllDead)
    MO.setIsKill(true);
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

FunctionPass *llvm::createPostRAScheduler(CodeGenOpt::Level OptLevel) {
  return new PostRAScheduler(OptLevel);
}
