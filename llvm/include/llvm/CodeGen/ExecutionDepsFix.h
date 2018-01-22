//==- llvm/CodeGen/ExecutionDepsFix.h - Execution Dependency Fix -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file Execution Dependency Fix pass.
///
/// Some X86 SSE instructions like mov, and, or, xor are available in different
/// variants for different operand types. These variant instructions are
/// equivalent, but on Nehalem and newer cpus there is extra latency
/// transferring data between integer and floating point domains.  ARM cores
/// have similar issues when they are configured with both VFP and NEON
/// pipelines.
///
/// This pass changes the variant instructions to minimize domain crossings.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EXECUTIONDEPSFIX_H
#define LLVM_CODEGEN_EXECUTIONDEPSFIX_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/RegisterClassInfo.h"

namespace llvm {

class MachineBasicBlock;
class MachineInstr;
class TargetInstrInfo;

/// A DomainValue is a bit like LiveIntervals' ValNo, but it also keeps track
/// of execution domains.
///
/// An open DomainValue represents a set of instructions that can still switch
/// execution domain. Multiple registers may refer to the same open
/// DomainValue - they will eventually be collapsed to the same execution
/// domain.
///
/// A collapsed DomainValue represents a single register that has been forced
/// into one of more execution domains. There is a separate collapsed
/// DomainValue for each register, but it may contain multiple execution
/// domains. A register value is initially created in a single execution
/// domain, but if we were forced to pay the penalty of a domain crossing, we
/// keep track of the fact that the register is now available in multiple
/// domains.
struct DomainValue {
  /// Basic reference counting.
  unsigned Refs = 0;

  /// Bitmask of available domains. For an open DomainValue, it is the still
  /// possible domains for collapsing. For a collapsed DomainValue it is the
  /// domains where the register is available for free.
  unsigned AvailableDomains;

  /// Pointer to the next DomainValue in a chain.  When two DomainValues are
  /// merged, Victim.Next is set to point to Victor, so old DomainValue
  /// references can be updated by following the chain.
  DomainValue *Next;

  /// Twiddleable instructions using or defining these registers.
  SmallVector<MachineInstr *, 8> Instrs;

  DomainValue() { clear(); }

  /// A collapsed DomainValue has no instructions to twiddle - it simply keeps
  /// track of the domains where the registers are already available.
  bool isCollapsed() const { return Instrs.empty(); }

  /// Is domain available?
  bool hasDomain(unsigned domain) const {
    assert(domain <
               static_cast<unsigned>(std::numeric_limits<unsigned>::digits) &&
           "undefined behavior");
    return AvailableDomains & (1u << domain);
  }

  /// Mark domain as available.
  void addDomain(unsigned domain) { AvailableDomains |= 1u << domain; }

  // Restrict to a single domain available.
  void setSingleDomain(unsigned domain) { AvailableDomains = 1u << domain; }

  /// Return bitmask of domains that are available and in mask.
  unsigned getCommonDomains(unsigned mask) const {
    return AvailableDomains & mask;
  }

  /// First domain available.
  unsigned getFirstDomain() const {
    return countTrailingZeros(AvailableDomains);
  }

  /// Clear this DomainValue and point to next which has all its data.
  void clear() {
    AvailableDomains = 0;
    Next = nullptr;
    Instrs.clear();
  }
};

/// This class provides the basic blocks traversal order used by passes like
/// ReachingDefAnalysis and ExecutionDomainFix.
/// It identifies basic blocks that are part of loops and should to be visited
/// twice and returns efficient traversal order for all the blocks.
///
/// We want to visit every instruction in every basic block in order to update
/// it's execution domain or collect clearance information. However, for the
/// clearance calculation, we need to know clearances from all predecessors
/// (including any backedges), therfore we need to visit some blocks twice.
/// As an example, consider the following loop.
///
///
///    PH -> A -> B (xmm<Undef> -> xmm<Def>) -> C -> D -> EXIT
///          ^                                  |
///          +----------------------------------+
///
/// The iteration order this pass will return is as follows:
/// Optimized: PH A B C A' B' C' D
///
/// The basic block order is constructed as follows:
/// Once we finish processing some block, we update the counters in MBBInfos
/// and re-process any successors that are now 'done'.
/// We call a block that is ready for its final round of processing `done`
/// (isBlockDone), e.g. when all predecessor information is known.
///
/// Note that a naive traversal order would be to do two complete passes over
/// all basic blocks/instructions, the first for recording clearances, the
/// second for updating clearance based on backedges.
/// However, for functions without backedges, or functions with a lot of
/// straight-line code, and a small loop, that would be a lot of unnecessary
/// work (since only the BBs that are part of the loop require two passes).
///
/// E.g., the naive iteration order for the above exmple is as follows:
/// Naive: PH A B C D A' B' C' D'
///
/// In the optimized approach we avoid processing D twice, because we
/// can entirely process the predecessors before getting to D.
class LoopTraversal {
private:
  struct MBBInfo {
    /// Whether we have gotten to this block in primary processing yet.
    bool PrimaryCompleted = false;

    /// The number of predecessors for which primary processing has completed
    unsigned IncomingProcessed = 0;

    /// The value of `IncomingProcessed` at the start of primary processing
    unsigned PrimaryIncoming = 0;

    /// The number of predecessors for which all processing steps are done.
    unsigned IncomingCompleted = 0;

    MBBInfo() = default;
  };
  using MBBInfoMap = SmallVector<MBBInfo, 4>;
  /// Helps keep track if we proccessed this block and all its predecessors.
  MBBInfoMap MBBInfos;

public:
  struct TraversedMBBInfo {
    /// The basic block.
    MachineBasicBlock *MBB = nullptr;

    /// True if this is the first time we process the basic block.
    bool PrimaryPass = true;

    /// True if the block that is ready for its final round of processing.
    bool IsDone = true;

    TraversedMBBInfo(MachineBasicBlock *BB = nullptr, bool Primary = true,
                     bool Done = true)
        : MBB(BB), PrimaryPass(Primary), IsDone(Done) {}
  };
  LoopTraversal() {}

  /// \brief Identifies basic blocks that are part of loops and should to be
  ///  visited twise and returns efficient traversal order for all the blocks.
  typedef SmallVector<TraversedMBBInfo, 4> TraversalOrder;
  TraversalOrder traverse(MachineFunction &MF);

private:
  /// Returens true if the block is ready for its final round of processing.
  bool isBlockDone(MachineBasicBlock *MBB);
};

/// This class provides the reaching def analysis.
class ReachingDefAnalysis : public MachineFunctionPass {
private:
  MachineFunction *MF;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  unsigned NumRegUnits;
  /// Instruction that defined each register, relative to the beginning of the
  /// current basic block.  When a LiveRegsDefInfo is used to represent a
  /// live-out register, this value is relative to the end of the basic block,
  /// so it will be a negative number.
  using LiveRegsDefInfo = std::vector<int>;
  LiveRegsDefInfo LiveRegs;

  /// Keeps clearance information for all registers. Note that this
  /// is different from the usual definition notion of liveness. The CPU
  /// doesn't care whether or not we consider a register killed.
  using OutRegsInfoMap = SmallVector<LiveRegsDefInfo, 4>;
  OutRegsInfoMap MBBOutRegsInfos;

  /// Current instruction number.
  /// The first instruction in each basic block is 0.
  int CurInstr;

  /// Maps instructions to their instruction Ids, relative to the begining of
  /// their basic blocks.
  DenseMap<MachineInstr *, int> InstIds;

  /// All reaching defs of a given RegUnit for a given MBB.
  using MBBRegUnitDefs = SmallVector<int, 1>;
  /// All reaching defs of all reg units for a given MBB
  using MBBDefsInfo = std::vector<MBBRegUnitDefs>;
  /// All reaching defs of all reg units for a all MBBs
  using MBBReachingDefsInfo = SmallVector<MBBDefsInfo, 4>;
  MBBReachingDefsInfo MBBReachingDefs;

  /// Default values are 'nothing happened a long time ago'.
  const int ReachingDedDefaultVal = -(1 << 20);

public:
  static char ID; // Pass identification, replacement for typeid

  ReachingDefAnalysis() : MachineFunctionPass(ID) {
    initializeReachingDefAnalysisPass(*PassRegistry::getPassRegistry());
  }
  void releaseMemory() override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  /// Provides the instruction id of the closest reaching def instruction of
  /// PhysReg that reaches MI, relative to the begining of MI's basic block.
  int getReachingDef(MachineInstr *MI, int PhysReg);

  /// Provides the clearance - the number of instructions since the closest
  /// reaching def instuction of PhysReg that reaches MI.
  int getClearance(MachineInstr *MI, MCPhysReg PhysReg);

private:
  /// Set up LiveRegs by merging predecessor live-out values.
  void enterBasicBlock(const LoopTraversal::TraversedMBBInfo &TraversedMBB);

  /// Update live-out values.
  void leaveBasicBlock(const LoopTraversal::TraversedMBBInfo &TraversedMBB);

  /// Process he given basic block.
  void processBasicBlock(const LoopTraversal::TraversedMBBInfo &TraversedMBB);

  /// Update def-ages for registers defined by MI.
  /// Also break dependencies on partial defs and undef uses.
  void processDefs(MachineInstr *);
};

class ExecutionDomainFix : public MachineFunctionPass {
  SpecificBumpPtrAllocator<DomainValue> Allocator;
  SmallVector<DomainValue *, 16> Avail;

  const TargetRegisterClass *const RC;
  MachineFunction *MF;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  std::vector<SmallVector<int, 1>> AliasMap;
  const unsigned NumRegs;
  /// Value currently in each register, or NULL when no value is being tracked.
  /// This counts as a DomainValue reference.
  using LiveRegsDVInfo = std::vector<DomainValue *>;
  LiveRegsDVInfo LiveRegs;
  /// Keeps domain information for all registers. Note that this
  /// is different from the usual definition notion of liveness. The CPU
  /// doesn't care whether or not we consider a register killed.
  using OutRegsInfoMap = SmallVector<LiveRegsDVInfo, 4>;
  OutRegsInfoMap MBBOutRegsInfos;

  ReachingDefAnalysis *RDA;

public:
  ExecutionDomainFix(char &PassID, const TargetRegisterClass &RC)
      : MachineFunctionPass(PassID), RC(&RC), NumRegs(RC.getNumRegs()) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<ReachingDefAnalysis>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  /// Translate TRI register number to a list of indices into our smaller tables
  /// of interesting registers.
  iterator_range<SmallVectorImpl<int>::const_iterator>
  regIndices(unsigned Reg) const;

  /// DomainValue allocation.
  DomainValue *alloc(int domain = -1);

  /// Add reference to DV.
  DomainValue *retain(DomainValue *DV) {
    if (DV)
      ++DV->Refs;
    return DV;
  }

  /// Release a reference to DV.  When the last reference is released,
  /// collapse if needed.
  void release(DomainValue *);

  /// Follow the chain of dead DomainValues until a live DomainValue is reached.
  /// Update the referenced pointer when necessary.
  DomainValue *resolve(DomainValue *&);

  /// Set LiveRegs[rx] = dv, updating reference counts.
  void setLiveReg(int rx, DomainValue *DV);

  /// Kill register rx, recycle or collapse any DomainValue.
  void kill(int rx);

  /// Force register rx into domain.
  void force(int rx, unsigned domain);

  /// Collapse open DomainValue into given domain. If there are multiple
  /// registers using dv, they each get a unique collapsed DomainValue.
  void collapse(DomainValue *dv, unsigned domain);

  /// All instructions and registers in B are moved to A, and B is released.
  bool merge(DomainValue *A, DomainValue *B);

  /// Set up LiveRegs by merging predecessor live-out values.
  void enterBasicBlock(const LoopTraversal::TraversedMBBInfo &TraversedMBB);

  /// Update live-out values.
  void leaveBasicBlock(const LoopTraversal::TraversedMBBInfo &TraversedMBB);

  /// Process he given basic block.
  void processBasicBlock(const LoopTraversal::TraversedMBBInfo &TraversedMBB);

  /// Visit given insturcion.
  bool visitInstr(MachineInstr *);

  /// Update def-ages for registers defined by MI.
  /// If Kill is set, also kill off DomainValues clobbered by the defs.
  void processDefs(MachineInstr *, bool Kill);

  /// A soft instruction can be changed to work in other domains given by mask.
  void visitSoftInstr(MachineInstr *, unsigned mask);

  /// A hard instruction only works in one domain. All input registers will be
  /// forced into that domain.
  void visitHardInstr(MachineInstr *, unsigned domain);
};

class BreakFalseDeps : public MachineFunctionPass {
private:
  MachineFunction *MF;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  RegisterClassInfo RegClassInfo;

  /// List of undefined register reads in this block in forward order.
  std::vector<std::pair<MachineInstr *, unsigned>> UndefReads;

  /// Storage for register unit liveness.
  LivePhysRegs LiveRegSet;

  ReachingDefAnalysis *RDA;

public:
  static char ID; // Pass identification, replacement for typeid

  BreakFalseDeps() : MachineFunctionPass(ID) {
    initializeBreakFalseDepsPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<ReachingDefAnalysis>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  /// Process he given basic block.
  void processBasicBlock(MachineBasicBlock *MBB);

  /// Update def-ages for registers defined by MI.
  /// Also break dependencies on partial defs and undef uses.
  void processDefs(MachineInstr *MI);

  /// \brief Helps avoid false dependencies on undef registers by updating the
  /// machine instructions' undef operand to use a register that the instruction
  /// is truly dependent on, or use a register with clearance higher than Pref.
  /// Returns true if it was able to find a true dependency, thus not requiring
  /// a dependency breaking instruction regardless of clearance.
  bool pickBestRegisterForUndef(MachineInstr *MI, unsigned OpIdx,
                                unsigned Pref);

  /// \brief Return true to if it makes sense to break dependence on a partial
  /// def or undef use.
  bool shouldBreakDependence(MachineInstr *, unsigned OpIdx, unsigned Pref);

  /// \brief Break false dependencies on undefined register reads.
  /// Walk the block backward computing precise liveness. This is expensive, so
  /// we only do it on demand. Note that the occurrence of undefined register
  /// reads that should be broken is very rare, but when they occur we may have
  /// many in a single block.
  void processUndefReads(MachineBasicBlock *);
};

} // namespace llvm

#endif // LLVM_CODEGEN_EXECUTIONDEPSFIX_H
