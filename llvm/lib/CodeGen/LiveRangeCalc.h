//===---- LiveRangeCalc.h - Calculate live ranges ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The LiveRangeCalc class can be used to compute live ranges from scratch.  It
// caches information about values in the CFG to speed up repeated operations
// on the same live range.  The cache can be shared by non-overlapping live
// ranges.  SplitKit uses that when computing the live range of split products.
//
// A low-level interface is available to clients that know where a variable is
// live, but don't know which value it has as every point.  LiveRangeCalc will
// propagate values down the dominator tree, and even insert PHI-defs where
// needed.  SplitKit uses this faster interface when possible.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVERANGECALC_H
#define LLVM_CODEGEN_LIVERANGECALC_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/CodeGen/LiveInterval.h"

namespace llvm {

/// Forward declarations for MachineDominators.h:
class MachineDominatorTree;
template <class NodeT> class DomTreeNodeBase;
typedef DomTreeNodeBase<MachineBasicBlock> MachineDomTreeNode;

class LiveRangeCalc {
  const MachineFunction *MF;
  const MachineRegisterInfo *MRI;
  SlotIndexes *Indexes;
  MachineDominatorTree *DomTree;
  VNInfo::Allocator *Alloc;

  /// Seen - Bit vector of active entries in LiveOut, also used as a visited
  /// set by findReachingDefs.  One entry per basic block, indexed by block
  /// number.  This is kept as a separate bit vector because it can be cleared
  /// quickly when switching live ranges.
  BitVector Seen;

  /// LiveOutPair - A value and the block that defined it.  The domtree node is
  /// redundant, it can be computed as: MDT[Indexes.getMBBFromIndex(VNI->def)].
  typedef std::pair<VNInfo*, MachineDomTreeNode*> LiveOutPair;

  /// LiveOutMap - Map basic blocks to the value leaving the block.
  typedef IndexedMap<LiveOutPair, MBB2NumberFunctor> LiveOutMap;

  /// LiveOut - Map each basic block where a live range is live out to the
  /// live-out value and its defining block.
  ///
  /// For every basic block, MBB, one of these conditions shall be true:
  ///
  ///  1. !Seen.count(MBB->getNumber())
  ///     Blocks without a Seen bit are ignored.
  ///  2. LiveOut[MBB].second.getNode() == MBB
  ///     The live-out value is defined in MBB.
  ///  3. forall P in preds(MBB): LiveOut[P] == LiveOut[MBB]
  ///     The live-out value passses through MBB. All predecessors must carry
  ///     the same value.
  ///
  /// The domtree node may be null, it can be computed.
  ///
  /// The map can be shared by multiple live ranges as long as no two are
  /// live-out of the same block.
  LiveOutMap LiveOut;

  /// LiveInBlock - Information about a basic block where a live range is known
  /// to be live-in, but the value has not yet been determined.
  struct LiveInBlock {
    // The live range set that is live-in to this block.  The algorithms can
    // handle multiple non-overlapping live ranges simultaneously.
    LiveRange &LR;

    // DomNode - Dominator tree node for the block.
    // Cleared when the final value has been determined and LI has been updated.
    MachineDomTreeNode *DomNode;

    // Position in block where the live-in range ends, or SlotIndex() if the
    // range passes through the block.  When the final value has been
    // determined, the range from the block start to Kill will be added to LI.
    SlotIndex Kill;

    // Live-in value filled in by updateSSA once it is known.
    VNInfo *Value;

    LiveInBlock(LiveRange &LR, MachineDomTreeNode *node, SlotIndex kill)
      : LR(LR), DomNode(node), Kill(kill), Value(nullptr) {}
  };

  /// LiveIn - Work list of blocks where the live-in value has yet to be
  /// determined.  This list is typically computed by findReachingDefs() and
  /// used as a work list by updateSSA().  The low-level interface may also be
  /// used to add entries directly.
  SmallVector<LiveInBlock, 16> LiveIn;

  /// Assuming that LI is live-in to KillMBB and killed at Kill, find the set
  /// of defs that can reach it.
  ///
  /// If only one def can reach Kill, all paths from the def to kill are added
  /// to LI, and the function returns true.
  ///
  /// If multiple values can reach Kill, the blocks that need LI to be live in
  /// are added to the LiveIn array, and the function returns false.
  ///
  /// PhysReg, when set, is used to verify live-in lists on basic blocks.
  bool findReachingDefs(LiveRange &LR, MachineBasicBlock &KillMBB,
                        SlotIndex Kill, unsigned PhysReg);

  /// updateSSA - Compute the values that will be live in to all requested
  /// blocks in LiveIn.  Create PHI-def values as required to preserve SSA form.
  ///
  /// Every live-in block must be jointly dominated by the added live-out
  /// blocks.  No values are read from the live ranges.
  void updateSSA();

  /// Add liveness as specified in the LiveIn vector.
  void updateLiveIns();

public:
  LiveRangeCalc() : MF(nullptr), MRI(nullptr), Indexes(nullptr),
                    DomTree(nullptr), Alloc(nullptr) {}

  //===--------------------------------------------------------------------===//
  // High-level interface.
  //===--------------------------------------------------------------------===//
  //
  // Calculate live ranges from scratch.
  //

  /// reset - Prepare caches for a new set of non-overlapping live ranges.  The
  /// caches must be reset before attempting calculations with a live range
  /// that may overlap a previously computed live range, and before the first
  /// live range in a function.  If live ranges are not known to be
  /// non-overlapping, call reset before each.
  void reset(const MachineFunction *MF,
             SlotIndexes*,
             MachineDominatorTree*,
             VNInfo::Allocator*);

  //===--------------------------------------------------------------------===//
  // Mid-level interface.
  //===--------------------------------------------------------------------===//
  //
  // Modify existing live ranges.
  //

  /// extend - Extend the live range of LI to reach Kill.
  ///
  /// The existing values in LI must be live so they jointly dominate Kill.  If
  /// Kill is not dominated by a single existing value, PHI-defs are inserted
  /// as required to preserve SSA form.  If Kill is known to be dominated by a
  /// single existing value, Alloc may be null.
  ///
  /// PhysReg, when set, is used to verify live-in lists on basic blocks.
  void extend(LiveRange &LR, SlotIndex Kill, unsigned PhysReg = 0);

  /// createDeadDefs - Create a dead def in LI for every def operand of Reg.
  /// Each instruction defining Reg gets a new VNInfo with a corresponding
  /// minimal live range.
  void createDeadDefs(LiveRange &LR, unsigned Reg);

  /// createDeadDefs - Create a dead def in LI for every def of LI->reg.
  void createDeadDefs(LiveInterval &LI) {
    createDeadDefs(LI, LI.reg);
  }

  /// extendToUses - Extend the live range of LI to reach all uses of Reg.
  ///
  /// All uses must be jointly dominated by existing liveness.  PHI-defs are
  /// inserted as needed to preserve SSA form.
  void extendToUses(LiveRange &LR, unsigned Reg);

  /// extendToUses - Extend the live range of LI to reach all uses of LI->reg.
  void extendToUses(LiveInterval &LI) {
    extendToUses(LI, LI.reg);
  }

  //===--------------------------------------------------------------------===//
  // Low-level interface.
  //===--------------------------------------------------------------------===//
  //
  // These functions can be used to compute live ranges where the live-in and
  // live-out blocks are already known, but the SSA value in each block is
  // unknown.
  //
  // After calling reset(), add known live-out values and known live-in blocks.
  // Then call calculateValues() to compute the actual value that is
  // live-in to each block, and add liveness to the live ranges.
  //

  /// setLiveOutValue - Indicate that VNI is live out from MBB.  The
  /// calculateValues() function will not add liveness for MBB, the caller
  /// should take care of that.
  ///
  /// VNI may be null only if MBB is a live-through block also passed to
  /// addLiveInBlock().
  void setLiveOutValue(MachineBasicBlock *MBB, VNInfo *VNI) {
    Seen.set(MBB->getNumber());
    LiveOut[MBB] = LiveOutPair(VNI, nullptr);
  }

  /// addLiveInBlock - Add a block with an unknown live-in value.  This
  /// function can only be called once per basic block.  Once the live-in value
  /// has been determined, calculateValues() will add liveness to LI.
  ///
  /// @param LR      The live range that is live-in to the block.
  /// @param DomNode The domtree node for the block.
  /// @param Kill    Index in block where LI is killed.  If the value is
  ///                live-through, set Kill = SLotIndex() and also call
  ///                setLiveOutValue(MBB, 0).
  void addLiveInBlock(LiveRange &LR,
                      MachineDomTreeNode *DomNode,
                      SlotIndex Kill = SlotIndex()) {
    LiveIn.push_back(LiveInBlock(LR, DomNode, Kill));
  }

  /// calculateValues - Calculate the value that will be live-in to each block
  /// added with addLiveInBlock.  Add PHI-def values as needed to preserve SSA
  /// form.  Add liveness to all live-in blocks up to the Kill point, or the
  /// whole block for live-through blocks.
  ///
  /// Every predecessor of a live-in block must have been given a value with
  /// setLiveOutValue, the value may be null for live-trough blocks.
  void calculateValues();
};

} // end namespace llvm

#endif
