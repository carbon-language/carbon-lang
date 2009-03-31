//===-- llvm/CodeGen/Spiller.h - Spiller -*- C++ -*------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SPILLER_H
#define LLVM_CODEGEN_SPILLER_H

#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Streams.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "VirtRegMap.h"
#include <map>

namespace llvm {
  
  /// Spiller interface: Implementations of this interface assign spilled
  /// virtual registers to stack slots, rewriting the code.
  struct Spiller {
    virtual ~Spiller();
    virtual bool runOnMachineFunction(MachineFunction &MF,
                                      VirtRegMap &VRM) = 0;
  };

  /// createSpiller - Create an return a spiller object, as specified on the
  /// command line.
  Spiller* createSpiller();
  
  // ************************************************************************ //
  
  // Simple Spiller Implementation
  struct VISIBILITY_HIDDEN SimpleSpiller : public Spiller {
    bool runOnMachineFunction(MachineFunction& mf, VirtRegMap &VRM);
  };
  
  // ************************************************************************ //
  
  /// AvailableSpills - As the local spiller is scanning and rewriting an MBB
  /// from top down, keep track of which spills slots or remat are available in
  /// each register.
  ///
  /// Note that not all physregs are created equal here.  In particular, some
  /// physregs are reloads that we are allowed to clobber or ignore at any time.
  /// Other physregs are values that the register allocated program is using
  /// that we cannot CHANGE, but we can read if we like.  We keep track of this
  /// on a per-stack-slot / remat id basis as the low bit in the value of the
  /// SpillSlotsAvailable entries.  The predicate 'canClobberPhysReg()' checks
  /// this bit and addAvailable sets it if.
  class VISIBILITY_HIDDEN AvailableSpills {
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;

    // SpillSlotsOrReMatsAvailable - This map keeps track of all of the spilled
    // or remat'ed virtual register values that are still available, due to
    // being loaded or stored to, but not invalidated yet.
    std::map<int, unsigned> SpillSlotsOrReMatsAvailable;

    // PhysRegsAvailable - This is the inverse of SpillSlotsOrReMatsAvailable,
    // indicating which stack slot values are currently held by a physreg.  This
    // is used to invalidate entries in SpillSlotsOrReMatsAvailable when a
    // physreg is modified.
    std::multimap<unsigned, int> PhysRegsAvailable;

    void disallowClobberPhysRegOnly(unsigned PhysReg);

    void ClobberPhysRegOnly(unsigned PhysReg);
  public:
    AvailableSpills(const TargetRegisterInfo *tri, const TargetInstrInfo *tii)
      : TRI(tri), TII(tii) {
    }

    /// clear - Reset the state.
    void clear() {
      SpillSlotsOrReMatsAvailable.clear();
      PhysRegsAvailable.clear();
    }

    const TargetRegisterInfo *getRegInfo() const { return TRI; }

    /// getSpillSlotOrReMatPhysReg - If the specified stack slot or remat is
    /// available in a  physical register, return that PhysReg, otherwise
    /// return 0.
    unsigned getSpillSlotOrReMatPhysReg(int Slot) const {
      std::map<int, unsigned>::const_iterator I =
        SpillSlotsOrReMatsAvailable.find(Slot);
      if (I != SpillSlotsOrReMatsAvailable.end()) {
        return I->second >> 1;  // Remove the CanClobber bit.
      }
      return 0;
    }

    /// addAvailable - Mark that the specified stack slot / remat is available
    /// in the specified physreg.  If CanClobber is true, the physreg can be
    /// modified at any time without changing the semantics of the program.
    void addAvailable(int SlotOrReMat, unsigned Reg, bool CanClobber = true) {
      // If this stack slot is thought to be available in some other physreg, 
      // remove its record.
      ModifyStackSlotOrReMat(SlotOrReMat);

      PhysRegsAvailable.insert(std::make_pair(Reg, SlotOrReMat));
      SpillSlotsOrReMatsAvailable[SlotOrReMat]= (Reg << 1) |
                                                (unsigned)CanClobber;

      if (SlotOrReMat > VirtRegMap::MAX_STACK_SLOT)
        DOUT << "Remembering RM#" << SlotOrReMat-VirtRegMap::MAX_STACK_SLOT-1;
      else
        DOUT << "Remembering SS#" << SlotOrReMat;
      DOUT << " in physreg " << TRI->getName(Reg) << "\n";
    }

    /// canClobberPhysRegForSS - Return true if the spiller is allowed to change
    /// the value of the specified stackslot register if it desires. The
    /// specified stack slot must be available in a physreg for this query to
    /// make sense.
    bool canClobberPhysRegForSS(int SlotOrReMat) const {
      assert(SpillSlotsOrReMatsAvailable.count(SlotOrReMat) &&
             "Value not available!");
      return SpillSlotsOrReMatsAvailable.find(SlotOrReMat)->second & 1;
    }

    /// canClobberPhysReg - Return true if the spiller is allowed to clobber the
    /// physical register where values for some stack slot(s) might be
    /// available.
    bool canClobberPhysReg(unsigned PhysReg) const {
      std::multimap<unsigned, int>::const_iterator I =
        PhysRegsAvailable.lower_bound(PhysReg);
      while (I != PhysRegsAvailable.end() && I->first == PhysReg) {
        int SlotOrReMat = I->second;
        I++;
        if (!canClobberPhysRegForSS(SlotOrReMat))
          return false;
      }
      return true;
    }

    /// disallowClobberPhysReg - Unset the CanClobber bit of the specified
    /// stackslot register. The register is still available but is no longer
    /// allowed to be modifed.
    void disallowClobberPhysReg(unsigned PhysReg);

    /// ClobberPhysReg - This is called when the specified physreg changes
    /// value.  We use this to invalidate any info about stuff that lives in
    /// it and any of its aliases.
    void ClobberPhysReg(unsigned PhysReg);

    /// ModifyStackSlotOrReMat - This method is called when the value in a stack
    /// slot changes.  This removes information about which register the
    /// previous value for this slot lives in (as the previous value is dead
    /// now).
    void ModifyStackSlotOrReMat(int SlotOrReMat);

    /// AddAvailableRegsToLiveIn - Availability information is being kept coming
    /// into the specified MBB. Add available physical registers as potential
    /// live-in's. If they are reused in the MBB, they will be added to the
    /// live-in set to make register scavenger and post-allocation scheduler.
    void AddAvailableRegsToLiveIn(MachineBasicBlock &MBB, BitVector &RegKills,
                                  std::vector<MachineOperand*> &KillOps);
  };
  
  // ************************************************************************ //
  
  // ReusedOp - For each reused operand, we keep track of a bit of information,
  // in case we need to rollback upon processing a new operand.  See comments
  // below.
  struct ReusedOp {
    // The MachineInstr operand that reused an available value.
    unsigned Operand;

    // StackSlotOrReMat - The spill slot or remat id of the value being reused.
    unsigned StackSlotOrReMat;

    // PhysRegReused - The physical register the value was available in.
    unsigned PhysRegReused;

    // AssignedPhysReg - The physreg that was assigned for use by the reload.
    unsigned AssignedPhysReg;
    
    // VirtReg - The virtual register itself.
    unsigned VirtReg;

    ReusedOp(unsigned o, unsigned ss, unsigned prr, unsigned apr,
             unsigned vreg)
      : Operand(o), StackSlotOrReMat(ss), PhysRegReused(prr),
        AssignedPhysReg(apr), VirtReg(vreg) {}
  };
  
  /// ReuseInfo - This maintains a collection of ReuseOp's for each operand that
  /// is reused instead of reloaded.
  class VISIBILITY_HIDDEN ReuseInfo {
    MachineInstr &MI;
    std::vector<ReusedOp> Reuses;
    BitVector PhysRegsClobbered;
  public:
    ReuseInfo(MachineInstr &mi, const TargetRegisterInfo *tri) : MI(mi) {
      PhysRegsClobbered.resize(tri->getNumRegs());
    }
    
    bool hasReuses() const {
      return !Reuses.empty();
    }
    
    /// addReuse - If we choose to reuse a virtual register that is already
    /// available instead of reloading it, remember that we did so.
    void addReuse(unsigned OpNo, unsigned StackSlotOrReMat,
                  unsigned PhysRegReused, unsigned AssignedPhysReg,
                  unsigned VirtReg) {
      // If the reload is to the assigned register anyway, no undo will be
      // required.
      if (PhysRegReused == AssignedPhysReg) return;
      
      // Otherwise, remember this.
      Reuses.push_back(ReusedOp(OpNo, StackSlotOrReMat, PhysRegReused, 
                                AssignedPhysReg, VirtReg));
    }

    void markClobbered(unsigned PhysReg) {
      PhysRegsClobbered.set(PhysReg);
    }

    bool isClobbered(unsigned PhysReg) const {
      return PhysRegsClobbered.test(PhysReg);
    }
    
    /// GetRegForReload - We are about to emit a reload into PhysReg.  If there
    /// is some other operand that is using the specified register, either pick
    /// a new register to use, or evict the previous reload and use this reg. 
    unsigned GetRegForReload(unsigned PhysReg, MachineInstr *MI,
                             AvailableSpills &Spills,
                             std::vector<MachineInstr*> &MaybeDeadStores,
                             SmallSet<unsigned, 8> &Rejected,
                             BitVector &RegKills,
                             std::vector<MachineOperand*> &KillOps,
                             VirtRegMap &VRM);

    /// GetRegForReload - Helper for the above GetRegForReload(). Add a
    /// 'Rejected' set to remember which registers have been considered and
    /// rejected for the reload. This avoids infinite looping in case like
    /// this:
    /// t1 := op t2, t3
    /// t2 <- assigned r0 for use by the reload but ended up reuse r1
    /// t3 <- assigned r1 for use by the reload but ended up reuse r0
    /// t1 <- desires r1
    ///       sees r1 is taken by t2, tries t2's reload register r0
    ///       sees r0 is taken by t3, tries t3's reload register r1
    ///       sees r1 is taken by t2, tries t2's reload register r0 ...
    unsigned GetRegForReload(unsigned PhysReg, MachineInstr *MI,
                             AvailableSpills &Spills,
                             std::vector<MachineInstr*> &MaybeDeadStores,
                             BitVector &RegKills,
                             std::vector<MachineOperand*> &KillOps,
                             VirtRegMap &VRM) {
      SmallSet<unsigned, 8> Rejected;
      return GetRegForReload(PhysReg, MI, Spills, MaybeDeadStores, Rejected,
                             RegKills, KillOps, VRM);
    }
  };
  
  // ************************************************************************ //
  
  /// LocalSpiller - This spiller does a simple pass over the machine basic
  /// block to attempt to keep spills in registers as much as possible for
  /// blocks that have low register pressure (the vreg may be spilled due to
  /// register pressure in other blocks).
  class VISIBILITY_HIDDEN LocalSpiller : public Spiller {
    MachineRegisterInfo *RegInfo;
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;
    DenseMap<MachineInstr*, unsigned> DistanceMap;
    std::vector<MachineInstr*> AddedSpills;
  public:
    bool runOnMachineFunction(MachineFunction &MF, VirtRegMap &VRM);
  private:
    void TransferDeadness(MachineBasicBlock *MBB, unsigned CurDist,
                          unsigned Reg, BitVector &RegKills,
                          std::vector<MachineOperand*> &KillOps);
    bool PrepForUnfoldOpti(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &MII,
                           std::vector<MachineInstr*> &MaybeDeadStores,
                           AvailableSpills &Spills, BitVector &RegKills,
                           std::vector<MachineOperand*> &KillOps,
                           VirtRegMap &VRM);
    bool CommuteToFoldReload(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator &MII,
                             unsigned VirtReg, unsigned SrcReg, int SS,
                             AvailableSpills &Spills,
                             BitVector &RegKills,
                             std::vector<MachineOperand*> &KillOps,
                             const TargetRegisterInfo *TRI,
                             VirtRegMap &VRM);
    void RemoveDeadStore(MachineInstr *Store,
                         MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator &MII,
                         SmallSet<MachineInstr*, 4> &ReMatDefs,
                         BitVector &RegKills,
                         std::vector<MachineOperand*> &KillOps,
                         VirtRegMap &VRM);

    void SpillRegToStackSlot(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator &MII,
                             int Idx, unsigned PhysReg, int StackSlot,
                             const TargetRegisterClass *RC,
                             bool isAvailable, MachineInstr *&LastStore,
                             AvailableSpills &Spills,
                             SmallSet<MachineInstr*, 4> &ReMatDefs,
                             BitVector &RegKills,
                             std::vector<MachineOperand*> &KillOps,
                             VirtRegMap &VRM);
    void RewriteMBB(MachineBasicBlock &MBB, VirtRegMap &VRM,
                    AvailableSpills &Spills,
                    BitVector &RegKills, std::vector<MachineOperand*> &KillOps);
  };
}

#endif
