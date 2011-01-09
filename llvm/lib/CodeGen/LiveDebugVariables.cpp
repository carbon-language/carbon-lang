//===- LiveDebugVariables.cpp - Tracking debug info variables -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveDebugVariables analysis.
//
// Remove all DBG_VALUE instructions referencing virtual registers and replace
// them with a data structure tracking where live user variables are kept - in a
// virtual register or in a stack slot.
//
// Allow the data structure to be updated during register allocation when values
// are moved between registers and stack slots. Finally emit new DBG_VALUE
// instructions after register allocation is complete.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "livedebug"
#include "LiveDebugVariables.h"
#include "VirtRegMap.h"
#include "llvm/Constants.h"
#include "llvm/Metadata.h"
#include "llvm/Value.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

static cl::opt<bool>
EnableLDV("live-debug-variables",
          cl::desc("Enable the live debug variables pass"), cl::Hidden);

char LiveDebugVariables::ID = 0;

INITIALIZE_PASS_BEGIN(LiveDebugVariables, "livedebugvars",
                "Debug Variable Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(LiveDebugVariables, "livedebugvars",
                "Debug Variable Analysis", false, false)

void LiveDebugVariables::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTree>();
  AU.addRequiredTransitive<LiveIntervals>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

LiveDebugVariables::LiveDebugVariables() : MachineFunctionPass(ID), pImpl(0) {
  initializeLiveDebugVariablesPass(*PassRegistry::getPassRegistry());
}

/// LocMap - Map of where a user value is live, and its location.
typedef IntervalMap<SlotIndex, unsigned, 4> LocMap;

/// UserValue - A user value is a part of a debug info user variable.
///
/// A DBG_VALUE instruction notes that (a sub-register of) a virtual register
/// holds part of a user variable. The part is identified by a byte offset.
///
/// UserValues are grouped into equivalence classes for easier searching. Two
/// user values are related if they refer to the same variable, or if they are
/// held by the same virtual register. The equivalence class is the transitive
/// closure of that relation.
namespace {
class UserValue {
  const MDNode *variable; ///< The debug info variable we are part of.
  unsigned offset;        ///< Byte offset into variable.

  UserValue *leader;      ///< Equivalence class leader.
  UserValue *next;        ///< Next value in equivalence class, or null.

  /// Numbered locations referenced by locmap.
  SmallVector<MachineOperand, 4> locations;

  /// Map of slot indices where this value is live.
  LocMap locInts;

  /// coalesceLocation - After LocNo was changed, check if it has become
  /// identical to another location, and coalesce them. This may cause LocNo or
  /// a later location to be erased, but no earlier location will be erased.
  void coalesceLocation(unsigned LocNo);

  /// insertDebugValue - Insert a DBG_VALUE into MBB at Idx for LocNo.
  void insertDebugValue(MachineBasicBlock *MBB, SlotIndex Idx, unsigned LocNo,
                        LiveIntervals &LIS, const TargetInstrInfo &TII);

  /// insertDebugKill - Insert an undef DBG_VALUE into MBB at Idx.
  void insertDebugKill(MachineBasicBlock *MBB, SlotIndex Idx,
                       LiveIntervals &LIS, const TargetInstrInfo &TII);

public:
  /// UserValue - Create a new UserValue.
  UserValue(const MDNode *var, unsigned o, LocMap::Allocator &alloc)
    : variable(var), offset(o), leader(this), next(0), locInts(alloc)
  {}

  /// getLeader - Get the leader of this value's equivalence class.
  UserValue *getLeader() {
    UserValue *l = leader;
    while (l != l->leader)
      l = l->leader;
    return leader = l;
  }

  /// getNext - Return the next UserValue in the equivalence class.
  UserValue *getNext() const { return next; }

  /// match - Does this UserValue match the aprameters?
  bool match(const MDNode *Var, unsigned Offset) const {
    return Var == variable && Offset == offset;
  }

  /// merge - Merge equivalence classes.
  static UserValue *merge(UserValue *L1, UserValue *L2) {
    L2 = L2->getLeader();
    if (!L1)
      return L2;
    L1 = L1->getLeader();
    if (L1 == L2)
      return L1;
    // Splice L2 before L1's members.
    UserValue *End = L2;
    while (End->next)
      End->leader = L1, End = End->next;
    End->leader = L1;
    End->next = L1->next;
    L1->next = L2;
    return L1;
  }

  /// getLocationNo - Return the location number that matches Loc.
  unsigned getLocationNo(const MachineOperand &LocMO) {
    if (LocMO.isReg() && LocMO.getReg() == 0)
      return ~0u;
    for (unsigned i = 0, e = locations.size(); i != e; ++i)
      if (LocMO.isIdenticalTo(locations[i]))
        return i;
    locations.push_back(LocMO);
    // We are storing a MachineOperand outside a MachineInstr.
    locations.back().clearParent();
    return locations.size() - 1;
  }

  /// addDef - Add a definition point to this value.
  void addDef(SlotIndex Idx, const MachineOperand &LocMO) {
    // Add a singular (Idx,Idx) -> Loc mapping.
    LocMap::iterator I = locInts.find(Idx);
    if (!I.valid() || I.start() != Idx)
      I.insert(Idx, Idx.getNextSlot(), getLocationNo(LocMO));
  }

  /// extendDef - Extend the current definition as far as possible down the
  /// dominator tree. Stop when meeting an existing def or when leaving the live
  /// range of VNI.
  /// @param Idx   Starting point for the definition.
  /// @param LocNo Location number to propagate.
  /// @param LI    Restrict liveness to where LI has the value VNI. May be null.
  /// @param VNI   When LI is not null, this is the value to restrict to.
  /// @param LIS   Live intervals analysis.
  /// @param MDT   Dominator tree.
  void extendDef(SlotIndex Idx, unsigned LocNo,
                 LiveInterval *LI, const VNInfo *VNI,
                 LiveIntervals &LIS, MachineDominatorTree &MDT);

  /// computeIntervals - Compute the live intervals of all locations after
  /// collecting all their def points.
  void computeIntervals(LiveIntervals &LIS, MachineDominatorTree &MDT);

  /// renameRegister - Update locations to rewrite OldReg as NewReg:SubIdx.
  void renameRegister(unsigned OldReg, unsigned NewReg, unsigned SubIdx,
                      const TargetRegisterInfo *TRI);

  /// rewriteLocations - Rewrite virtual register locations according to the
  /// provided virtual register map.
  void rewriteLocations(VirtRegMap &VRM, const TargetRegisterInfo &TRI);

  /// emitDebugVariables - Recreate DBG_VALUE instruction from data structures.
  void emitDebugValues(VirtRegMap *VRM,
                       LiveIntervals &LIS, const TargetInstrInfo &TRI);

  void print(raw_ostream&, const TargetRegisterInfo*);
};
} // namespace

/// LDVImpl - Implementation of the LiveDebugVariables pass.
namespace {
class LDVImpl {
  LiveDebugVariables &pass;
  LocMap::Allocator allocator;
  MachineFunction *MF;
  LiveIntervals *LIS;
  MachineDominatorTree *MDT;
  const TargetRegisterInfo *TRI;

  /// userValues - All allocated UserValue instances.
  SmallVector<UserValue*, 8> userValues;

  /// Map virtual register to eq class leader.
  typedef DenseMap<unsigned, UserValue*> VRMap;
  VRMap virtRegToEqClass;

  /// Map user variable to eq class leader.
  typedef DenseMap<const MDNode *, UserValue*> UVMap;
  UVMap userVarMap;

  /// getUserValue - Find or create a UserValue.
  UserValue *getUserValue(const MDNode *Var, unsigned Offset);

  /// lookupVirtReg - Find the EC leader for VirtReg or null.
  UserValue *lookupVirtReg(unsigned VirtReg);

  /// mapVirtReg - Map virtual register to an equivalence class.
  void mapVirtReg(unsigned VirtReg, UserValue *EC);

  /// handleDebugValue - Add DBG_VALUE instruction to our maps.
  /// @param MI  DBG_VALUE instruction
  /// @param Idx Last valid SLotIndex before instruction.
  /// @return    True if the DBG_VALUE instruction should be deleted.
  bool handleDebugValue(MachineInstr *MI, SlotIndex Idx);

  /// collectDebugValues - Collect and erase all DBG_VALUE instructions, adding
  /// a UserValue def for each instruction.
  /// @param mf MachineFunction to be scanned.
  /// @return True if any debug values were found.
  bool collectDebugValues(MachineFunction &mf);

  /// computeIntervals - Compute the live intervals of all user values after
  /// collecting all their def points.
  void computeIntervals();

public:
  LDVImpl(LiveDebugVariables *ps) : pass(*ps) {}
  bool runOnMachineFunction(MachineFunction &mf);

  /// clear - Relase all memory.
  void clear() {
    DeleteContainerPointers(userValues);
    userValues.clear();
    virtRegToEqClass.clear();
    userVarMap.clear();
  }

  /// renameRegister - Replace all references to OldReg wiht NewReg:SubIdx.
  void renameRegister(unsigned OldReg, unsigned NewReg, unsigned SubIdx);

  /// emitDebugVariables - Recreate DBG_VALUE instruction from data structures.
  void emitDebugValues(VirtRegMap *VRM);

  void print(raw_ostream&);
};
} // namespace

void UserValue::print(raw_ostream &OS, const TargetRegisterInfo *TRI) {
  if (const MDString *MDS = dyn_cast<MDString>(variable->getOperand(2)))
    OS << "!\"" << MDS->getString() << "\"\t";
  if (offset)
    OS << '+' << offset;
  for (LocMap::const_iterator I = locInts.begin(); I.valid(); ++I) {
    OS << " [" << I.start() << ';' << I.stop() << "):";
    if (I.value() == ~0u)
      OS << "undef";
    else
      OS << I.value();
  }
  for (unsigned i = 0, e = locations.size(); i != e; ++i)
    OS << " Loc" << i << '=' << locations[i];
  OS << '\n';
}

void LDVImpl::print(raw_ostream &OS) {
  OS << "********** DEBUG VARIABLES **********\n";
  for (unsigned i = 0, e = userValues.size(); i != e; ++i)
    userValues[i]->print(OS, TRI);
}

void UserValue::coalesceLocation(unsigned LocNo) {
  unsigned KeepLoc = 0;
  for (unsigned e = locations.size(); KeepLoc != e; ++KeepLoc) {
    if (KeepLoc == LocNo)
      continue;
    if (locations[KeepLoc].isIdenticalTo(locations[LocNo]))
      break;
  }
  // No matches.
  if (KeepLoc == locations.size())
    return;

  // Keep the smaller location, erase the larger one.
  unsigned EraseLoc = LocNo;
  if (KeepLoc > EraseLoc)
    std::swap(KeepLoc, EraseLoc);
  locations.erase(locations.begin() + EraseLoc);

  // Rewrite values.
  for (LocMap::iterator I = locInts.begin(); I.valid(); ++I) {
    unsigned v = I.value();
    if (v == EraseLoc)
      I.setValue(KeepLoc);      // Coalesce when possible.
    else if (v > EraseLoc)
      I.setValueUnchecked(v-1); // Avoid coalescing with untransformed values.
  }
}

UserValue *LDVImpl::getUserValue(const MDNode *Var, unsigned Offset) {
  UserValue *&Leader = userVarMap[Var];
  if (Leader) {
    UserValue *UV = Leader->getLeader();
    Leader = UV;
    for (; UV; UV = UV->getNext())
      if (UV->match(Var, Offset))
        return UV;
  }

  UserValue *UV = new UserValue(Var, Offset, allocator);
  userValues.push_back(UV);
  Leader = UserValue::merge(Leader, UV);
  return UV;
}

void LDVImpl::mapVirtReg(unsigned VirtReg, UserValue *EC) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) && "Only map VirtRegs");
  UserValue *&Leader = virtRegToEqClass[VirtReg];
  Leader = UserValue::merge(Leader, EC);
}

UserValue *LDVImpl::lookupVirtReg(unsigned VirtReg) {
  if (UserValue *UV = virtRegToEqClass.lookup(VirtReg))
    return UV->getLeader();
  return 0;
}

bool LDVImpl::handleDebugValue(MachineInstr *MI, SlotIndex Idx) {
  // DBG_VALUE loc, offset, variable
  if (MI->getNumOperands() != 3 ||
      !MI->getOperand(1).isImm() || !MI->getOperand(2).isMetadata()) {
    DEBUG(dbgs() << "Can't handle " << *MI);
    return false;
  }

  // Get or create the UserValue for (variable,offset).
  unsigned Offset = MI->getOperand(1).getImm();
  const MDNode *Var = MI->getOperand(2).getMetadata();
  UserValue *UV = getUserValue(Var, Offset);

  // If the location is a virtual register, make sure it is mapped.
  if (MI->getOperand(0).isReg()) {
    unsigned Reg = MI->getOperand(0).getReg();
    if (Reg && TargetRegisterInfo::isVirtualRegister(Reg))
      mapVirtReg(Reg, UV);
  }

  UV->addDef(Idx, MI->getOperand(0));
  return true;
}

bool LDVImpl::collectDebugValues(MachineFunction &mf) {
  bool Changed = false;
  for (MachineFunction::iterator MFI = mf.begin(), MFE = mf.end(); MFI != MFE;
       ++MFI) {
    MachineBasicBlock *MBB = MFI;
    for (MachineBasicBlock::iterator MBBI = MBB->begin(), MBBE = MBB->end();
         MBBI != MBBE;) {
      if (!MBBI->isDebugValue()) {
        ++MBBI;
        continue;
      }
      // DBG_VALUE has no slot index, use the previous instruction instead.
      SlotIndex Idx = MBBI == MBB->begin() ?
        LIS->getMBBStartIdx(MBB) :
        LIS->getInstructionIndex(llvm::prior(MBBI)).getDefIndex();
      // Handle consecutive DBG_VALUE instructions with the same slot index.
      do {
        if (handleDebugValue(MBBI, Idx)) {
          MBBI = MBB->erase(MBBI);
          Changed = true;
        } else
          ++MBBI;
      } while (MBBI != MBBE && MBBI->isDebugValue());
    }
  }
  return Changed;
}

void UserValue::extendDef(SlotIndex Idx, unsigned LocNo,
                          LiveInterval *LI, const VNInfo *VNI,
                          LiveIntervals &LIS, MachineDominatorTree &MDT) {
  SmallVector<SlotIndex, 16> Todo;
  Todo.push_back(Idx);

  do {
    SlotIndex Start = Todo.pop_back_val();
    MachineBasicBlock *MBB = LIS.getMBBFromIndex(Start);
    SlotIndex Stop = LIS.getMBBEndIdx(MBB);
    LocMap::iterator I = locInts.find(Idx);

    // Limit to VNI's live range.
    bool ToEnd = true;
    if (LI && VNI) {
      LiveRange *Range = LI->getLiveRangeContaining(Start);
      if (!Range || Range->valno != VNI)
        continue;
      if (Range->end < Stop)
        Stop = Range->end, ToEnd = false;
    }

    // There could already be a short def at Start.
    if (I.valid() && I.start() <= Start) {
      // Stop when meeting a different location or an already extended interval.
      Start = Start.getNextSlot();
      if (I.value() != LocNo || I.stop() != Start)
        continue;
      // This is a one-slot placeholder. Just skip it.
      ++I;
    }

    // Limited by the next def.
    if (I.valid() && I.start() < Stop)
      Stop = I.start(), ToEnd = false;

    if (Start >= Stop)
      continue;

    I.insert(Start, Stop, LocNo);

    // If we extended to the MBB end, propagate down the dominator tree.
    if (!ToEnd)
      continue;
    const std::vector<MachineDomTreeNode*> &Children =
      MDT.getNode(MBB)->getChildren();
    for (unsigned i = 0, e = Children.size(); i != e; ++i)
      Todo.push_back(LIS.getMBBStartIdx(Children[i]->getBlock()));
  } while (!Todo.empty());
}

void
UserValue::computeIntervals(LiveIntervals &LIS, MachineDominatorTree &MDT) {
  SmallVector<std::pair<SlotIndex, unsigned>, 16> Defs;

  // Collect all defs to be extended (Skipping undefs).
  for (LocMap::const_iterator I = locInts.begin(); I.valid(); ++I)
    if (I.value() != ~0u)
      Defs.push_back(std::make_pair(I.start(), I.value()));

  for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
    SlotIndex Idx = Defs[i].first;
    unsigned LocNo = Defs[i].second;
    const MachineOperand &Loc = locations[LocNo];

    // Register locations are constrained to where the register value is live.
    if (Loc.isReg() && LIS.hasInterval(Loc.getReg())) {
      LiveInterval *LI = &LIS.getInterval(Loc.getReg());
      const VNInfo *VNI = LI->getVNInfoAt(Idx);
      extendDef(Idx, LocNo, LI, VNI, LIS, MDT);
    } else
      extendDef(Idx, LocNo, 0, 0, LIS, MDT);
  }

  // Finally, erase all the undefs.
  for (LocMap::iterator I = locInts.begin(); I.valid();)
    if (I.value() == ~0u)
      I.erase();
    else
      ++I;
}

void LDVImpl::computeIntervals() {
  for (unsigned i = 0, e = userValues.size(); i != e; ++i)
    userValues[i]->computeIntervals(*LIS, *MDT);
}

bool LDVImpl::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  LIS = &pass.getAnalysis<LiveIntervals>();
  MDT = &pass.getAnalysis<MachineDominatorTree>();
  TRI = mf.getTarget().getRegisterInfo();
  clear();
  DEBUG(dbgs() << "********** COMPUTING LIVE DEBUG VARIABLES: "
               << ((Value*)mf.getFunction())->getName()
               << " **********\n");

  bool Changed = collectDebugValues(mf);
  computeIntervals();
  DEBUG(print(dbgs()));
  return Changed;
}

bool LiveDebugVariables::runOnMachineFunction(MachineFunction &mf) {
  if (!EnableLDV)
    return false;
  if (!pImpl)
    pImpl = new LDVImpl(this);
  return static_cast<LDVImpl*>(pImpl)->runOnMachineFunction(mf);
}

void LiveDebugVariables::releaseMemory() {
  if (pImpl)
    static_cast<LDVImpl*>(pImpl)->clear();
}

LiveDebugVariables::~LiveDebugVariables() {
  if (pImpl)
    delete static_cast<LDVImpl*>(pImpl);
}

void UserValue::
renameRegister(unsigned OldReg, unsigned NewReg, unsigned SubIdx,
               const TargetRegisterInfo *TRI) {
  for (unsigned i = locations.size(); i; --i) {
    unsigned LocNo = i - 1;
    MachineOperand &Loc = locations[LocNo];
    if (!Loc.isReg() || Loc.getReg() != OldReg)
      continue;
    if (TargetRegisterInfo::isPhysicalRegister(NewReg))
      Loc.substPhysReg(NewReg, *TRI);
    else
      Loc.substVirtReg(NewReg, SubIdx, *TRI);
    coalesceLocation(LocNo);
  }
}

void LDVImpl::
renameRegister(unsigned OldReg, unsigned NewReg, unsigned SubIdx) {
  UserValue *UV = lookupVirtReg(OldReg);
  if (!UV)
    return;

  if (TargetRegisterInfo::isVirtualRegister(NewReg))
    mapVirtReg(NewReg, UV);
  virtRegToEqClass.erase(OldReg);

  do {
    UV->renameRegister(OldReg, NewReg, SubIdx, TRI);
    UV = UV->getNext();
  } while (UV);
}

void LiveDebugVariables::
renameRegister(unsigned OldReg, unsigned NewReg, unsigned SubIdx) {
  if (pImpl)
    static_cast<LDVImpl*>(pImpl)->renameRegister(OldReg, NewReg, SubIdx);
}

void
UserValue::rewriteLocations(VirtRegMap &VRM, const TargetRegisterInfo &TRI) {
  // Iterate over locations in reverse makes it easier to handle coalescing.
  for (unsigned i = locations.size(); i ; --i) {
    unsigned LocNo = i-1;
    MachineOperand &Loc = locations[LocNo];
    // Only virtual registers are rewritten.
    if (!Loc.isReg() || !Loc.getReg() ||
        !TargetRegisterInfo::isVirtualRegister(Loc.getReg()))
      continue;
    unsigned VirtReg = Loc.getReg();
    if (VRM.isAssignedReg(VirtReg)) {
      Loc.substPhysReg(VRM.getPhys(VirtReg), TRI);
    } else if (VRM.getStackSlot(VirtReg) != VirtRegMap::NO_STACK_SLOT) {
      // FIXME: Translate SubIdx to a stackslot offset.
      Loc = MachineOperand::CreateFI(VRM.getStackSlot(VirtReg));
    } else {
      Loc.setReg(0);
      Loc.setSubReg(0);
    }
    coalesceLocation(LocNo);
  }
  DEBUG(print(dbgs(), &TRI));
}

/// findInsertLocation - Find an iterator and DebugLoc for inserting a DBG_VALUE
/// instruction.
static MachineBasicBlock::iterator
findInsertLocation(MachineBasicBlock *MBB, SlotIndex Idx, DebugLoc &DL,
                   LiveIntervals &LIS) {
  SlotIndex Start = LIS.getMBBStartIdx(MBB);
  Idx = Idx.getBaseIndex();

  // Try to find an insert location by going backwards from Idx.
  MachineInstr *MI;
  while (!(MI = LIS.getInstructionFromIndex(Idx))) {
    // We've reached the beginning of MBB.
    if (Idx == Start) {
      MachineBasicBlock::iterator I = MBB->SkipPHIsAndLabels(MBB->begin());
      if (I != MBB->end())
        DL = I->getDebugLoc();
      return I;
    }
    Idx = Idx.getPrevIndex();
  }
  // We found an instruction. The insert point is after the instr.
  DL = MI->getDebugLoc();
  return llvm::next(MachineBasicBlock::iterator(MI));
}

void UserValue::insertDebugValue(MachineBasicBlock *MBB, SlotIndex Idx,
                                 unsigned LocNo,
                                 LiveIntervals &LIS,
                                 const TargetInstrInfo &TII) {
  DebugLoc DL;
  MachineBasicBlock::iterator I = findInsertLocation(MBB, Idx, DL, LIS);
  MachineOperand &Loc = locations[LocNo];

  // Frame index locations may require a target callback.
  if (Loc.isFI()) {
    MachineInstr *MI = TII.emitFrameIndexDebugValue(*MBB->getParent(),
                                          Loc.getIndex(), offset, variable, DL);
    if (MI) {
      MBB->insert(I, MI);
      return;
    }
  }
  // This is not a frame index, or the target is happy with a standard FI.
  BuildMI(*MBB, I, DL, TII.get(TargetOpcode::DBG_VALUE))
    .addOperand(Loc).addImm(offset).addMetadata(variable);
}

void UserValue::insertDebugKill(MachineBasicBlock *MBB, SlotIndex Idx,
                               LiveIntervals &LIS, const TargetInstrInfo &TII) {
  DebugLoc DL;
  MachineBasicBlock::iterator I = findInsertLocation(MBB, Idx, DL, LIS);
  BuildMI(*MBB, I, DL, TII.get(TargetOpcode::DBG_VALUE)).addReg(0)
    .addImm(offset).addMetadata(variable);
}

void UserValue::emitDebugValues(VirtRegMap *VRM, LiveIntervals &LIS,
                                const TargetInstrInfo &TII) {
  MachineFunction::iterator MFEnd = VRM->getMachineFunction().end();

  for (LocMap::const_iterator I = locInts.begin(); I.valid();) {
    SlotIndex Start = I.start();
    SlotIndex Stop = I.stop();
    unsigned LocNo = I.value();
    DEBUG(dbgs() << "\t[" << Start << ';' << Stop << "):" << LocNo);
    MachineFunction::iterator MBB = LIS.getMBBFromIndex(Start);
    SlotIndex MBBEnd = LIS.getMBBEndIdx(MBB);

    DEBUG(dbgs() << " BB#" << MBB->getNumber() << '-' << MBBEnd);
    insertDebugValue(MBB, Start, LocNo, LIS, TII);

    // This interval may span multiple basic blocks.
    // Insert a DBG_VALUE into each one.
    while(Stop > MBBEnd) {
      // Move to the next block.
      Start = MBBEnd;
      if (++MBB == MFEnd)
        break;
      MBBEnd = LIS.getMBBEndIdx(MBB);
      DEBUG(dbgs() << " BB#" << MBB->getNumber() << '-' << MBBEnd);
      insertDebugValue(MBB, Start, LocNo, LIS, TII);
    }
    DEBUG(dbgs() << '\n');
    if (MBB == MFEnd)
      break;

    ++I;
    if (Stop == MBBEnd)
      continue;
    // The current interval ends before MBB.
    // Insert a kill if there is a gap.
    if (!I.valid() || I.start() > Stop)
      insertDebugKill(MBB, Stop, LIS, TII);
  }
}

void LDVImpl::emitDebugValues(VirtRegMap *VRM) {
  DEBUG(dbgs() << "********** EMITTING LIVE DEBUG VARIABLES **********\n");
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();
  for (unsigned i = 0, e = userValues.size(); i != e; ++i) {
    userValues[i]->rewriteLocations(*VRM, *TRI);
    userValues[i]->emitDebugValues(VRM, *LIS, *TII);
  }
}

void LiveDebugVariables::emitDebugValues(VirtRegMap *VRM) {
  if (pImpl)
    static_cast<LDVImpl*>(pImpl)->emitDebugValues(VRM);
}


#ifndef NDEBUG
void LiveDebugVariables::dump() {
  if (pImpl)
    static_cast<LDVImpl*>(pImpl)->print(dbgs());
}
#endif

