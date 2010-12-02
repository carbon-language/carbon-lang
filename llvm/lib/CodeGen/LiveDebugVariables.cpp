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
#include "llvm/Constants.h"
#include "llvm/Metadata.h"
#include "llvm/Value.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
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

/// Location - All the different places a user value can reside.
/// Note that this includes immediate values that technically aren't locations.
namespace {
struct Location {
  /// kind - What kind of location is this?
  enum Kind {
    locUndef = 0,
    locImm   = 0x80000000,
    locFPImm
  };
  /// Kind - One of the following:
  /// 1. locUndef
  /// 2. Register number (physical or virtual), data.SubIdx is the subreg index.
  /// 3. ~Frame index, data.Offset is the offset.
  /// 4. locImm, data.ImmVal is the constant integer value.
  /// 5. locFPImm, data.CFP points to the floating point constant.
  unsigned Kind;

  /// Data - Extra data about location.
  union {
    unsigned SubIdx;          ///< For virtual registers.
    int64_t Offset;           ///< For frame indices.
    int64_t ImmVal;           ///< For locImm.
    const ConstantFP *CFP;    ///< For locFPImm.
  } Data;

  Location(const MachineOperand &MO) {
    switch(MO.getType()) {
    case MachineOperand::MO_Register:
      Kind = MO.getReg();
      Data.SubIdx = MO.getSubReg();
      return;
    case MachineOperand::MO_Immediate:
      Kind = locImm;
      Data.ImmVal = MO.getImm();
      return;
    case MachineOperand::MO_FPImmediate:
      Kind = locFPImm;
      Data.CFP = MO.getFPImm();
      return;
    case MachineOperand::MO_FrameIndex:
      Kind = ~MO.getIndex();
      // FIXME: MO_FrameIndex should support an offset.
      Data.Offset = 0;
      return;
    default:
      Kind = locUndef;
      return;
    }
  }

  bool operator==(const Location &RHS) const {
    if (Kind != RHS.Kind)
      return false;
    switch (Kind) {
    case locUndef:
      return true;
    case locImm:
      return Data.ImmVal == RHS.Data.ImmVal;
    case locFPImm:
      return Data.CFP == RHS.Data.CFP;
    default:
      if (isReg())
        return Data.SubIdx == RHS.Data.SubIdx;
      else
         return Data.Offset == RHS.Data.Offset;
    }
  }

  /// isUndef - is this the singleton undef?
  bool isUndef() const { return Kind == locUndef; }

  /// isReg - is this a register location?
  bool isReg() const { return Kind && Kind < locImm; }

  void print(raw_ostream&, const TargetRegisterInfo*);
};
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
  SmallVector<Location, 4> locations;

  /// Map of slot indices where this value is live.
  LocMap locInts;

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
  unsigned getLocationNo(Location Loc) {
    if (Loc.isUndef())
      return ~0u;
    unsigned n = std::find(locations.begin(), locations.end(), Loc) -
                 locations.begin();
    if (n == locations.size())
      locations.push_back(Loc);
    return n;
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
  VRMap virtRegMap;

  /// Map user variable to eq class leader.
  typedef DenseMap<const MDNode *, UserValue*> UVMap;
  UVMap userVarMap;

  /// getUserValue - Find or create a UserValue.
  UserValue *getUserValue(const MDNode *Var, unsigned Offset);

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
    virtRegMap.clear();
    userVarMap.clear();
  }

  void print(raw_ostream&);
};
} // namespace

void Location::print(raw_ostream &OS, const TargetRegisterInfo *TRI) {
  switch (Kind) {
  case locUndef:
    OS << "undef";
    return;
  case locImm:
    OS << "int:" << Data.ImmVal;
    return;
  case locFPImm:
    OS << "fp:" << Data.CFP->getValueAPF().convertToDouble();
    return;
  default:
    if (isReg()) {
      if (TargetRegisterInfo::isVirtualRegister(Kind)) {
        OS << "%reg" << Kind;
        if (Data.SubIdx)
          OS << ':' << TRI->getSubRegIndexName(Data.SubIdx);
      } else
        OS << '%' << TRI->getName(Kind);
    } else {
      OS << "fi#" << ~Kind;
      if (Data.Offset)
        OS << '+' << Data.Offset;
    }
    return;
  }
}

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
  for (unsigned i = 0, e = locations.size(); i != e; ++i) {
    OS << " Loc" << i << '=';
    locations[i].print(OS, TRI);
  }
  OS << '\n';
}

void LDVImpl::print(raw_ostream &OS) {
  OS << "********** DEBUG VARIABLES **********\n";
  for (unsigned i = 0, e = userValues.size(); i != e; ++i)
    userValues[i]->print(OS, TRI);
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
  UserValue *&Leader = virtRegMap[VirtReg];
  Leader = UserValue::merge(Leader, EC);
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
    const Location &Loc = locations[LocNo];

    // Register locations are constrained to where the register value is live.
    if (Loc.isReg() && LIS.hasInterval(Loc.Kind)) {
      LiveInterval *LI = &LIS.getInterval(Loc.Kind);
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
