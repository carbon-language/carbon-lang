//===-- MipsDelaySlotFiller.cpp - Mips Delay Slot Filler ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simple pass to fill delay slots with useful instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "delay-slot-filler"

#include "Mips.h"
#include "MipsTargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

STATISTIC(FilledSlots, "Number of delay slots filled");
STATISTIC(UsefulSlots, "Number of delay slots filled with instructions that"
                       " are not NOP.");

static cl::opt<bool> DisableDelaySlotFiller(
  "disable-mips-delay-filler",
  cl::init(false),
  cl::desc("Fill all delay slots with NOPs."),
  cl::Hidden);

// This option can be used to silence complaints by machine verifier passes.
static cl::opt<bool> SkipDelaySlotFiller(
  "skip-mips-delay-filler",
  cl::init(false),
  cl::desc("Skip MIPS' delay slot filling pass."),
  cl::Hidden);

static cl::opt<bool> DisableForwardSearch(
  "disable-mips-df-forward-search",
  cl::init(true),
  cl::desc("Disallow MIPS delay filler to search forward."),
  cl::Hidden);

namespace {
  class RegDefsUses {
  public:
    RegDefsUses(TargetMachine &TM);
    void init(const MachineInstr &MI);

    /// This function sets all caller-saved registers in Defs.
    void setCallerSaved(const MachineInstr &MI);

    bool update(const MachineInstr &MI, unsigned Begin, unsigned End);

  private:
    bool checkRegDefsUses(BitVector &NewDefs, BitVector &NewUses, unsigned Reg,
                          bool IsDef) const;

    /// Returns true if Reg or its alias is in RegSet.
    bool isRegInSet(const BitVector &RegSet, unsigned Reg) const;

    const TargetRegisterInfo &TRI;
    BitVector Defs, Uses;
  };

  /// Base class for inspecting loads and stores.
  class InspectMemInstr {
  public:
    virtual bool hasHazard(const MachineInstr &MI) = 0;
    virtual ~InspectMemInstr() {}
  };

  /// This subclass rejects any memory instructions.
  class NoMemInstr : public InspectMemInstr {
  public:
    virtual bool hasHazard(const MachineInstr &MI);
  };

  /// This subclass uses memory dependence information to determine whether a
  /// memory instruction can be moved to a delay slot.
  class MemDefsUses : public InspectMemInstr {
  public:
    MemDefsUses(const MachineFrameInfo *MFI);

    /// Return true if MI cannot be moved to delay slot.
    virtual bool hasHazard(const MachineInstr &MI);

  private:
    /// Update Defs and Uses. Return true if there exist dependences that
    /// disqualify the delay slot candidate between V and values in Uses and Defs.
    bool updateDefsUses(const Value *V, bool MayStore);

    /// Get the list of underlying objects of MI's memory operand.
    bool getUnderlyingObjects(const MachineInstr &MI,
                              SmallVectorImpl<const Value *> &Objects) const;

    const MachineFrameInfo *MFI;
    SmallPtrSet<const Value*, 4> Uses, Defs;

    /// Flags indicating whether loads or stores have been seen.
    bool SeenLoad, SeenStore;

    /// Flags indicating whether loads or stores with no underlying objects have
    /// been seen.
    bool SeenNoObjLoad, SeenNoObjStore;

    /// Memory instructions are not allowed to move to delay slot if this flag
    /// is true.
    bool ForbidMemInstr;
  };

  class Filler : public MachineFunctionPass {
  public:
    Filler(TargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm), TII(tm.getInstrInfo()) { }

    virtual const char *getPassName() const {
      return "Mips Delay Slot Filler";
    }

    bool runOnMachineFunction(MachineFunction &F) {
      if (SkipDelaySlotFiller)
        return false;

      bool Changed = false;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI)
        Changed |= runOnMachineBasicBlock(*FI);
      return Changed;
    }

  private:
    typedef MachineBasicBlock::iterator Iter;
    typedef MachineBasicBlock::reverse_iterator ReverseIter;

    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);

    /// This function checks if it is valid to move Candidate to the delay slot
    /// and returns true if it isn't. It also updates memory and register
    /// dependence information.
    bool delayHasHazard(const MachineInstr &Candidate, RegDefsUses &RegDU,
                        InspectMemInstr &IM) const;

    /// This function searches range [Begin, End) for an instruction that can be
    /// moved to the delay slot. Returns true on success.
    template<typename IterTy>
    bool searchRange(MachineBasicBlock &MBB, IterTy Begin, IterTy End,
                     RegDefsUses &RegDU, InspectMemInstr &IM, IterTy &Filler) const;

    /// This function searches in the backward direction for an instruction that
    /// can be moved to the delay slot. Returns true on success.
    bool searchBackward(MachineBasicBlock &MBB, Iter Slot) const;

    /// This function searches MBB in the forward direction for an instruction
    /// that can be moved to the delay slot. Returns true on success.
    bool searchForward(MachineBasicBlock &MBB, Iter Slot) const;

    bool terminateSearch(const MachineInstr &Candidate) const;

    TargetMachine &TM;
    const TargetInstrInfo *TII;

    static char ID;
  };
  char Filler::ID = 0;
} // end of anonymous namespace

RegDefsUses::RegDefsUses(TargetMachine &TM)
  : TRI(*TM.getRegisterInfo()), Defs(TRI.getNumRegs(), false),
    Uses(TRI.getNumRegs(), false) {}

void RegDefsUses::init(const MachineInstr &MI) {
  // Add all register operands which are explicit and non-variadic.
  update(MI, 0, MI.getDesc().getNumOperands());

  // If MI is a call, add RA to Defs to prevent users of RA from going into
  // delay slot.
  if (MI.isCall())
    Defs.set(Mips::RA);

  // Add all implicit register operands of branch instructions except
  // register AT.
  if (MI.isBranch()) {
    update(MI, MI.getDesc().getNumOperands(), MI.getNumOperands());
    Defs.reset(Mips::AT);
  }
}

void RegDefsUses::setCallerSaved(const MachineInstr &MI) {
  assert(MI.isCall());

  // If MI is a call, add all caller-saved registers to Defs.
  BitVector CallerSavedRegs(TRI.getNumRegs(), true);

  CallerSavedRegs.reset(Mips::ZERO);
  CallerSavedRegs.reset(Mips::ZERO_64);

  for (const MCPhysReg *R = TRI.getCalleeSavedRegs(); *R; ++R)
    for (MCRegAliasIterator AI(*R, &TRI, true); AI.isValid(); ++AI)
      CallerSavedRegs.reset(*AI);

  Defs |= CallerSavedRegs;
}

bool RegDefsUses::update(const MachineInstr &MI, unsigned Begin, unsigned End) {
  BitVector NewDefs(TRI.getNumRegs()), NewUses(TRI.getNumRegs());
  bool HasHazard = false;

  for (unsigned I = Begin; I != End; ++I) {
    const MachineOperand &MO = MI.getOperand(I);

    if (MO.isReg() && MO.getReg())
      HasHazard |= checkRegDefsUses(NewDefs, NewUses, MO.getReg(), MO.isDef());
  }

  Defs |= NewDefs;
  Uses |= NewUses;

  return HasHazard;
}

bool RegDefsUses::checkRegDefsUses(BitVector &NewDefs, BitVector &NewUses,
                                   unsigned Reg, bool IsDef) const {
  if (IsDef) {
    NewDefs.set(Reg);
    // check whether Reg has already been defined or used.
    return (isRegInSet(Defs, Reg) || isRegInSet(Uses, Reg));
  }

  NewUses.set(Reg);
  // check whether Reg has already been defined.
  return isRegInSet(Defs, Reg);
}

bool RegDefsUses::isRegInSet(const BitVector &RegSet, unsigned Reg) const {
  // Check Reg and all aliased Registers.
  for (MCRegAliasIterator AI(Reg, &TRI, true); AI.isValid(); ++AI)
    if (RegSet.test(*AI))
      return true;
  return false;
}

bool NoMemInstr::hasHazard(const MachineInstr &MI) {
  // Return true if MI accesses memory.
  return (MI.mayStore() || MI.mayLoad());
}

MemDefsUses::MemDefsUses(const MachineFrameInfo *MFI_)
  : MFI(MFI_), SeenLoad(false), SeenStore(false), SeenNoObjLoad(false),
    SeenNoObjStore(false),  ForbidMemInstr(false) {}

bool MemDefsUses::hasHazard(const MachineInstr &MI) {
  if (!MI.mayStore() && !MI.mayLoad())
    return false;

  if (ForbidMemInstr)
    return true;

  bool OrigSeenLoad = SeenLoad, OrigSeenStore = SeenStore;

  SeenLoad |= MI.mayLoad();
  SeenStore |= MI.mayStore();

  // If MI is an ordered or volatile memory reference, disallow moving
  // subsequent loads and stores to delay slot.
  if (MI.hasOrderedMemoryRef() && (OrigSeenLoad || OrigSeenStore)) {
    ForbidMemInstr = true;
    return true;
  }

  bool HasHazard = false;
  SmallVector<const Value *, 4> Objs;

  // Check underlying object list.
  if (getUnderlyingObjects(MI, Objs)) {
    for (SmallVector<const Value *, 4>::const_iterator I = Objs.begin();
         I != Objs.end(); ++I)
      HasHazard |= updateDefsUses(*I, MI.mayStore());

    return HasHazard;
  }

  // No underlying objects found.
  HasHazard = MI.mayStore() && (OrigSeenLoad || OrigSeenStore);
  HasHazard |= MI.mayLoad() || OrigSeenStore;

  SeenNoObjLoad |= MI.mayLoad();
  SeenNoObjStore |= MI.mayStore();

  return HasHazard;
}

bool MemDefsUses::updateDefsUses(const Value *V, bool MayStore) {
  if (MayStore)
    return !Defs.insert(V) || Uses.count(V) || SeenNoObjStore || SeenNoObjLoad;

  Uses.insert(V);
  return Defs.count(V) || SeenNoObjStore;
}

bool MemDefsUses::
getUnderlyingObjects(const MachineInstr &MI,
                     SmallVectorImpl<const Value *> &Objects) const {
  if (!MI.hasOneMemOperand() || !(*MI.memoperands_begin())->getValue())
    return false;

  const Value *V = (*MI.memoperands_begin())->getValue();

  SmallVector<Value *, 4> Objs;
  GetUnderlyingObjects(const_cast<Value *>(V), Objs);

  for (SmallVector<Value*, 4>::iterator I = Objs.begin(), E = Objs.end();
       I != E; ++I) {
    if (const PseudoSourceValue *PSV = dyn_cast<PseudoSourceValue>(*I)) {
      if (PSV->isAliased(MFI))
        return false;
    } else if (!isIdentifiedObject(V))
      return false;

    Objects.push_back(*I);
  }

  return true;
}

/// runOnMachineBasicBlock - Fill in delay slots for the given basic block.
/// We assume there is only one delay slot per delayed instruction.
bool Filler::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (Iter I = MBB.begin(); I != MBB.end(); ++I) {
    if (!I->hasDelaySlot())
      continue;

    ++FilledSlots;
    Changed = true;

    // Delay slot filling is disabled at -O0.
    if (!DisableDelaySlotFiller && (TM.getOptLevel() != CodeGenOpt::None) &&
        (searchBackward(MBB, I) || searchForward(MBB, I)))
      continue;

    // Bundle the NOP to the instruction with the delay slot.
    BuildMI(MBB, llvm::next(I), I->getDebugLoc(), TII->get(Mips::NOP));
    MIBundleBuilder(MBB, I, llvm::next(llvm::next(I)));
  }

  return Changed;
}

/// createMipsDelaySlotFillerPass - Returns a pass that fills in delay
/// slots in Mips MachineFunctions
FunctionPass *llvm::createMipsDelaySlotFillerPass(MipsTargetMachine &tm) {
  return new Filler(tm);
}

template<typename IterTy>
bool Filler::searchRange(MachineBasicBlock &MBB, IterTy Begin, IterTy End,
                         RegDefsUses &RegDU, InspectMemInstr& IM,
                         IterTy &Filler) const {
  for (IterTy I = Begin; I != End; ++I) {
    // skip debug value
    if (I->isDebugValue())
      continue;

    if (terminateSearch(*I))
      break;

    assert((!I->isCall() && !I->isReturn() && !I->isBranch()) &&
           "Cannot put calls, returns or branches in delay slot.");

    if (delayHasHazard(*I, RegDU, IM))
      continue;

    Filler = I;
    return true;
  }

  return false;
}

bool Filler::searchBackward(MachineBasicBlock &MBB, Iter Slot) const {
  RegDefsUses RegDU(TM);
  MemDefsUses MemDU(MBB.getParent()->getFrameInfo());
  ReverseIter Filler;

  RegDU.init(*Slot);

  if (searchRange(MBB, ReverseIter(Slot), MBB.rend(), RegDU, MemDU, Filler)) {
    MBB.splice(llvm::next(Slot), &MBB, llvm::next(Filler).base());
    MIBundleBuilder(MBB, Slot, llvm::next(llvm::next(Slot)));
    ++UsefulSlots;
    return true;
  }

  return false;
}

bool Filler::searchForward(MachineBasicBlock &MBB, Iter Slot) const {
  // Can handle only calls.
  if (!Slot->isCall())
    return false;

  RegDefsUses RegDU(TM);
  NoMemInstr NM;
  Iter Filler;

  RegDU.setCallerSaved(*Slot);

  if (searchRange(MBB, llvm::next(Slot), MBB.end(), RegDU, NM, Filler)) {
    MBB.splice(llvm::next(Slot), &MBB, Filler);
    MIBundleBuilder(MBB, Slot, llvm::next(llvm::next(Slot)));
    ++UsefulSlots;
    return true;
  }

  return false;
}

bool Filler::delayHasHazard(const MachineInstr &Candidate, RegDefsUses &RegDU,
                            InspectMemInstr &IM) const {
  bool HasHazard = (Candidate.isImplicitDef() || Candidate.isKill());

  HasHazard |= IM.hasHazard(Candidate);
  HasHazard |= RegDU.update(Candidate, 0, Candidate.getNumOperands());

  return HasHazard;
}

bool Filler::terminateSearch(const MachineInstr &Candidate) const {
  return (Candidate.isTerminator() || Candidate.isCall() ||
          Candidate.isLabel() || Candidate.isInlineAsm() ||
          Candidate.hasUnmodeledSideEffects());
}
