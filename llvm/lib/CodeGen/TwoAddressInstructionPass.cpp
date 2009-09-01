//===-- TwoAddressInstructionPass.cpp - Two-Address instruction pass ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TwoAddress instruction pass which is used
// by most register allocators. Two-Address instructions are rewritten
// from:
//
//     A = B op C
//
// to:
//
//     A = B
//     A op= C
//
// Note that if a register allocator chooses to use this pass, that it
// has to be capable of handling the non-SSA nature of these rewritten
// virtual registers.
//
// It is also worth noting that the duplicate operand of the two
// address instruction is removed.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "twoaddrinstr"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

STATISTIC(NumTwoAddressInstrs, "Number of two-address instructions");
STATISTIC(NumCommuted        , "Number of instructions commuted to coalesce");
STATISTIC(NumAggrCommuted    , "Number of instructions aggressively commuted");
STATISTIC(NumConvertedTo3Addr, "Number of instructions promoted to 3-address");
STATISTIC(Num3AddrSunk,        "Number of 3-address instructions sunk");
STATISTIC(NumReMats,           "Number of instructions re-materialized");
STATISTIC(NumDeletes,          "Number of dead instructions deleted");

namespace {
  class VISIBILITY_HIDDEN TwoAddressInstructionPass
    : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    MachineRegisterInfo *MRI;
    LiveVariables *LV;

    // DistanceMap - Keep track the distance of a MI from the start of the
    // current basic block.
    DenseMap<MachineInstr*, unsigned> DistanceMap;

    // SrcRegMap - A map from virtual registers to physical registers which
    // are likely targets to be coalesced to due to copies from physical
    // registers to virtual registers. e.g. v1024 = move r0.
    DenseMap<unsigned, unsigned> SrcRegMap;

    // DstRegMap - A map from virtual registers to physical registers which
    // are likely targets to be coalesced to due to copies to physical
    // registers from virtual registers. e.g. r1 = move v1024.
    DenseMap<unsigned, unsigned> DstRegMap;

    bool Sink3AddrInstruction(MachineBasicBlock *MBB, MachineInstr *MI,
                              unsigned Reg,
                              MachineBasicBlock::iterator OldPos);

    bool isProfitableToReMat(unsigned Reg, const TargetRegisterClass *RC,
                             MachineInstr *MI, MachineInstr *DefMI,
                             MachineBasicBlock *MBB, unsigned Loc);

    bool NoUseAfterLastDef(unsigned Reg, MachineBasicBlock *MBB, unsigned Dist,
                           unsigned &LastDef);

    MachineInstr *FindLastUseInMBB(unsigned Reg, MachineBasicBlock *MBB,
                                   unsigned Dist);

    bool isProfitableToCommute(unsigned regB, unsigned regC,
                               MachineInstr *MI, MachineBasicBlock *MBB,
                               unsigned Dist);

    bool CommuteInstruction(MachineBasicBlock::iterator &mi,
                            MachineFunction::iterator &mbbi,
                            unsigned RegB, unsigned RegC, unsigned Dist);

    bool isProfitableToConv3Addr(unsigned RegA);

    bool ConvertInstTo3Addr(MachineBasicBlock::iterator &mi,
                            MachineBasicBlock::iterator &nmi,
                            MachineFunction::iterator &mbbi,
                            unsigned RegB, unsigned Dist);

    typedef std::pair<std::pair<unsigned, bool>, MachineInstr*> NewKill;
    bool canUpdateDeletedKills(SmallVector<unsigned, 4> &Kills,
                               SmallVector<NewKill, 4> &NewKills,
                               MachineBasicBlock *MBB, unsigned Dist);
    bool DeleteUnusedInstr(MachineBasicBlock::iterator &mi,
                           MachineBasicBlock::iterator &nmi,
                           MachineFunction::iterator &mbbi,
                           unsigned regB, unsigned regBIdx, unsigned Dist);

    void ProcessCopy(MachineInstr *MI, MachineBasicBlock *MBB,
                     SmallPtrSet<MachineInstr*, 8> &Processed);

  public:
    static char ID; // Pass identification, replacement for typeid
    TwoAddressInstructionPass() : MachineFunctionPass(&ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addPreserved<LiveVariables>();
      AU.addPreservedID(MachineLoopInfoID);
      AU.addPreservedID(MachineDominatorsID);
      if (StrongPHIElim)
        AU.addPreservedID(StrongPHIEliminationID);
      else
        AU.addPreservedID(PHIEliminationID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    /// runOnMachineFunction - Pass entry point.
    bool runOnMachineFunction(MachineFunction&);
  };
}

char TwoAddressInstructionPass::ID = 0;
static RegisterPass<TwoAddressInstructionPass>
X("twoaddressinstruction", "Two-Address instruction pass");

const PassInfo *const llvm::TwoAddressInstructionPassID = &X;

/// Sink3AddrInstruction - A two-address instruction has been converted to a
/// three-address instruction to avoid clobbering a register. Try to sink it
/// past the instruction that would kill the above mentioned register to reduce
/// register pressure.
bool TwoAddressInstructionPass::Sink3AddrInstruction(MachineBasicBlock *MBB,
                                           MachineInstr *MI, unsigned SavedReg,
                                           MachineBasicBlock::iterator OldPos) {
  // Check if it's safe to move this instruction.
  bool SeenStore = true; // Be conservative.
  if (!MI->isSafeToMove(TII, SeenStore))
    return false;

  unsigned DefReg = 0;
  SmallSet<unsigned, 4> UseRegs;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned MOReg = MO.getReg();
    if (!MOReg)
      continue;
    if (MO.isUse() && MOReg != SavedReg)
      UseRegs.insert(MO.getReg());
    if (!MO.isDef())
      continue;
    if (MO.isImplicit())
      // Don't try to move it if it implicitly defines a register.
      return false;
    if (DefReg)
      // For now, don't move any instructions that define multiple registers.
      return false;
    DefReg = MO.getReg();
  }

  // Find the instruction that kills SavedReg.
  MachineInstr *KillMI = NULL;
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(SavedReg),
         UE = MRI->use_end(); UI != UE; ++UI) {
    MachineOperand &UseMO = UI.getOperand();
    if (!UseMO.isKill())
      continue;
    KillMI = UseMO.getParent();
    break;
  }

  if (!KillMI || KillMI->getParent() != MBB || KillMI == MI)
    return false;

  // If any of the definitions are used by another instruction between the
  // position and the kill use, then it's not safe to sink it.
  // 
  // FIXME: This can be sped up if there is an easy way to query whether an
  // instruction is before or after another instruction. Then we can use
  // MachineRegisterInfo def / use instead.
  MachineOperand *KillMO = NULL;
  MachineBasicBlock::iterator KillPos = KillMI;
  ++KillPos;

  unsigned NumVisited = 0;
  for (MachineBasicBlock::iterator I = next(OldPos); I != KillPos; ++I) {
    MachineInstr *OtherMI = I;
    if (NumVisited > 30)  // FIXME: Arbitrary limit to reduce compile time cost.
      return false;
    ++NumVisited;
    for (unsigned i = 0, e = OtherMI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = OtherMI->getOperand(i);
      if (!MO.isReg())
        continue;
      unsigned MOReg = MO.getReg();
      if (!MOReg)
        continue;
      if (DefReg == MOReg)
        return false;

      if (MO.isKill()) {
        if (OtherMI == KillMI && MOReg == SavedReg)
          // Save the operand that kills the register. We want to unset the kill
          // marker if we can sink MI past it.
          KillMO = &MO;
        else if (UseRegs.count(MOReg))
          // One of the uses is killed before the destination.
          return false;
      }
    }
  }

  // Update kill and LV information.
  KillMO->setIsKill(false);
  KillMO = MI->findRegisterUseOperand(SavedReg, false, TRI);
  KillMO->setIsKill(true);
  
  if (LV)
    LV->replaceKillInstruction(SavedReg, KillMI, MI);

  // Move instruction to its destination.
  MBB->remove(MI);
  MBB->insert(KillPos, MI);

  ++Num3AddrSunk;
  return true;
}

/// isTwoAddrUse - Return true if the specified MI is using the specified
/// register as a two-address operand.
static bool isTwoAddrUse(MachineInstr *UseMI, unsigned Reg) {
  const TargetInstrDesc &TID = UseMI->getDesc();
  for (unsigned i = 0, e = TID.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = UseMI->getOperand(i);
    if (MO.isReg() && MO.getReg() == Reg &&
        (MO.isDef() || UseMI->isRegTiedToDefOperand(i)))
      // Earlier use is a two-address one.
      return true;
  }
  return false;
}

/// isProfitableToReMat - Return true if the heuristics determines it is likely
/// to be profitable to re-materialize the definition of Reg rather than copy
/// the register.
bool
TwoAddressInstructionPass::isProfitableToReMat(unsigned Reg,
                                         const TargetRegisterClass *RC,
                                         MachineInstr *MI, MachineInstr *DefMI,
                                         MachineBasicBlock *MBB, unsigned Loc) {
  bool OtherUse = false;
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(Reg),
         UE = MRI->use_end(); UI != UE; ++UI) {
    MachineOperand &UseMO = UI.getOperand();
    MachineInstr *UseMI = UseMO.getParent();
    MachineBasicBlock *UseMBB = UseMI->getParent();
    if (UseMBB == MBB) {
      DenseMap<MachineInstr*, unsigned>::iterator DI = DistanceMap.find(UseMI);
      if (DI != DistanceMap.end() && DI->second == Loc)
        continue;  // Current use.
      OtherUse = true;
      // There is at least one other use in the MBB that will clobber the
      // register. 
      if (isTwoAddrUse(UseMI, Reg))
        return true;
    }
  }

  // If other uses in MBB are not two-address uses, then don't remat.
  if (OtherUse)
    return false;

  // No other uses in the same block, remat if it's defined in the same
  // block so it does not unnecessarily extend the live range.
  return MBB == DefMI->getParent();
}

/// NoUseAfterLastDef - Return true if there are no intervening uses between the
/// last instruction in the MBB that defines the specified register and the
/// two-address instruction which is being processed. It also returns the last
/// def location by reference
bool TwoAddressInstructionPass::NoUseAfterLastDef(unsigned Reg,
                                           MachineBasicBlock *MBB, unsigned Dist,
                                           unsigned &LastDef) {
  LastDef = 0;
  unsigned LastUse = Dist;
  for (MachineRegisterInfo::reg_iterator I = MRI->reg_begin(Reg),
         E = MRI->reg_end(); I != E; ++I) {
    MachineOperand &MO = I.getOperand();
    MachineInstr *MI = MO.getParent();
    if (MI->getParent() != MBB)
      continue;
    DenseMap<MachineInstr*, unsigned>::iterator DI = DistanceMap.find(MI);
    if (DI == DistanceMap.end())
      continue;
    if (MO.isUse() && DI->second < LastUse)
      LastUse = DI->second;
    if (MO.isDef() && DI->second > LastDef)
      LastDef = DI->second;
  }

  return !(LastUse > LastDef && LastUse < Dist);
}

MachineInstr *TwoAddressInstructionPass::FindLastUseInMBB(unsigned Reg,
                                                         MachineBasicBlock *MBB,
                                                         unsigned Dist) {
  unsigned LastUseDist = 0;
  MachineInstr *LastUse = 0;
  for (MachineRegisterInfo::reg_iterator I = MRI->reg_begin(Reg),
         E = MRI->reg_end(); I != E; ++I) {
    MachineOperand &MO = I.getOperand();
    MachineInstr *MI = MO.getParent();
    if (MI->getParent() != MBB)
      continue;
    DenseMap<MachineInstr*, unsigned>::iterator DI = DistanceMap.find(MI);
    if (DI == DistanceMap.end())
      continue;
    if (DI->second >= Dist)
      continue;

    if (MO.isUse() && DI->second > LastUseDist) {
      LastUse = DI->first;
      LastUseDist = DI->second;
    }
  }
  return LastUse;
}

/// isCopyToReg - Return true if the specified MI is a copy instruction or
/// a extract_subreg instruction. It also returns the source and destination
/// registers and whether they are physical registers by reference.
static bool isCopyToReg(MachineInstr &MI, const TargetInstrInfo *TII,
                        unsigned &SrcReg, unsigned &DstReg,
                        bool &IsSrcPhys, bool &IsDstPhys) {
  SrcReg = 0;
  DstReg = 0;
  unsigned SrcSubIdx, DstSubIdx;
  if (!TII->isMoveInstr(MI, SrcReg, DstReg, SrcSubIdx, DstSubIdx)) {
    if (MI.getOpcode() == TargetInstrInfo::EXTRACT_SUBREG) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
    } else if (MI.getOpcode() == TargetInstrInfo::INSERT_SUBREG) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(2).getReg();
    } else if (MI.getOpcode() == TargetInstrInfo::SUBREG_TO_REG) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(2).getReg();
    }
  }

  if (DstReg) {
    IsSrcPhys = TargetRegisterInfo::isPhysicalRegister(SrcReg);
    IsDstPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);
    return true;
  }
  return false;
}

/// isKilled - Test if the given register value, which is used by the given
/// instruction, is killed by the given instruction. This looks through
/// coalescable copies to see if the original value is potentially not killed.
///
/// For example, in this code:
///
///   %reg1034 = copy %reg1024
///   %reg1035 = copy %reg1025<kill>
///   %reg1036 = add %reg1034<kill>, %reg1035<kill>
///
/// %reg1034 is not considered to be killed, since it is copied from a
/// register which is not killed. Treating it as not killed lets the
/// normal heuristics commute the (two-address) add, which lets
/// coalescing eliminate the extra copy.
///
static bool isKilled(MachineInstr &MI, unsigned Reg,
                     const MachineRegisterInfo *MRI,
                     const TargetInstrInfo *TII) {
  MachineInstr *DefMI = &MI;
  for (;;) {
    if (!DefMI->killsRegister(Reg))
      return false;
    if (TargetRegisterInfo::isPhysicalRegister(Reg))
      return true;
    MachineRegisterInfo::def_iterator Begin = MRI->def_begin(Reg);
    // If there are multiple defs, we can't do a simple analysis, so just
    // go with what the kill flag says.
    if (next(Begin) != MRI->def_end())
      return true;
    DefMI = &*Begin;
    bool IsSrcPhys, IsDstPhys;
    unsigned SrcReg,  DstReg;
    // If the def is something other than a copy, then it isn't going to
    // be coalesced, so follow the kill flag.
    if (!isCopyToReg(*DefMI, TII, SrcReg, DstReg, IsSrcPhys, IsDstPhys))
      return true;
    Reg = SrcReg;
  }
}

/// isTwoAddrUse - Return true if the specified MI uses the specified register
/// as a two-address use. If so, return the destination register by reference.
static bool isTwoAddrUse(MachineInstr &MI, unsigned Reg, unsigned &DstReg) {
  const TargetInstrDesc &TID = MI.getDesc();
  unsigned NumOps = (MI.getOpcode() == TargetInstrInfo::INLINEASM)
    ? MI.getNumOperands() : TID.getNumOperands();
  for (unsigned i = 0; i != NumOps; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || !MO.isUse() || MO.getReg() != Reg)
      continue;
    unsigned ti;
    if (MI.isRegTiedToDefOperand(i, &ti)) {
      DstReg = MI.getOperand(ti).getReg();
      return true;
    }
  }
  return false;
}

/// findOnlyInterestingUse - Given a register, if has a single in-basic block
/// use, return the use instruction if it's a copy or a two-address use.
static
MachineInstr *findOnlyInterestingUse(unsigned Reg, MachineBasicBlock *MBB,
                                     MachineRegisterInfo *MRI,
                                     const TargetInstrInfo *TII,
                                     bool &IsCopy,
                                     unsigned &DstReg, bool &IsDstPhys) {
  MachineRegisterInfo::use_iterator UI = MRI->use_begin(Reg);
  if (UI == MRI->use_end())
    return 0;
  MachineInstr &UseMI = *UI;
  if (++UI != MRI->use_end())
    // More than one use.
    return 0;
  if (UseMI.getParent() != MBB)
    return 0;
  unsigned SrcReg;
  bool IsSrcPhys;
  if (isCopyToReg(UseMI, TII, SrcReg, DstReg, IsSrcPhys, IsDstPhys)) {
    IsCopy = true;
    return &UseMI;
  }
  IsDstPhys = false;
  if (isTwoAddrUse(UseMI, Reg, DstReg)) {
    IsDstPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);
    return &UseMI;
  }
  return 0;
}

/// getMappedReg - Return the physical register the specified virtual register
/// might be mapped to.
static unsigned
getMappedReg(unsigned Reg, DenseMap<unsigned, unsigned> &RegMap) {
  while (TargetRegisterInfo::isVirtualRegister(Reg))  {
    DenseMap<unsigned, unsigned>::iterator SI = RegMap.find(Reg);
    if (SI == RegMap.end())
      return 0;
    Reg = SI->second;
  }
  if (TargetRegisterInfo::isPhysicalRegister(Reg))
    return Reg;
  return 0;
}

/// regsAreCompatible - Return true if the two registers are equal or aliased.
///
static bool
regsAreCompatible(unsigned RegA, unsigned RegB, const TargetRegisterInfo *TRI) {
  if (RegA == RegB)
    return true;
  if (!RegA || !RegB)
    return false;
  return TRI->regsOverlap(RegA, RegB);
}


/// isProfitableToReMat - Return true if it's potentially profitable to commute
/// the two-address instruction that's being processed.
bool
TwoAddressInstructionPass::isProfitableToCommute(unsigned regB, unsigned regC,
                                       MachineInstr *MI, MachineBasicBlock *MBB,
                                       unsigned Dist) {
  // Determine if it's profitable to commute this two address instruction. In
  // general, we want no uses between this instruction and the definition of
  // the two-address register.
  // e.g.
  // %reg1028<def> = EXTRACT_SUBREG %reg1027<kill>, 1
  // %reg1029<def> = MOV8rr %reg1028
  // %reg1029<def> = SHR8ri %reg1029, 7, %EFLAGS<imp-def,dead>
  // insert => %reg1030<def> = MOV8rr %reg1028
  // %reg1030<def> = ADD8rr %reg1028<kill>, %reg1029<kill>, %EFLAGS<imp-def,dead>
  // In this case, it might not be possible to coalesce the second MOV8rr
  // instruction if the first one is coalesced. So it would be profitable to
  // commute it:
  // %reg1028<def> = EXTRACT_SUBREG %reg1027<kill>, 1
  // %reg1029<def> = MOV8rr %reg1028
  // %reg1029<def> = SHR8ri %reg1029, 7, %EFLAGS<imp-def,dead>
  // insert => %reg1030<def> = MOV8rr %reg1029
  // %reg1030<def> = ADD8rr %reg1029<kill>, %reg1028<kill>, %EFLAGS<imp-def,dead>  

  if (!MI->killsRegister(regC))
    return false;

  // Ok, we have something like:
  // %reg1030<def> = ADD8rr %reg1028<kill>, %reg1029<kill>, %EFLAGS<imp-def,dead>
  // let's see if it's worth commuting it.

  // Look for situations like this:
  // %reg1024<def> = MOV r1
  // %reg1025<def> = MOV r0
  // %reg1026<def> = ADD %reg1024, %reg1025
  // r0            = MOV %reg1026
  // Commute the ADD to hopefully eliminate an otherwise unavoidable copy.
  unsigned FromRegB = getMappedReg(regB, SrcRegMap);
  unsigned FromRegC = getMappedReg(regC, SrcRegMap);
  unsigned ToRegB = getMappedReg(regB, DstRegMap);
  unsigned ToRegC = getMappedReg(regC, DstRegMap);
  if (!regsAreCompatible(FromRegB, ToRegB, TRI) &&
      (regsAreCompatible(FromRegB, ToRegC, TRI) ||
       regsAreCompatible(FromRegC, ToRegB, TRI)))
    return true;

  // If there is a use of regC between its last def (could be livein) and this
  // instruction, then bail.
  unsigned LastDefC = 0;
  if (!NoUseAfterLastDef(regC, MBB, Dist, LastDefC))
    return false;

  // If there is a use of regB between its last def (could be livein) and this
  // instruction, then go ahead and make this transformation.
  unsigned LastDefB = 0;
  if (!NoUseAfterLastDef(regB, MBB, Dist, LastDefB))
    return true;

  // Since there are no intervening uses for both registers, then commute
  // if the def of regC is closer. Its live interval is shorter.
  return LastDefB && LastDefC && LastDefC > LastDefB;
}

/// CommuteInstruction - Commute a two-address instruction and update the basic
/// block, distance map, and live variables if needed. Return true if it is
/// successful.
bool
TwoAddressInstructionPass::CommuteInstruction(MachineBasicBlock::iterator &mi,
                               MachineFunction::iterator &mbbi,
                               unsigned RegB, unsigned RegC, unsigned Dist) {
  MachineInstr *MI = mi;
  DEBUG(errs() << "2addr: COMMUTING  : " << *MI);
  MachineInstr *NewMI = TII->commuteInstruction(MI);

  if (NewMI == 0) {
    DEBUG(errs() << "2addr: COMMUTING FAILED!\n");
    return false;
  }

  DEBUG(errs() << "2addr: COMMUTED TO: " << *NewMI);
  // If the instruction changed to commute it, update livevar.
  if (NewMI != MI) {
    if (LV)
      // Update live variables
      LV->replaceKillInstruction(RegC, MI, NewMI);

    mbbi->insert(mi, NewMI);           // Insert the new inst
    mbbi->erase(mi);                   // Nuke the old inst.
    mi = NewMI;
    DistanceMap.insert(std::make_pair(NewMI, Dist));
  }

  // Update source register map.
  unsigned FromRegC = getMappedReg(RegC, SrcRegMap);
  if (FromRegC) {
    unsigned RegA = MI->getOperand(0).getReg();
    SrcRegMap[RegA] = FromRegC;
  }

  return true;
}

/// isProfitableToConv3Addr - Return true if it is profitable to convert the
/// given 2-address instruction to a 3-address one.
bool
TwoAddressInstructionPass::isProfitableToConv3Addr(unsigned RegA) {
  // Look for situations like this:
  // %reg1024<def> = MOV r1
  // %reg1025<def> = MOV r0
  // %reg1026<def> = ADD %reg1024, %reg1025
  // r2            = MOV %reg1026
  // Turn ADD into a 3-address instruction to avoid a copy.
  unsigned FromRegA = getMappedReg(RegA, SrcRegMap);
  unsigned ToRegA = getMappedReg(RegA, DstRegMap);
  return (FromRegA && ToRegA && !regsAreCompatible(FromRegA, ToRegA, TRI));
}

/// ConvertInstTo3Addr - Convert the specified two-address instruction into a
/// three address one. Return true if this transformation was successful.
bool
TwoAddressInstructionPass::ConvertInstTo3Addr(MachineBasicBlock::iterator &mi,
                                              MachineBasicBlock::iterator &nmi,
                                              MachineFunction::iterator &mbbi,
                                              unsigned RegB, unsigned Dist) {
  MachineInstr *NewMI = TII->convertToThreeAddress(mbbi, mi, LV);
  if (NewMI) {
    DEBUG(errs() << "2addr: CONVERTING 2-ADDR: " << *mi);
    DEBUG(errs() << "2addr:         TO 3-ADDR: " << *NewMI);
    bool Sunk = false;

    if (NewMI->findRegisterUseOperand(RegB, false, TRI))
      // FIXME: Temporary workaround. If the new instruction doesn't
      // uses RegB, convertToThreeAddress must have created more
      // then one instruction.
      Sunk = Sink3AddrInstruction(mbbi, NewMI, RegB, mi);

    mbbi->erase(mi); // Nuke the old inst.

    if (!Sunk) {
      DistanceMap.insert(std::make_pair(NewMI, Dist));
      mi = NewMI;
      nmi = next(mi);
    }
    return true;
  }

  return false;
}

/// ProcessCopy - If the specified instruction is not yet processed, process it
/// if it's a copy. For a copy instruction, we find the physical registers the
/// source and destination registers might be mapped to. These are kept in
/// point-to maps used to determine future optimizations. e.g.
/// v1024 = mov r0
/// v1025 = mov r1
/// v1026 = add v1024, v1025
/// r1    = mov r1026
/// If 'add' is a two-address instruction, v1024, v1026 are both potentially
/// coalesced to r0 (from the input side). v1025 is mapped to r1. v1026 is
/// potentially joined with r1 on the output side. It's worthwhile to commute
/// 'add' to eliminate a copy.
void TwoAddressInstructionPass::ProcessCopy(MachineInstr *MI,
                                     MachineBasicBlock *MBB,
                                     SmallPtrSet<MachineInstr*, 8> &Processed) {
  if (Processed.count(MI))
    return;

  bool IsSrcPhys, IsDstPhys;
  unsigned SrcReg, DstReg;
  if (!isCopyToReg(*MI, TII, SrcReg, DstReg, IsSrcPhys, IsDstPhys))
    return;

  if (IsDstPhys && !IsSrcPhys)
    DstRegMap.insert(std::make_pair(SrcReg, DstReg));
  else if (!IsDstPhys && IsSrcPhys) {
    bool isNew = SrcRegMap.insert(std::make_pair(DstReg, SrcReg)).second;
    if (!isNew)
      assert(SrcRegMap[DstReg] == SrcReg &&
             "Can't map to two src physical registers!");

    SmallVector<unsigned, 4> VirtRegPairs;
    bool IsCopy = false;
    unsigned NewReg = 0;
    while (MachineInstr *UseMI = findOnlyInterestingUse(DstReg, MBB, MRI,TII,
                                                   IsCopy, NewReg, IsDstPhys)) {
      if (IsCopy) {
        if (!Processed.insert(UseMI))
          break;
      }

      DenseMap<MachineInstr*, unsigned>::iterator DI = DistanceMap.find(UseMI);
      if (DI != DistanceMap.end())
        // Earlier in the same MBB.Reached via a back edge.
        break;

      if (IsDstPhys) {
        VirtRegPairs.push_back(NewReg);
        break;
      }
      bool isNew = SrcRegMap.insert(std::make_pair(NewReg, DstReg)).second;
      if (!isNew)
        assert(SrcRegMap[NewReg] == DstReg &&
               "Can't map to two src physical registers!");
      VirtRegPairs.push_back(NewReg);
      DstReg = NewReg;
    }

    if (!VirtRegPairs.empty()) {
      unsigned ToReg = VirtRegPairs.back();
      VirtRegPairs.pop_back();
      while (!VirtRegPairs.empty()) {
        unsigned FromReg = VirtRegPairs.back();
        VirtRegPairs.pop_back();
        bool isNew = DstRegMap.insert(std::make_pair(FromReg, ToReg)).second;
        if (!isNew)
          assert(DstRegMap[FromReg] == ToReg &&
                 "Can't map to two dst physical registers!");
        ToReg = FromReg;
      }
    }
  }

  Processed.insert(MI);
}

/// isSafeToDelete - If the specified instruction does not produce any side
/// effects and all of its defs are dead, then it's safe to delete.
static bool isSafeToDelete(MachineInstr *MI, unsigned Reg,
                           const TargetInstrInfo *TII,
                           SmallVector<unsigned, 4> &Kills) {
  const TargetInstrDesc &TID = MI->getDesc();
  if (TID.mayStore() || TID.isCall())
    return false;
  if (TID.isTerminator() || TID.hasUnmodeledSideEffects())
    return false;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    if (MO.isDef() && !MO.isDead())
      return false;
    if (MO.isUse() && MO.getReg() != Reg && MO.isKill())
      Kills.push_back(MO.getReg());
  }

  return true;
}

/// canUpdateDeletedKills - Check if all the registers listed in Kills are
/// killed by instructions in MBB preceding the current instruction at
/// position Dist.  If so, return true and record information about the
/// preceding kills in NewKills.
bool TwoAddressInstructionPass::
canUpdateDeletedKills(SmallVector<unsigned, 4> &Kills,
                      SmallVector<NewKill, 4> &NewKills,
                      MachineBasicBlock *MBB, unsigned Dist) {
  while (!Kills.empty()) {
    unsigned Kill = Kills.back();
    Kills.pop_back();
    if (TargetRegisterInfo::isPhysicalRegister(Kill))
      return false;

    MachineInstr *LastKill = FindLastUseInMBB(Kill, MBB, Dist);
    if (!LastKill)
      return false;

    bool isModRef = LastKill->modifiesRegister(Kill);
    NewKills.push_back(std::make_pair(std::make_pair(Kill, isModRef),
                                      LastKill));
  }
  return true;
}

/// DeleteUnusedInstr - If an instruction with a tied register operand can
/// be safely deleted, just delete it.
bool
TwoAddressInstructionPass::DeleteUnusedInstr(MachineBasicBlock::iterator &mi,
                                             MachineBasicBlock::iterator &nmi,
                                             MachineFunction::iterator &mbbi,
                                             unsigned regB, unsigned regBIdx,
                                             unsigned Dist) {
  // Check if the instruction has no side effects and if all its defs are dead.
  SmallVector<unsigned, 4> Kills;
  if (!isSafeToDelete(mi, regB, TII, Kills))
    return false;

  // If this instruction kills some virtual registers, we need to
  // update the kill information. If it's not possible to do so,
  // then bail out.
  SmallVector<NewKill, 4> NewKills;
  if (!canUpdateDeletedKills(Kills, NewKills, &*mbbi, Dist))
    return false;

  if (LV) {
    while (!NewKills.empty()) {
      MachineInstr *NewKill = NewKills.back().second;
      unsigned Kill = NewKills.back().first.first;
      bool isDead = NewKills.back().first.second;
      NewKills.pop_back();
      if (LV->removeVirtualRegisterKilled(Kill, mi)) {
        if (isDead)
          LV->addVirtualRegisterDead(Kill, NewKill);
        else
          LV->addVirtualRegisterKilled(Kill, NewKill);
      }
    }

    // If regB was marked as a kill, update its Kills list.
    if (mi->getOperand(regBIdx).isKill())
      LV->removeVirtualRegisterKilled(regB, mi);
  }

  mbbi->erase(mi); // Nuke the old inst.
  mi = nmi;
  return true;
}

/// runOnMachineFunction - Reduce two-address instructions to two operands.
///
bool TwoAddressInstructionPass::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(errs() << "Machine Function\n");
  const TargetMachine &TM = MF.getTarget();
  MRI = &MF.getRegInfo();
  TII = TM.getInstrInfo();
  TRI = TM.getRegisterInfo();
  LV = getAnalysisIfAvailable<LiveVariables>();

  bool MadeChange = false;

  DEBUG(errs() << "********** REWRITING TWO-ADDR INSTRS **********\n");
  DEBUG(errs() << "********** Function: " 
        << MF.getFunction()->getName() << '\n');

  // ReMatRegs - Keep track of the registers whose def's are remat'ed.
  BitVector ReMatRegs;
  ReMatRegs.resize(MRI->getLastVirtReg()+1);

  SmallPtrSet<MachineInstr*, 8> Processed;
  for (MachineFunction::iterator mbbi = MF.begin(), mbbe = MF.end();
       mbbi != mbbe; ++mbbi) {
    unsigned Dist = 0;
    DistanceMap.clear();
    SrcRegMap.clear();
    DstRegMap.clear();
    Processed.clear();
    for (MachineBasicBlock::iterator mi = mbbi->begin(), me = mbbi->end();
         mi != me; ) {
      MachineBasicBlock::iterator nmi = next(mi);
      const TargetInstrDesc &TID = mi->getDesc();
      bool FirstTied = true;

      DistanceMap.insert(std::make_pair(mi, ++Dist));

      ProcessCopy(&*mi, &*mbbi, Processed);

      unsigned NumOps = (mi->getOpcode() == TargetInstrInfo::INLINEASM)
        ? mi->getNumOperands() : TID.getNumOperands();
      for (unsigned si = 0; si < NumOps; ++si) {
        unsigned ti = 0;
        if (!mi->isRegTiedToDefOperand(si, &ti))
          continue;

        if (FirstTied) {
          ++NumTwoAddressInstrs;
          DEBUG(errs() << '\t' << *mi);
        }

        FirstTied = false;

        assert(mi->getOperand(si).isReg() && mi->getOperand(si).getReg() &&
               mi->getOperand(si).isUse() && "two address instruction invalid");

        // If the two operands are the same, nothing needs to be done.
        if (mi->getOperand(ti).getReg() == mi->getOperand(si).getReg())
          continue;

        // Rewrite:
        //     a = b op c
        // to:
        //     a = b
        //     a = a op c
        unsigned regA = mi->getOperand(ti).getReg();
        unsigned regB = mi->getOperand(si).getReg();
        unsigned regASubIdx = mi->getOperand(ti).getSubReg();

        assert(TargetRegisterInfo::isVirtualRegister(regB) &&
               "cannot make instruction into two-address form");

#ifndef NDEBUG
        // First, verify that we don't have a use of a in the instruction (a =
        // b + a for example) because our transformation will not work. This
        // should never occur because we are in SSA form.
        for (unsigned i = 0; i != mi->getNumOperands(); ++i)
          assert(i == ti ||
                 !mi->getOperand(i).isReg() ||
                 mi->getOperand(i).getReg() != regA);
#endif

        // If this instruction is not the killing user of B, see if we can
        // rearrange the code to make it so.  Making it the killing user will
        // allow us to coalesce A and B together, eliminating the copy we are
        // about to insert.
        if (!isKilled(*mi, regB, MRI, TII)) {

          // If regA is dead and the instruction can be deleted, just delete
          // it so it doesn't clobber regB.
          if (mi->getOperand(ti).isDead() &&
              DeleteUnusedInstr(mi, nmi, mbbi, regB, si, Dist)) {
            ++NumDeletes;
            break; // Done with this instruction.
          }

          // If this instruction is commutative, check to see if C dies.  If
          // so, swap the B and C operands.  This makes the live ranges of A
          // and C joinable.
          // FIXME: This code also works for A := B op C instructions.
          unsigned SrcOp1, SrcOp2;
          if (TID.isCommutable() && mi->getNumOperands() >= 3 &&
              TII->findCommutedOpIndices(mi, SrcOp1, SrcOp2)) {
            unsigned regC = 0;
            if (si == SrcOp1)
              regC = mi->getOperand(SrcOp2).getReg();
            else if (si == SrcOp2)
              regC = mi->getOperand(SrcOp1).getReg();
            if (isKilled(*mi, regC, MRI, TII)) {
              if (CommuteInstruction(mi, mbbi, regB, regC, Dist)) {
                ++NumCommuted;
                regB = regC;
                goto InstructionRearranged;
              }
            }
          }

          // If this instruction is potentially convertible to a true
          // three-address instruction,
          if (TID.isConvertibleTo3Addr()) {
            // FIXME: This assumes there are no more operands which are tied
            // to another register.
#ifndef NDEBUG
            for (unsigned i = si + 1, e = TID.getNumOperands(); i < e; ++i)
              assert(TID.getOperandConstraint(i, TOI::TIED_TO) == -1);
#endif

            if (ConvertInstTo3Addr(mi, nmi, mbbi, regB, Dist)) {
              ++NumConvertedTo3Addr;
              break; // Done with this instruction.
            }
          }
        }

        // If it's profitable to commute the instruction, do so.
        unsigned SrcOp1, SrcOp2;
        if (TID.isCommutable() && mi->getNumOperands() >= 3 &&
            TII->findCommutedOpIndices(mi, SrcOp1, SrcOp2)) {
          unsigned regC = 0;
          if (si == SrcOp1)
            regC = mi->getOperand(SrcOp2).getReg();
          else if (si == SrcOp2)
            regC = mi->getOperand(SrcOp1).getReg();
            
          if (regC && isProfitableToCommute(regB, regC, mi, mbbi, Dist))
            if (CommuteInstruction(mi, mbbi, regB, regC, Dist)) {
              ++NumAggrCommuted;
              ++NumCommuted;
              regB = regC;
              goto InstructionRearranged;
            }
        }

        // If it's profitable to convert the 2-address instruction to a
        // 3-address one, do so.
        if (TID.isConvertibleTo3Addr() && isProfitableToConv3Addr(regA)) {
          if (ConvertInstTo3Addr(mi, nmi, mbbi, regB, Dist)) {
            ++NumConvertedTo3Addr;
            break; // Done with this instruction.
          }
        }

      InstructionRearranged:
        const TargetRegisterClass* rc = MRI->getRegClass(regB);
        MachineInstr *DefMI = MRI->getVRegDef(regB);
        // If it's safe and profitable, remat the definition instead of
        // copying it.
        if (DefMI &&
            DefMI->getDesc().isAsCheapAsAMove() &&
            DefMI->isSafeToReMat(TII, regB) &&
            isProfitableToReMat(regB, rc, mi, DefMI, mbbi, Dist)){
          DEBUG(errs() << "2addr: REMATTING : " << *DefMI << "\n");
          TII->reMaterialize(*mbbi, mi, regA, regASubIdx, DefMI);
          ReMatRegs.set(regB);
          ++NumReMats;
        } else {
          bool Emitted = TII->copyRegToReg(*mbbi, mi, regA, regB, rc, rc);
          (void)Emitted;
          assert(Emitted && "Unable to issue a copy instruction!\n");
        }

        MachineBasicBlock::iterator prevMI = prior(mi);
        // Update DistanceMap.
        DistanceMap.insert(std::make_pair(prevMI, Dist));
        DistanceMap[mi] = ++Dist;
          
        // Scan the operands to find: (1) the use operand that kills regB (if
        // any); (2) whether the kill operand is being replaced by regA on
        // this iteration; and (3) the first use of regB that is not being
        // replaced on this iteration.  A use of regB will not replaced if it
        // is tied to a different destination register and will be handled on
        // a later iteration.
        MachineOperand *KillMO = NULL;
        MachineOperand *FirstKeptMO = NULL;
        bool KillMOKept = false;
        for (unsigned i = 0, e = mi->getNumOperands(); i != e; ++i) {
          MachineOperand &MO = mi->getOperand(i);
          if (MO.isReg() && MO.getReg() == regB && MO.isUse()) {

            // Check if this operand is tied to a different destination.
            bool isKept = false;
            unsigned dsti = 0;
            if (mi->isRegTiedToDefOperand(i, &dsti) && dsti != ti) {
              isKept = true;
              if (!FirstKeptMO)
                FirstKeptMO = &MO;
            }

            if (MO.isKill()) {
              KillMO = &MO;
              KillMOKept = isKept;
            }
          }
        }

        // Update live variables for regB.
        if (KillMO) {
          if (!FirstKeptMO) {
            // All uses of regB are being replaced; move the kill to prevMI.
            if (LV && LV->removeVirtualRegisterKilled(regB, mi))
              LV->addVirtualRegisterKilled(regB, prevMI);
          } else {
            if (!KillMOKept) {
              // The kill marker is on an operand being replaced, but there
              // are other uses of regB remaining.  Move the kill marker to
              // one of them.
              KillMO->setIsKill(false);
              FirstKeptMO->setIsKill(true);
            }
          }
        }

        DEBUG(errs() << "\t\tprepend:\t" << *prevMI);

        // Replace uses of regB with regA.
        for (unsigned i = 0, e = mi->getNumOperands(); i != e; ++i) {
          MachineOperand &MO = mi->getOperand(i);
          if (MO.isReg() && MO.getReg() == regB && MO.isUse()) {

            // Skip operands that are tied to other register definitions.
            unsigned dsti = 0;
            if (mi->isRegTiedToDefOperand(i, &dsti) && dsti != ti)
              continue;

            MO.setReg(regA);
          }
        }

        assert(mi->getOperand(ti).isDef() && mi->getOperand(si).isUse());
        mi->getOperand(ti).setReg(mi->getOperand(si).getReg());
        MadeChange = true;

        DEBUG(errs() << "\t\trewrite to:\t" << *mi);
      }

      mi = nmi;
    }
  }

  // Some remat'ed instructions are dead.
  int VReg = ReMatRegs.find_first();
  while (VReg != -1) {
    if (MRI->use_empty(VReg)) {
      MachineInstr *DefMI = MRI->getVRegDef(VReg);
      DefMI->eraseFromParent();
    }
    VReg = ReMatRegs.find_next(VReg);
  }

  return MadeChange;
}
