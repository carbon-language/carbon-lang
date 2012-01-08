//===- MachineCopyPropagation.cpp - Machine Copy Propagation Pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an extremely simple MachineInstr-level copy propagation pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "codegen-cp"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumDeletes, "Number of dead copies deleted");

namespace {
  class MachineCopyPropagation : public MachineFunctionPass {
    const TargetRegisterInfo *TRI;
    BitVector ReservedRegs;
    
  public:
    static char ID; // Pass identification, replacement for typeid
    MachineCopyPropagation() : MachineFunctionPass(ID) {
     initializeMachineCopyPropagationPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);

  private:
    void SourceNoLongerAvailable(unsigned Reg,
                               DenseMap<unsigned, unsigned> &SrcMap,
                               DenseMap<unsigned, MachineInstr*> &AvailCopyMap);
    bool CopyPropagateBlock(MachineBasicBlock &MBB);
  };
}
char MachineCopyPropagation::ID = 0;

INITIALIZE_PASS(MachineCopyPropagation, "machine-cp",
                "Machine Copy Propagation Pass", false, false)

FunctionPass *llvm::createMachineCopyPropagationPass() {
  return new MachineCopyPropagation();
}

void
MachineCopyPropagation::SourceNoLongerAvailable(unsigned Reg,
                              DenseMap<unsigned, unsigned> &SrcMap,
                              DenseMap<unsigned, MachineInstr*> &AvailCopyMap) {
  DenseMap<unsigned, unsigned>::iterator SI = SrcMap.find(Reg);
  if (SI != SrcMap.end()) {
    unsigned MappedDef = SI->second;
    // Source of copy is no longer available for propagation.
    if (AvailCopyMap.erase(MappedDef)) {
      for (const unsigned *SR = TRI->getSubRegisters(MappedDef); *SR; ++SR)
        AvailCopyMap.erase(*SR);
    }
  }
  for (const unsigned *AS = TRI->getAliasSet(Reg); *AS; ++AS) {
    SI = SrcMap.find(*AS);
    if (SI != SrcMap.end()) {
      unsigned MappedDef = SI->second;
      if (AvailCopyMap.erase(MappedDef)) {
        for (const unsigned *SR = TRI->getSubRegisters(MappedDef); *SR; ++SR)
          AvailCopyMap.erase(*SR);
      }
    }
  }
}

static bool NoInterveningSideEffect(const MachineInstr *CopyMI,
                                    const MachineInstr *MI) {
  const MachineBasicBlock *MBB = CopyMI->getParent();
  if (MI->getParent() != MBB)
    return false;
  MachineBasicBlock::const_iterator I = CopyMI;
  MachineBasicBlock::const_iterator E = MBB->end();
  MachineBasicBlock::const_iterator E2 = MI;

  ++I;
  while (I != E && I != E2) {
    if (I->hasUnmodeledSideEffects() || I->isCall() ||
        I->isTerminator())
      return false;
    ++I;
  }
  return true;
}

bool MachineCopyPropagation::CopyPropagateBlock(MachineBasicBlock &MBB) {
  SmallSetVector<MachineInstr*, 8> MaybeDeadCopies; // Candidates for deletion
  DenseMap<unsigned, MachineInstr*> AvailCopyMap;   // Def -> available copies map
  DenseMap<unsigned, MachineInstr*> CopyMap;        // Def -> copies map
  DenseMap<unsigned, unsigned> SrcMap;              // Src -> Def map

  bool Changed = false;
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ) {
    MachineInstr *MI = &*I;
    ++I;

    if (MI->isCopy()) {
      unsigned Def = MI->getOperand(0).getReg();
      unsigned Src = MI->getOperand(1).getReg();

      if (TargetRegisterInfo::isVirtualRegister(Def) ||
          TargetRegisterInfo::isVirtualRegister(Src))
        report_fatal_error("MachineCopyPropagation should be run after"
                           " register allocation!");

      DenseMap<unsigned, MachineInstr*>::iterator CI = AvailCopyMap.find(Src);
      if (CI != AvailCopyMap.end()) {
        MachineInstr *CopyMI = CI->second;
        unsigned SrcSrc = CopyMI->getOperand(1).getReg();
        if (!ReservedRegs.test(Def) &&
            (!ReservedRegs.test(Src) || NoInterveningSideEffect(CopyMI, MI)) &&
            (SrcSrc == Def || TRI->isSubRegister(SrcSrc, Def))) {
          // The two copies cancel out and the source of the first copy
          // hasn't been overridden, eliminate the second one. e.g.
          //  %ECX<def> = COPY %EAX<kill>
          //  ... nothing clobbered EAX.
          //  %EAX<def> = COPY %ECX
          // =>
          //  %ECX<def> = COPY %EAX
          //
          // Also avoid eliminating a copy from reserved registers unless the
          // definition is proven not clobbered. e.g.
          // %RSP<def> = COPY %RAX
          // CALL
          // %RAX<def> = COPY %RSP
          CopyMI->getOperand(1).setIsKill(false);
          MI->eraseFromParent();
          Changed = true;
          ++NumDeletes;
          continue;
        }
      }

      // If Src is defined by a previous copy, it cannot be eliminated.
      CI = CopyMap.find(Src);
      if (CI != CopyMap.end())
        MaybeDeadCopies.remove(CI->second);
      for (const unsigned *AS = TRI->getAliasSet(Src); *AS; ++AS) {
        CI = CopyMap.find(*AS);
        if (CI != CopyMap.end())
          MaybeDeadCopies.remove(CI->second);
      }

      // Copy is now a candidate for deletion.
      MaybeDeadCopies.insert(MI);

      // If 'Src' is previously source of another copy, then this earlier copy's
      // source is no longer available. e.g.
      // %xmm9<def> = copy %xmm2
      // ...
      // %xmm2<def> = copy %xmm0
      // ...
      // %xmm2<def> = copy %xmm9
      SourceNoLongerAvailable(Def, SrcMap, AvailCopyMap);

      // Remember Def is defined by the copy.
      CopyMap[Def] = MI;
      AvailCopyMap[Def] = MI;
      for (const unsigned *SR = TRI->getSubRegisters(Def); *SR; ++SR) {
        CopyMap[*SR] = MI;
        AvailCopyMap[*SR] = MI;
      }

      // Remember source that's copied to Def. Once it's clobbered, then
      // it's no longer available for copy propagation.
      SrcMap[Src] = Def;

      continue;
    }

    // Not a copy.
    SmallVector<unsigned, 2> Defs;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;

      if (TargetRegisterInfo::isVirtualRegister(Reg))
        report_fatal_error("MachineCopyPropagation should be run after"
                           " register allocation!");

      if (MO.isDef()) {
        Defs.push_back(Reg);
        continue;
      }

      // If 'Reg' is defined by a copy, the copy is no longer a candidate
      // for elimination.
      DenseMap<unsigned, MachineInstr*>::iterator CI = CopyMap.find(Reg);
      if (CI != CopyMap.end())
        MaybeDeadCopies.remove(CI->second);
      for (const unsigned *AS = TRI->getAliasSet(Reg); *AS; ++AS) {
        CI = CopyMap.find(*AS);
        if (CI != CopyMap.end())
          MaybeDeadCopies.remove(CI->second);
      }
    }

    for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
      unsigned Reg = Defs[i];

      // No longer defined by a copy.
      CopyMap.erase(Reg);
      AvailCopyMap.erase(Reg);
      for (const unsigned *AS = TRI->getAliasSet(Reg); *AS; ++AS) {
        CopyMap.erase(*AS);
        AvailCopyMap.erase(*AS);
      }

      // If 'Reg' is previously source of a copy, it is no longer available for
      // copy propagation.
      SourceNoLongerAvailable(Reg, SrcMap, AvailCopyMap);
    }
  }

  // If MBB doesn't have successors, delete the copies whose defs are not used.
  // If MBB does have successors, then conservative assume the defs are live-out
  // since we don't want to trust live-in lists.
  if (MBB.succ_empty()) {
    for (SmallSetVector<MachineInstr*, 8>::iterator
           DI = MaybeDeadCopies.begin(), DE = MaybeDeadCopies.end();
         DI != DE; ++DI) {
      if (!ReservedRegs.test((*DI)->getOperand(0).getReg())) {
        (*DI)->eraseFromParent();
        Changed = true;
        ++NumDeletes;
      }
    }
  }

  return Changed;
}

bool MachineCopyPropagation::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  TRI = MF.getTarget().getRegisterInfo();
  ReservedRegs = TRI->getReservedRegs(MF);

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    Changed |= CopyPropagateBlock(*I);

  return Changed;
}
