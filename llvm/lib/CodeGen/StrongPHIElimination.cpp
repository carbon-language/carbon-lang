//===- StrongPhiElimination.cpp - Eliminate PHI nodes by inserting copies -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "strongphielim"
#include "PHIEliminationUtils.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

namespace {
  class StrongPHIElimination : public MachineFunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid
    StrongPHIElimination() : MachineFunctionPass(ID) {
      initializeStrongPHIEliminationPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage&) const;
    bool runOnMachineFunction(MachineFunction&);

  private:
    void InsertCopiesForPHI(MachineInstr*, MachineBasicBlock*);

    MachineRegisterInfo* MRI;
    const TargetInstrInfo* TII;
    LiveIntervals* LI;

    typedef DenseSet<std::pair<MachineBasicBlock*, unsigned> > CopySet;
    CopySet InsertedCopies;
  };
} // namespace

char StrongPHIElimination::ID = 0;
INITIALIZE_PASS_BEGIN(StrongPHIElimination, "strong-phi-node-elimination",
  "Eliminate PHI nodes for register allocation, intelligently", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(StrongPHIElimination, "strong-phi-node-elimination",
  "Eliminate PHI nodes for register allocation, intelligently", false, false)

char &llvm::StrongPHIEliminationID = StrongPHIElimination::ID;

void StrongPHIElimination::getAnalysisUsage(AnalysisUsage& AU) const {
  AU.setPreservesCFG();
  AU.addRequired<MachineDominatorTree>();
  AU.addRequired<SlotIndexes>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static MachineOperand* findLastUse(MachineBasicBlock* MBB, unsigned Reg) {
  // FIXME: This only needs to check from the first terminator, as only the
  // first terminator can use a virtual register.
  for (MachineBasicBlock::reverse_iterator RI = MBB->rbegin(); ; ++RI) {
    assert (RI != MBB->rend());
    MachineInstr* MI = &*RI;

    for (MachineInstr::mop_iterator OI = MI->operands_begin(),
         OE = MI->operands_end(); OI != OE; ++OI) {
      MachineOperand& MO = *OI;
      if (MO.isReg() && MO.isUse() && MO.getReg() == Reg)
        return &MO;
    }
  }
  return NULL;
}

bool StrongPHIElimination::runOnMachineFunction(MachineFunction& MF) {
  MRI = &MF.getRegInfo();
  TII = MF.getTarget().getInstrInfo();
  LI = &getAnalysis<LiveIntervals>();

  // Insert copies for all PHI source and destination registers.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    for (MachineBasicBlock::iterator BBI = I->begin(), BBE = I->end();
         BBI != BBE && BBI->isPHI(); ++BBI) {
      InsertCopiesForPHI(BBI, I);
    }
  }

  // Adjust the live intervals of all PHI source registers to handle the case
  // where the PHIs in successor blocks were the only later uses of the source
  // register.
  for (CopySet::iterator I = InsertedCopies.begin(), E = InsertedCopies.end();
       I != E; ++I) {
    MachineBasicBlock* MBB = I->first;
    unsigned SrcReg = I->second;
    LiveInterval& SrcLI = LI->getInterval(SrcReg);

    bool isLiveOut = false;
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI) {
      if (SrcLI.liveAt(LI->getMBBStartIdx(*SI))) {
        isLiveOut = true;
        break;
      }
    }

    if (isLiveOut)
      continue;

    MachineOperand* LastUse = findLastUse(MBB, SrcReg);
    assert(LastUse);
    SrcLI.removeRange(LI->getInstructionIndex(LastUse->getParent()).getDefIndex(),
                      LI->getMBBEndIdx(MBB));
    LastUse->setIsKill(true);
  }

  // Remove all PHI instructions from the function.
  bool Changed = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    MachineBasicBlock::iterator BBI = I->begin(), BBE = I->end();
    while (BBI != BBE && BBI->isPHI()) {
      MachineInstr* PHI = BBI;
      ++BBI;
      LI->RemoveMachineInstrFromMaps(PHI);
      PHI->eraseFromParent();
      Changed = true;
    }
  }

  LI->renumber();
  MF.verify(this);

  InsertedCopies.clear();

  return Changed;
}

void StrongPHIElimination::InsertCopiesForPHI(MachineInstr* PHI,
                                              MachineBasicBlock* MBB) {
  assert(PHI->isPHI());
  unsigned DestReg = PHI->getOperand(0).getReg();

  const TargetRegisterClass* RC = MRI->getRegClass(DestReg);
  unsigned CopyReg = MRI->createVirtualRegister(RC);

  MachineInstr* CopyInstr = BuildMI(*MBB,
                                    MBB->SkipPHIsAndLabels(MBB->begin()),
                                    PHI->getDebugLoc(),
                                    TII->get(TargetOpcode::COPY),
                                    DestReg).addReg(CopyReg);
  LI->InsertMachineInstrInMaps(CopyInstr);
  CopyInstr->getOperand(1).setIsKill(true);

  // Add the region from the beginning of MBB to the copy instruction to
  // CopyReg's live interval, and give the VNInfo the phidef flag.
  LiveInterval& CopyLI = LI->getOrCreateInterval(CopyReg);
  SlotIndex MBBStartIndex = LI->getMBBStartIdx(MBB);
  SlotIndex DestCopyIndex = LI->getInstructionIndex(CopyInstr);
  VNInfo* CopyVNI = CopyLI.getNextValue(MBBStartIndex,
                                        CopyInstr,
                                        LI->getVNInfoAllocator());
  CopyVNI->setIsPHIDef(true);
  CopyLI.addRange(LiveRange(MBBStartIndex,
                            DestCopyIndex.getDefIndex(),
                            CopyVNI));

  // Adjust DestReg's live interval to adjust for its new definition at
  // CopyInstr.
  LiveInterval& DestLI = LI->getOrCreateInterval(DestReg);
  SlotIndex PHIIndex = LI->getInstructionIndex(PHI);
  DestLI.removeRange(PHIIndex.getDefIndex(), DestCopyIndex.getDefIndex());

  VNInfo* DestVNI = DestLI.getVNInfoAt(DestCopyIndex.getDefIndex());
  assert(DestVNI);
  DestVNI->def = DestCopyIndex.getDefIndex();

  SmallPtrSet<MachineBasicBlock*, 8> MBBsInsertedInto;
  for (unsigned i = 1; i < PHI->getNumOperands(); i += 2) {
    MachineOperand& SrcMO = PHI->getOperand(i);

    // If a source is defined by an implicit def, there is no need to insert a
    // copy in the predecessor.
    if (SrcMO.isUndef())
      continue;

    unsigned SrcReg = SrcMO.getReg();
    unsigned SrcSubReg = SrcMO.getSubReg();

    assert(TargetRegisterInfo::isVirtualRegister(SrcReg) &&
           "Machine PHI Operands must all be virtual registers!");

    MachineBasicBlock* PredBB = PHI->getOperand(i + 1).getMBB();

    // A copy may have already been inserted in the predecessor in the case of a
    // block with multiple incoming edges.
    if (!MBBsInsertedInto.insert(PredBB))
      continue;

    MachineBasicBlock::iterator
      CopyInsertPoint = findPHICopyInsertPoint(PredBB, MBB, SrcReg);
    MachineInstr* CopyInstr = BuildMI(*PredBB,
                                      CopyInsertPoint,
                                      PHI->getDebugLoc(),
                                      TII->get(TargetOpcode::COPY),
                                      CopyReg).addReg(SrcReg, 0, SrcSubReg);
    LI->InsertMachineInstrInMaps(CopyInstr);

    // addLiveRangeToEndOfBlock() also adds the phikill flag to the VNInfo for
    // the newly added range.
    LI->addLiveRangeToEndOfBlock(CopyReg, CopyInstr);
    InsertedCopies.insert(std::make_pair(PredBB, SrcReg));

    // If SrcReg is not live beyond the PHI, trim its interval so that it is no
    // longer live-in to MBB. Note that SrcReg may appear in other PHIs that are
    // processed later, but this is still correct to do at this point because we
    // never rely on LiveIntervals being correct while inserting copies.
    // FIXME: Should this just count uses at PHIs like the normal PHIElimination
    // pass does?
    LiveInterval& SrcLI = LI->getInterval(SrcReg);
    SlotIndex MBBStartIndex = LI->getMBBStartIdx(MBB);
    SlotIndex PHIIndex = LI->getInstructionIndex(PHI);
    SlotIndex NextInstrIndex = PHIIndex.getNextIndex();
    if (SrcLI.liveAt(MBBStartIndex) && SrcLI.expiredAt(NextInstrIndex))
      SrcLI.removeRange(MBBStartIndex, PHIIndex, true);
  }
}
