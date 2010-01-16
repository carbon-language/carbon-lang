//===-- TailDuplication.cpp - Duplicate blocks into predecessors' tails ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass duplicates basic blocks ending in unconditional branches into
// the tails of their predecessors.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "tailduplication"
#include "llvm/Function.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumTails     , "Number of tails duplicated");
STATISTIC(NumTailDups  , "Number of tail duplicated blocks");
STATISTIC(NumInstrDups , "Additional instructions due to tail duplication");
STATISTIC(NumDeadBlocks, "Number of dead blocks removed");

// Heuristic for tail duplication.
static cl::opt<unsigned>
TailDuplicateSize("tail-dup-size",
                  cl::desc("Maximum instructions to consider tail duplicating"),
                  cl::init(2), cl::Hidden);

static cl::opt<bool>
TailDupVerify("tail-dup-verify",
              cl::desc("Verify sanity of PHI instructions during taildup"),
              cl::init(false), cl::Hidden);

static cl::opt<unsigned>
TailDupLimit("tail-dup-limit", cl::init(~0U), cl::Hidden);

typedef std::vector<std::pair<MachineBasicBlock*,unsigned> > AvailableValsTy;

namespace {
  /// TailDuplicatePass - Perform tail duplication.
  class TailDuplicatePass : public MachineFunctionPass {
    bool PreRegAlloc;
    const TargetInstrInfo *TII;
    MachineModuleInfo *MMI;
    MachineRegisterInfo *MRI;

    // SSAUpdateVRs - A list of virtual registers for which to update SSA form.
    SmallVector<unsigned, 16> SSAUpdateVRs;

    // SSAUpdateVals - For each virtual register in SSAUpdateVals keep a list of
    // source virtual registers.
    DenseMap<unsigned, AvailableValsTy> SSAUpdateVals;

  public:
    static char ID;
    explicit TailDuplicatePass(bool PreRA) :
      MachineFunctionPass(&ID), PreRegAlloc(PreRA) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const { return "Tail Duplication"; }

  private:
    void AddSSAUpdateEntry(unsigned OrigReg, unsigned NewReg,
                           MachineBasicBlock *BB);
    void ProcessPHI(MachineInstr *MI, MachineBasicBlock *TailBB,
                    MachineBasicBlock *PredBB,
                    DenseMap<unsigned, unsigned> &LocalVRMap,
                    SmallVector<std::pair<unsigned,unsigned>, 4> &Copies);
    void DuplicateInstruction(MachineInstr *MI,
                              MachineBasicBlock *TailBB,
                              MachineBasicBlock *PredBB,
                              MachineFunction &MF,
                              DenseMap<unsigned, unsigned> &LocalVRMap);
    void UpdateSuccessorsPHIs(MachineBasicBlock *FromBB, bool isDead,
                              SmallVector<MachineBasicBlock*, 8> &TDBBs,
                              SmallSetVector<MachineBasicBlock*, 8> &Succs);
    bool TailDuplicateBlocks(MachineFunction &MF);
    bool TailDuplicate(MachineBasicBlock *TailBB, MachineFunction &MF,
                       SmallVector<MachineBasicBlock*, 8> &TDBBs,
                       SmallVector<MachineInstr*, 16> &Copies);
    void RemoveDeadBlock(MachineBasicBlock *MBB);
  };

  char TailDuplicatePass::ID = 0;
}

FunctionPass *llvm::createTailDuplicatePass(bool PreRegAlloc) {
  return new TailDuplicatePass(PreRegAlloc);
}

bool TailDuplicatePass::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  MRI = &MF.getRegInfo();
  MMI = getAnalysisIfAvailable<MachineModuleInfo>();

  bool MadeChange = false;
  while (TailDuplicateBlocks(MF))
    MadeChange = true;

  return MadeChange;
}

static void VerifyPHIs(MachineFunction &MF, bool CheckExtra) {
  for (MachineFunction::iterator I = ++MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = I;
    SmallSetVector<MachineBasicBlock*, 8> Preds(MBB->pred_begin(),
                                                MBB->pred_end());
    MachineBasicBlock::iterator MI = MBB->begin();
    while (MI != MBB->end()) {
      if (MI->getOpcode() != TargetInstrInfo::PHI)
        break;
      for (SmallSetVector<MachineBasicBlock *, 8>::iterator PI = Preds.begin(),
             PE = Preds.end(); PI != PE; ++PI) {
        MachineBasicBlock *PredBB = *PI;
        bool Found = false;
        for (unsigned i = 1, e = MI->getNumOperands(); i != e; i += 2) {
          MachineBasicBlock *PHIBB = MI->getOperand(i+1).getMBB();
          if (PHIBB == PredBB) {
            Found = true;
            break;
          }
        }
        if (!Found) {
          dbgs() << "Malformed PHI in BB#" << MBB->getNumber() << ": " << *MI;
          dbgs() << "  missing input from predecessor BB#"
                 << PredBB->getNumber() << '\n';
          llvm_unreachable(0);
        }
      }

      for (unsigned i = 1, e = MI->getNumOperands(); i != e; i += 2) {
        MachineBasicBlock *PHIBB = MI->getOperand(i+1).getMBB();
        if (CheckExtra && !Preds.count(PHIBB)) {
          // This is not a hard error.
          dbgs() << "Warning: malformed PHI in BB#" << MBB->getNumber()
                 << ": " << *MI;
          dbgs() << "  extra input from predecessor BB#"
                 << PHIBB->getNumber() << '\n';
        }
        if (PHIBB->getNumber() < 0) {
          dbgs() << "Malformed PHI in BB#" << MBB->getNumber() << ": " << *MI;
          dbgs() << "  non-existing BB#" << PHIBB->getNumber() << '\n';
          llvm_unreachable(0);
        }
      }
      ++MI;
    }
  }
}

/// TailDuplicateBlocks - Look for small blocks that are unconditionally
/// branched to and do not fall through. Tail-duplicate their instructions
/// into their predecessors to eliminate (dynamic) branches.
bool TailDuplicatePass::TailDuplicateBlocks(MachineFunction &MF) {
  bool MadeChange = false;

  if (PreRegAlloc && TailDupVerify) {
    DEBUG(dbgs() << "\n*** Before tail-duplicating\n");
    VerifyPHIs(MF, true);
  }

  SmallVector<MachineInstr*, 8> NewPHIs;
  MachineSSAUpdater SSAUpdate(MF, &NewPHIs);

  for (MachineFunction::iterator I = ++MF.begin(), E = MF.end(); I != E; ) {
    MachineBasicBlock *MBB = I++;

    if (NumTails == TailDupLimit)
      break;

    // Only duplicate blocks that end with unconditional branches.
    if (MBB->canFallThrough())
      continue;

    // Save the successors list.
    SmallSetVector<MachineBasicBlock*, 8> Succs(MBB->succ_begin(),
                                                MBB->succ_end());

    SmallVector<MachineBasicBlock*, 8> TDBBs;
    SmallVector<MachineInstr*, 16> Copies;
    if (TailDuplicate(MBB, MF, TDBBs, Copies)) {
      ++NumTails;

      // TailBB's immediate successors are now successors of those predecessors
      // which duplicated TailBB. Add the predecessors as sources to the PHI
      // instructions.
      bool isDead = MBB->pred_empty();
      if (PreRegAlloc)
        UpdateSuccessorsPHIs(MBB, isDead, TDBBs, Succs);

      // If it is dead, remove it.
      if (isDead) {
        NumInstrDups -= MBB->size();
        RemoveDeadBlock(MBB);
        ++NumDeadBlocks;
      }

      // Update SSA form.
      if (!SSAUpdateVRs.empty()) {
        for (unsigned i = 0, e = SSAUpdateVRs.size(); i != e; ++i) {
          unsigned VReg = SSAUpdateVRs[i];
          SSAUpdate.Initialize(VReg);

          // If the original definition is still around, add it as an available
          // value.
          MachineInstr *DefMI = MRI->getVRegDef(VReg);
          MachineBasicBlock *DefBB = 0;
          if (DefMI) {
            DefBB = DefMI->getParent();
            SSAUpdate.AddAvailableValue(DefBB, VReg);
          }

          // Add the new vregs as available values.
          DenseMap<unsigned, AvailableValsTy>::iterator LI =
            SSAUpdateVals.find(VReg);  
          for (unsigned j = 0, ee = LI->second.size(); j != ee; ++j) {
            MachineBasicBlock *SrcBB = LI->second[j].first;
            unsigned SrcReg = LI->second[j].second;
            SSAUpdate.AddAvailableValue(SrcBB, SrcReg);
          }

          // Rewrite uses that are outside of the original def's block.
          MachineRegisterInfo::use_iterator UI = MRI->use_begin(VReg);
          while (UI != MRI->use_end()) {
            MachineOperand &UseMO = UI.getOperand();
            MachineInstr *UseMI = &*UI;
            ++UI;
            if (UseMI->getParent() == DefBB)
              continue;
            SSAUpdate.RewriteUse(UseMO);
          }
        }

        SSAUpdateVRs.clear();
        SSAUpdateVals.clear();
      }

      // Eliminate some of the copies inserted by tail duplication to maintain
      // SSA form.
      for (unsigned i = 0, e = Copies.size(); i != e; ++i) {
        MachineInstr *Copy = Copies[i];
        unsigned Src, Dst, SrcSR, DstSR;
        if (TII->isMoveInstr(*Copy, Src, Dst, SrcSR, DstSR)) {
          MachineRegisterInfo::use_iterator UI = MRI->use_begin(Src);
          if (++UI == MRI->use_end()) {
            // Copy is the only use. Do trivial copy propagation here.
            MRI->replaceRegWith(Dst, Src);
            Copy->eraseFromParent();
          }
        }
      }

      if (PreRegAlloc && TailDupVerify)
        VerifyPHIs(MF, false);
      MadeChange = true;
    }
  }

  return MadeChange;
}

static bool isDefLiveOut(unsigned Reg, MachineBasicBlock *BB,
                         const MachineRegisterInfo *MRI) {
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(Reg),
         UE = MRI->use_end(); UI != UE; ++UI) {
    MachineInstr *UseMI = &*UI;
    if (UseMI->getParent() != BB)
      return true;
  }
  return false;
}

static unsigned getPHISrcRegOpIdx(MachineInstr *MI, MachineBasicBlock *SrcBB) {
  for (unsigned i = 1, e = MI->getNumOperands(); i != e; i += 2)
    if (MI->getOperand(i+1).getMBB() == SrcBB)
      return i;
  return 0;
}

/// AddSSAUpdateEntry - Add a definition and source virtual registers pair for
/// SSA update.
void TailDuplicatePass::AddSSAUpdateEntry(unsigned OrigReg, unsigned NewReg,
                                          MachineBasicBlock *BB) {
  DenseMap<unsigned, AvailableValsTy>::iterator LI= SSAUpdateVals.find(OrigReg);
  if (LI != SSAUpdateVals.end())
    LI->second.push_back(std::make_pair(BB, NewReg));
  else {
    AvailableValsTy Vals;
    Vals.push_back(std::make_pair(BB, NewReg));
    SSAUpdateVals.insert(std::make_pair(OrigReg, Vals));
    SSAUpdateVRs.push_back(OrigReg);
  }
}

/// ProcessPHI - Process PHI node in TailBB by turning it into a copy in PredBB.
/// Remember the source register that's contributed by PredBB and update SSA
/// update map.
void TailDuplicatePass::ProcessPHI(MachineInstr *MI,
                                   MachineBasicBlock *TailBB,
                                   MachineBasicBlock *PredBB,
                                   DenseMap<unsigned, unsigned> &LocalVRMap,
                         SmallVector<std::pair<unsigned,unsigned>, 4> &Copies) {
  unsigned DefReg = MI->getOperand(0).getReg();
  unsigned SrcOpIdx = getPHISrcRegOpIdx(MI, PredBB);
  assert(SrcOpIdx && "Unable to find matching PHI source?");
  unsigned SrcReg = MI->getOperand(SrcOpIdx).getReg();
  const TargetRegisterClass *RC = MRI->getRegClass(DefReg);
  LocalVRMap.insert(std::make_pair(DefReg, SrcReg));

  // Insert a copy from source to the end of the block. The def register is the
  // available value liveout of the block.
  unsigned NewDef = MRI->createVirtualRegister(RC);
  Copies.push_back(std::make_pair(NewDef, SrcReg));
  if (isDefLiveOut(DefReg, TailBB, MRI))
    AddSSAUpdateEntry(DefReg, NewDef, PredBB);

  // Remove PredBB from the PHI node.
  MI->RemoveOperand(SrcOpIdx+1);
  MI->RemoveOperand(SrcOpIdx);
  if (MI->getNumOperands() == 1)
    MI->eraseFromParent();
}

/// DuplicateInstruction - Duplicate a TailBB instruction to PredBB and update
/// the source operands due to earlier PHI translation.
void TailDuplicatePass::DuplicateInstruction(MachineInstr *MI,
                                     MachineBasicBlock *TailBB,
                                     MachineBasicBlock *PredBB,
                                     MachineFunction &MF,
                                     DenseMap<unsigned, unsigned> &LocalVRMap) {
  MachineInstr *NewMI = TII->duplicate(MI, MF);
  for (unsigned i = 0, e = NewMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = NewMI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg || TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    if (MO.isDef()) {
      const TargetRegisterClass *RC = MRI->getRegClass(Reg);
      unsigned NewReg = MRI->createVirtualRegister(RC);
      MO.setReg(NewReg);
      LocalVRMap.insert(std::make_pair(Reg, NewReg));
      if (isDefLiveOut(Reg, TailBB, MRI))
        AddSSAUpdateEntry(Reg, NewReg, PredBB);
    } else {
      DenseMap<unsigned, unsigned>::iterator VI = LocalVRMap.find(Reg);
      if (VI != LocalVRMap.end())
        MO.setReg(VI->second);
    }
  }
  PredBB->insert(PredBB->end(), NewMI);
}

/// UpdateSuccessorsPHIs - After FromBB is tail duplicated into its predecessor
/// blocks, the successors have gained new predecessors. Update the PHI
/// instructions in them accordingly.
void
TailDuplicatePass::UpdateSuccessorsPHIs(MachineBasicBlock *FromBB, bool isDead,
                                  SmallVector<MachineBasicBlock*, 8> &TDBBs,
                                  SmallSetVector<MachineBasicBlock*,8> &Succs) {
  for (SmallSetVector<MachineBasicBlock*, 8>::iterator SI = Succs.begin(),
         SE = Succs.end(); SI != SE; ++SI) {
    MachineBasicBlock *SuccBB = *SI;
    for (MachineBasicBlock::iterator II = SuccBB->begin(), EE = SuccBB->end();
         II != EE; ++II) {
      if (II->getOpcode() != TargetInstrInfo::PHI)
        break;
      unsigned Idx = 0;
      for (unsigned i = 1, e = II->getNumOperands(); i != e; i += 2) {
        MachineOperand &MO = II->getOperand(i+1);
        if (MO.getMBB() == FromBB) {
          Idx = i;
          break;
        }
      }

      assert(Idx != 0);
      MachineOperand &MO0 = II->getOperand(Idx);
      unsigned Reg = MO0.getReg();
      if (isDead) {
        // Folded into the previous BB.
        // There could be duplicate phi source entries. FIXME: Should sdisel
        // or earlier pass fixed this?
        for (unsigned i = II->getNumOperands()-2; i != Idx; i -= 2) {
          MachineOperand &MO = II->getOperand(i+1);
          if (MO.getMBB() == FromBB) {
            II->RemoveOperand(i+1);
            II->RemoveOperand(i);
          }
        }
        II->RemoveOperand(Idx+1);
        II->RemoveOperand(Idx);
      }
      DenseMap<unsigned,AvailableValsTy>::iterator LI=SSAUpdateVals.find(Reg);
      if (LI != SSAUpdateVals.end()) {
        // This register is defined in the tail block.
        for (unsigned j = 0, ee = LI->second.size(); j != ee; ++j) {
          MachineBasicBlock *SrcBB = LI->second[j].first;
          unsigned SrcReg = LI->second[j].second;
          II->addOperand(MachineOperand::CreateReg(SrcReg, false));
          II->addOperand(MachineOperand::CreateMBB(SrcBB));
        }
      } else {
        // Live in tail block, must also be live in predecessors.
        for (unsigned j = 0, ee = TDBBs.size(); j != ee; ++j) {
          MachineBasicBlock *SrcBB = TDBBs[j];
          II->addOperand(MachineOperand::CreateReg(Reg, false));
          II->addOperand(MachineOperand::CreateMBB(SrcBB));
        }
      }
    }
  }
}

/// TailDuplicate - If it is profitable, duplicate TailBB's contents in each
/// of its predecessors.
bool
TailDuplicatePass::TailDuplicate(MachineBasicBlock *TailBB, MachineFunction &MF,
                                 SmallVector<MachineBasicBlock*, 8> &TDBBs,
                                 SmallVector<MachineInstr*, 16> &Copies) {
  // Set the limit on the number of instructions to duplicate, with a default
  // of one less than the tail-merge threshold. When optimizing for size,
  // duplicate only one, because one branch instruction can be eliminated to
  // compensate for the duplication.
  unsigned MaxDuplicateCount;
  if (MF.getFunction()->hasFnAttr(Attribute::OptimizeForSize))
    MaxDuplicateCount = 1;
  else
    MaxDuplicateCount = TailDuplicateSize;

  if (PreRegAlloc) {
      // Pre-regalloc tail duplication hurts compile time and doesn't help
      // much except for indirect branches.
    if (TailBB->empty() || !TailBB->back().getDesc().isIndirectBranch())
      return false;
    // If the target has hardware branch prediction that can handle indirect
    // branches, duplicating them can often make them predictable when there
    // are common paths through the code.  The limit needs to be high enough
    // to allow undoing the effects of tail merging and other optimizations
    // that rearrange the predecessors of the indirect branch.
    MaxDuplicateCount = 20;
  }

  // Don't try to tail-duplicate single-block loops.
  if (TailBB->isSuccessor(TailBB))
    return false;

  // Check the instructions in the block to determine whether tail-duplication
  // is invalid or unlikely to be profitable.
  unsigned InstrCount = 0;
  bool HasCall = false;
  for (MachineBasicBlock::iterator I = TailBB->begin();
       I != TailBB->end(); ++I) {
    // Non-duplicable things shouldn't be tail-duplicated.
    if (I->getDesc().isNotDuplicable()) return false;
    // Do not duplicate 'return' instructions if this is a pre-regalloc run.
    // A return may expand into a lot more instructions (e.g. reload of callee
    // saved registers) after PEI.
    if (PreRegAlloc && I->getDesc().isReturn()) return false;
    // Don't duplicate more than the threshold.
    if (InstrCount == MaxDuplicateCount) return false;
    // Remember if we saw a call.
    if (I->getDesc().isCall()) HasCall = true;
    if (I->getOpcode() != TargetInstrInfo::PHI)
      InstrCount += 1;
  }
  // Heuristically, don't tail-duplicate calls if it would expand code size,
  // as it's less likely to be worth the extra cost.
  if (InstrCount > 1 && HasCall)
    return false;

  DEBUG(dbgs() << "\n*** Tail-duplicating BB#" << TailBB->getNumber() << '\n');

  // Iterate through all the unique predecessors and tail-duplicate this
  // block into them, if possible. Copying the list ahead of time also
  // avoids trouble with the predecessor list reallocating.
  bool Changed = false;
  SmallSetVector<MachineBasicBlock*, 8> Preds(TailBB->pred_begin(),
                                              TailBB->pred_end());
  for (SmallSetVector<MachineBasicBlock *, 8>::iterator PI = Preds.begin(),
       PE = Preds.end(); PI != PE; ++PI) {
    MachineBasicBlock *PredBB = *PI;

    assert(TailBB != PredBB &&
           "Single-block loop should have been rejected earlier!");
    if (PredBB->succ_size() > 1) continue;

    MachineBasicBlock *PredTBB, *PredFBB;
    SmallVector<MachineOperand, 4> PredCond;
    if (TII->AnalyzeBranch(*PredBB, PredTBB, PredFBB, PredCond, true))
      continue;
    if (!PredCond.empty())
      continue;
    // EH edges are ignored by AnalyzeBranch.
    if (PredBB->succ_size() != 1)
      continue;
    // Don't duplicate into a fall-through predecessor (at least for now).
    if (PredBB->isLayoutSuccessor(TailBB) && PredBB->canFallThrough())
      continue;

    DEBUG(dbgs() << "\nTail-duplicating into PredBB: " << *PredBB
                 << "From Succ: " << *TailBB);

    TDBBs.push_back(PredBB);

    // Remove PredBB's unconditional branch.
    TII->RemoveBranch(*PredBB);

    // Clone the contents of TailBB into PredBB.
    DenseMap<unsigned, unsigned> LocalVRMap;
    SmallVector<std::pair<unsigned,unsigned>, 4> CopyInfos;
    MachineBasicBlock::iterator I = TailBB->begin();
    while (I != TailBB->end()) {
      MachineInstr *MI = &*I;
      ++I;
      if (MI->getOpcode() == TargetInstrInfo::PHI) {
        // Replace the uses of the def of the PHI with the register coming
        // from PredBB.
        ProcessPHI(MI, TailBB, PredBB, LocalVRMap, CopyInfos);
      } else {
        // Replace def of virtual registers with new registers, and update
        // uses with PHI source register or the new registers.
        DuplicateInstruction(MI, TailBB, PredBB, MF, LocalVRMap);
      }
    }
    MachineBasicBlock::iterator Loc = PredBB->getFirstTerminator();
    for (unsigned i = 0, e = CopyInfos.size(); i != e; ++i) {
      const TargetRegisterClass *RC = MRI->getRegClass(CopyInfos[i].first);
      TII->copyRegToReg(*PredBB, Loc, CopyInfos[i].first,
                        CopyInfos[i].second, RC,RC);
      MachineInstr *CopyMI = prior(Loc);
      Copies.push_back(CopyMI);
    }
    NumInstrDups += TailBB->size() - 1; // subtract one for removed branch

    // Update the CFG.
    PredBB->removeSuccessor(PredBB->succ_begin());
    assert(PredBB->succ_empty() &&
           "TailDuplicate called on block with multiple successors!");
    for (MachineBasicBlock::succ_iterator I = TailBB->succ_begin(),
           E = TailBB->succ_end(); I != E; ++I)
      PredBB->addSuccessor(*I);

    Changed = true;
    ++NumTailDups;
  }

  // If TailBB was duplicated into all its predecessors except for the prior
  // block, which falls through unconditionally, move the contents of this
  // block into the prior block.
  MachineBasicBlock *PrevBB = prior(MachineFunction::iterator(TailBB));
  MachineBasicBlock *PriorTBB = 0, *PriorFBB = 0;
  SmallVector<MachineOperand, 4> PriorCond;
  bool PriorUnAnalyzable =
    TII->AnalyzeBranch(*PrevBB, PriorTBB, PriorFBB, PriorCond, true);
  // This has to check PrevBB->succ_size() because EH edges are ignored by
  // AnalyzeBranch.
  if (!PriorUnAnalyzable && PriorCond.empty() && !PriorTBB &&
      TailBB->pred_size() == 1 && PrevBB->succ_size() == 1 &&
      !TailBB->hasAddressTaken()) {
    DEBUG(dbgs() << "\nMerging into block: " << *PrevBB
          << "From MBB: " << *TailBB);
    if (PreRegAlloc) {
      DenseMap<unsigned, unsigned> LocalVRMap;
      SmallVector<std::pair<unsigned,unsigned>, 4> CopyInfos;
      MachineBasicBlock::iterator I = TailBB->begin();
      // Process PHI instructions first.
      while (I != TailBB->end() && I->getOpcode() == TargetInstrInfo::PHI) {
        // Replace the uses of the def of the PHI with the register coming
        // from PredBB.
        MachineInstr *MI = &*I++;
        ProcessPHI(MI, TailBB, PrevBB, LocalVRMap, CopyInfos);
        if (MI->getParent())
          MI->eraseFromParent();
      }

      // Now copy the non-PHI instructions.
      while (I != TailBB->end()) {
        // Replace def of virtual registers with new registers, and update
        // uses with PHI source register or the new registers.
        MachineInstr *MI = &*I++;
        DuplicateInstruction(MI, TailBB, PrevBB, MF, LocalVRMap);
        MI->eraseFromParent();
      }
      MachineBasicBlock::iterator Loc = PrevBB->getFirstTerminator();
      for (unsigned i = 0, e = CopyInfos.size(); i != e; ++i) {
        const TargetRegisterClass *RC = MRI->getRegClass(CopyInfos[i].first);
        TII->copyRegToReg(*PrevBB, Loc, CopyInfos[i].first,
                          CopyInfos[i].second, RC, RC);
        MachineInstr *CopyMI = prior(Loc);
        Copies.push_back(CopyMI);
      }
    } else {
      // No PHIs to worry about, just splice the instructions over.
      PrevBB->splice(PrevBB->end(), TailBB, TailBB->begin(), TailBB->end());
    }
    PrevBB->removeSuccessor(PrevBB->succ_begin());
    assert(PrevBB->succ_empty());
    PrevBB->transferSuccessors(TailBB);
    TDBBs.push_back(PrevBB);
    Changed = true;
  }

  return Changed;
}

/// RemoveDeadBlock - Remove the specified dead machine basic block from the
/// function, updating the CFG.
void TailDuplicatePass::RemoveDeadBlock(MachineBasicBlock *MBB) {
  assert(MBB->pred_empty() && "MBB must be dead!");
  DEBUG(dbgs() << "\nRemoving MBB: " << *MBB);

  // Remove all successors.
  while (!MBB->succ_empty())
    MBB->removeSuccessor(MBB->succ_end()-1);

  // If there are any labels in the basic block, unregister them from
  // MachineModuleInfo.
  if (MMI && !MBB->empty()) {
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      if (I->isLabel())
        // The label ID # is always operand #0, an immediate.
        MMI->InvalidateLabel(I->getOperand(0).getImm());
    }
  }

  // Remove the block.
  MBB->eraseFromParent();
}

