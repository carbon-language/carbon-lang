//===-- SIOptimizeExecMaskingPreRA.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass performs exec mask handling peephole optimizations which needs
/// to be done before register allocation to reduce register pressure.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-optimize-exec-masking-pre-ra"

namespace {

class SIOptimizeExecMaskingPreRA : public MachineFunctionPass {
private:
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;

  unsigned AndOpc;
  unsigned Andn2Opc;
  unsigned OrSaveExecOpc;
  unsigned XorTermrOpc;
  MCRegister CondReg;
  MCRegister ExecReg;

  Register optimizeVcndVcmpPair(MachineBasicBlock &MBB);
  bool optimizeElseBranch(MachineBasicBlock &MBB);

public:
  static char ID;

  SIOptimizeExecMaskingPreRA() : MachineFunctionPass(ID) {
    initializeSIOptimizeExecMaskingPreRAPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI optimize exec mask operations pre-RA";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervals>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIOptimizeExecMaskingPreRA, DEBUG_TYPE,
                      "SI optimize exec mask operations pre-RA", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(SIOptimizeExecMaskingPreRA, DEBUG_TYPE,
                    "SI optimize exec mask operations pre-RA", false, false)

char SIOptimizeExecMaskingPreRA::ID = 0;

char &llvm::SIOptimizeExecMaskingPreRAID = SIOptimizeExecMaskingPreRA::ID;

FunctionPass *llvm::createSIOptimizeExecMaskingPreRAPass() {
  return new SIOptimizeExecMaskingPreRA();
}

// See if there is a def between \p AndIdx and \p SelIdx that needs to live
// beyond \p AndIdx.
static bool isDefBetween(const LiveRange &LR, SlotIndex AndIdx,
                         SlotIndex SelIdx) {
  LiveQueryResult AndLRQ = LR.Query(AndIdx);
  return (!AndLRQ.isKill() && AndLRQ.valueIn() != LR.Query(SelIdx).valueOut());
}

// FIXME: Why do we bother trying to handle physical registers here?
static bool isDefBetween(const SIRegisterInfo &TRI,
                         LiveIntervals *LIS, Register Reg,
                         const MachineInstr &Sel, const MachineInstr &And) {
  SlotIndex AndIdx = LIS->getInstructionIndex(And);
  SlotIndex SelIdx = LIS->getInstructionIndex(Sel);

  if (Reg.isVirtual())
    return isDefBetween(LIS->getInterval(Reg), AndIdx, SelIdx);

  for (MCRegUnitIterator UI(Reg.asMCReg(), &TRI); UI.isValid(); ++UI) {
    if (isDefBetween(LIS->getRegUnit(*UI), AndIdx, SelIdx))
      return true;
  }

  return false;
}

// Optimize sequence
//    %sel = V_CNDMASK_B32_e64 0, 1, %cc
//    %cmp = V_CMP_NE_U32 1, %1
//    $vcc = S_AND_B64 $exec, %cmp
//    S_CBRANCH_VCC[N]Z
// =>
//    $vcc = S_ANDN2_B64 $exec, %cc
//    S_CBRANCH_VCC[N]Z
//
// It is the negation pattern inserted by DAGCombiner::visitBRCOND() in the
// rebuildSetCC(). We start with S_CBRANCH to avoid exhaustive search, but
// only 3 first instructions are really needed. S_AND_B64 with exec is a
// required part of the pattern since V_CNDMASK_B32 writes zeroes for inactive
// lanes.
//
// Returns %cc register on success.
Register
SIOptimizeExecMaskingPreRA::optimizeVcndVcmpPair(MachineBasicBlock &MBB) {
  auto I = llvm::find_if(MBB.terminators(), [](const MachineInstr &MI) {
                           unsigned Opc = MI.getOpcode();
                           return Opc == AMDGPU::S_CBRANCH_VCCZ ||
                                  Opc == AMDGPU::S_CBRANCH_VCCNZ; });
  if (I == MBB.terminators().end())
    return Register();

  auto *And =
      TRI->findReachingDef(CondReg, AMDGPU::NoSubRegister, *I, *MRI, LIS);
  if (!And || And->getOpcode() != AndOpc ||
      !And->getOperand(1).isReg() || !And->getOperand(2).isReg())
    return Register();

  MachineOperand *AndCC = &And->getOperand(1);
  Register CmpReg = AndCC->getReg();
  unsigned CmpSubReg = AndCC->getSubReg();
  if (CmpReg == Register(ExecReg)) {
    AndCC = &And->getOperand(2);
    CmpReg = AndCC->getReg();
    CmpSubReg = AndCC->getSubReg();
  } else if (And->getOperand(2).getReg() != Register(ExecReg)) {
    return Register();
  }

  auto *Cmp = TRI->findReachingDef(CmpReg, CmpSubReg, *And, *MRI, LIS);
  if (!Cmp || !(Cmp->getOpcode() == AMDGPU::V_CMP_NE_U32_e32 ||
                Cmp->getOpcode() == AMDGPU::V_CMP_NE_U32_e64) ||
      Cmp->getParent() != And->getParent())
    return Register();

  MachineOperand *Op1 = TII->getNamedOperand(*Cmp, AMDGPU::OpName::src0);
  MachineOperand *Op2 = TII->getNamedOperand(*Cmp, AMDGPU::OpName::src1);
  if (Op1->isImm() && Op2->isReg())
    std::swap(Op1, Op2);
  if (!Op1->isReg() || !Op2->isImm() || Op2->getImm() != 1)
    return Register();

  Register SelReg = Op1->getReg();
  auto *Sel = TRI->findReachingDef(SelReg, Op1->getSubReg(), *Cmp, *MRI, LIS);
  if (!Sel || Sel->getOpcode() != AMDGPU::V_CNDMASK_B32_e64)
    return Register();

  if (TII->hasModifiersSet(*Sel, AMDGPU::OpName::src0_modifiers) ||
      TII->hasModifiersSet(*Sel, AMDGPU::OpName::src1_modifiers))
    return Register();

  Op1 = TII->getNamedOperand(*Sel, AMDGPU::OpName::src0);
  Op2 = TII->getNamedOperand(*Sel, AMDGPU::OpName::src1);
  MachineOperand *CC = TII->getNamedOperand(*Sel, AMDGPU::OpName::src2);
  if (!Op1->isImm() || !Op2->isImm() || !CC->isReg() ||
      Op1->getImm() != 0 || Op2->getImm() != 1)
    return Register();

  Register CCReg = CC->getReg();

  // If there was a def between the select and the and, we would need to move it
  // to fold this.
  if (isDefBetween(*TRI, LIS, CCReg, *Sel, *And))
    return Register();

  LLVM_DEBUG(dbgs() << "Folding sequence:\n\t" << *Sel << '\t' << *Cmp << '\t'
                    << *And);

  LIS->RemoveMachineInstrFromMaps(*And);
  MachineInstr *Andn2 =
      BuildMI(MBB, *And, And->getDebugLoc(), TII->get(Andn2Opc),
              And->getOperand(0).getReg())
          .addReg(ExecReg)
          .addReg(CCReg, getUndefRegState(CC->isUndef()), CC->getSubReg());
  MachineOperand &AndSCC = And->getOperand(3);
  assert(AndSCC.getReg() == AMDGPU::SCC);
  MachineOperand &Andn2SCC = Andn2->getOperand(3);
  assert(Andn2SCC.getReg() == AMDGPU::SCC);
  Andn2SCC.setIsDead(AndSCC.isDead());
  And->eraseFromParent();
  LIS->InsertMachineInstrInMaps(*Andn2);

  LLVM_DEBUG(dbgs() << "=>\n\t" << *Andn2 << '\n');

  // Try to remove compare. Cmp value should not used in between of cmp
  // and s_and_b64 if VCC or just unused if any other register.
  if ((CmpReg.isVirtual() && MRI->use_nodbg_empty(CmpReg)) ||
      (CmpReg == Register(CondReg) &&
       std::none_of(std::next(Cmp->getIterator()), Andn2->getIterator(),
                    [&](const MachineInstr &MI) {
                      return MI.readsRegister(CondReg, TRI);
                    }))) {
    LLVM_DEBUG(dbgs() << "Erasing: " << *Cmp << '\n');

    LIS->RemoveMachineInstrFromMaps(*Cmp);
    Cmp->eraseFromParent();

    // Try to remove v_cndmask_b32.
    if (SelReg.isVirtual() && MRI->use_nodbg_empty(SelReg)) {
      LLVM_DEBUG(dbgs() << "Erasing: " << *Sel << '\n');

      LIS->RemoveMachineInstrFromMaps(*Sel);
      Sel->eraseFromParent();
    }
  }

  return CCReg;
}

// Optimize sequence
//    %dst = S_OR_SAVEEXEC %src
//    ... instructions not modifying exec ...
//    %tmp = S_AND $exec, %dst
//    $exec = S_XOR_term $exec, %tmp
// =>
//    %dst = S_OR_SAVEEXEC %src
//    ... instructions not modifying exec ...
//    $exec = S_XOR_term $exec, %dst
//
// Clean up potentially unnecessary code added for safety during
// control flow lowering.
//
// Return whether any changes were made to MBB.
bool SIOptimizeExecMaskingPreRA::optimizeElseBranch(MachineBasicBlock &MBB) {
  if (MBB.empty())
    return false;

  // Check this is an else block.
  auto First = MBB.begin();
  MachineInstr &SaveExecMI = *First;
  if (SaveExecMI.getOpcode() != OrSaveExecOpc)
    return false;

  auto I = llvm::find_if(MBB.terminators(), [this](const MachineInstr &MI) {
    return MI.getOpcode() == XorTermrOpc;
  });
  if (I == MBB.terminators().end())
    return false;

  MachineInstr &XorTermMI = *I;
  if (XorTermMI.getOperand(1).getReg() != Register(ExecReg))
    return false;

  Register SavedExecReg = SaveExecMI.getOperand(0).getReg();
  Register DstReg = XorTermMI.getOperand(2).getReg();

  // Find potentially unnecessary S_AND
  MachineInstr *AndExecMI = nullptr;
  I--;
  while (I != First && !AndExecMI) {
    if (I->getOpcode() == AndOpc && I->getOperand(0).getReg() == DstReg &&
        I->getOperand(1).getReg() == Register(ExecReg))
      AndExecMI = &*I;
    I--;
  }
  if (!AndExecMI)
    return false;

  // Check for exec modifying instructions.
  // Note: exec defs do not create live ranges beyond the
  // instruction so isDefBetween cannot be used.
  // Instead just check that the def segments are adjacent.
  SlotIndex StartIdx = LIS->getInstructionIndex(SaveExecMI);
  SlotIndex EndIdx = LIS->getInstructionIndex(*AndExecMI);
  for (MCRegUnitIterator UI(ExecReg, TRI); UI.isValid(); ++UI) {
    LiveRange &RegUnit = LIS->getRegUnit(*UI);
    if (RegUnit.find(StartIdx) != std::prev(RegUnit.find(EndIdx)))
      return false;
  }

  // Remove unnecessary S_AND
  LIS->removeInterval(SavedExecReg);
  LIS->removeInterval(DstReg);

  SaveExecMI.getOperand(0).setReg(DstReg);

  LIS->RemoveMachineInstrFromMaps(*AndExecMI);
  AndExecMI->eraseFromParent();

  LIS->createAndComputeVirtRegInterval(DstReg);

  return true;
}

bool SIOptimizeExecMaskingPreRA::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TRI = ST.getRegisterInfo();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  LIS = &getAnalysis<LiveIntervals>();

  const bool Wave32 = ST.isWave32();
  AndOpc = Wave32 ? AMDGPU::S_AND_B32 : AMDGPU::S_AND_B64;
  Andn2Opc = Wave32 ? AMDGPU::S_ANDN2_B32 : AMDGPU::S_ANDN2_B64;
  OrSaveExecOpc =
      Wave32 ? AMDGPU::S_OR_SAVEEXEC_B32 : AMDGPU::S_OR_SAVEEXEC_B64;
  XorTermrOpc = Wave32 ? AMDGPU::S_XOR_B32_term : AMDGPU::S_XOR_B64_term;
  CondReg = MCRegister::from(Wave32 ? AMDGPU::VCC_LO : AMDGPU::VCC);
  ExecReg = MCRegister::from(Wave32 ? AMDGPU::EXEC_LO : AMDGPU::EXEC);

  DenseSet<Register> RecalcRegs({AMDGPU::EXEC_LO, AMDGPU::EXEC_HI});
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {

    if (optimizeElseBranch(MBB)) {
      RecalcRegs.insert(AMDGPU::SCC);
      Changed = true;
    }

    if (Register Reg = optimizeVcndVcmpPair(MBB)) {
      RecalcRegs.insert(Reg);
      RecalcRegs.insert(AMDGPU::VCC_LO);
      RecalcRegs.insert(AMDGPU::VCC_HI);
      RecalcRegs.insert(AMDGPU::SCC);
      Changed = true;
    }

    // Try to remove unneeded instructions before s_endpgm.
    if (MBB.succ_empty()) {
      if (MBB.empty())
        continue;

      // Skip this if the endpgm has any implicit uses, otherwise we would need
      // to be careful to update / remove them.
      // S_ENDPGM always has a single imm operand that is not used other than to
      // end up in the encoding
      MachineInstr &Term = MBB.back();
      if (Term.getOpcode() != AMDGPU::S_ENDPGM || Term.getNumOperands() != 1)
        continue;

      SmallVector<MachineBasicBlock*, 4> Blocks({&MBB});

      while (!Blocks.empty()) {
        auto CurBB = Blocks.pop_back_val();
        auto I = CurBB->rbegin(), E = CurBB->rend();
        if (I != E) {
          if (I->isUnconditionalBranch() || I->getOpcode() == AMDGPU::S_ENDPGM)
            ++I;
          else if (I->isBranch())
            continue;
        }

        while (I != E) {
          if (I->isDebugInstr()) {
            I = std::next(I);
            continue;
          }

          if (I->mayStore() || I->isBarrier() || I->isCall() ||
              I->hasUnmodeledSideEffects() || I->hasOrderedMemoryRef())
            break;

          LLVM_DEBUG(dbgs()
                     << "Removing no effect instruction: " << *I << '\n');

          for (auto &Op : I->operands()) {
            if (Op.isReg())
              RecalcRegs.insert(Op.getReg());
          }

          auto Next = std::next(I);
          LIS->RemoveMachineInstrFromMaps(*I);
          I->eraseFromParent();
          I = Next;

          Changed = true;
        }

        if (I != E)
          continue;

        // Try to ascend predecessors.
        for (auto *Pred : CurBB->predecessors()) {
          if (Pred->succ_size() == 1)
            Blocks.push_back(Pred);
        }
      }
      continue;
    }

    // If the only user of a logical operation is move to exec, fold it now
    // to prevent forming of saveexec. I.e.:
    //
    //    %0:sreg_64 = COPY $exec
    //    %1:sreg_64 = S_AND_B64 %0:sreg_64, %2:sreg_64
    // =>
    //    %1 = S_AND_B64 $exec, %2:sreg_64
    unsigned ScanThreshold = 10;
    for (auto I = MBB.rbegin(), E = MBB.rend(); I != E
         && ScanThreshold--; ++I) {
      // Continue scanning if this is not a full exec copy
      if (!(I->isFullCopy() && I->getOperand(1).getReg() == Register(ExecReg)))
        continue;

      Register SavedExec = I->getOperand(0).getReg();
      if (SavedExec.isVirtual() && MRI->hasOneNonDBGUse(SavedExec)) {
        MachineInstr *SingleExecUser = &*MRI->use_instr_nodbg_begin(SavedExec);
        int Idx = SingleExecUser->findRegisterUseOperandIdx(SavedExec);
        assert(Idx != -1);
        if (SingleExecUser->getParent() == I->getParent() &&
            !SingleExecUser->getOperand(Idx).isImplicit() &&
            TII->isOperandLegal(*SingleExecUser, Idx, &I->getOperand(1))) {
          LLVM_DEBUG(dbgs() << "Redundant EXEC COPY: " << *I << '\n');
          LIS->RemoveMachineInstrFromMaps(*I);
          I->eraseFromParent();
          MRI->replaceRegWith(SavedExec, ExecReg);
          LIS->removeInterval(SavedExec);
          Changed = true;
        }
      }
      break;
    }
  }

  if (Changed) {
    for (auto Reg : RecalcRegs) {
      if (Reg.isVirtual()) {
        LIS->removeInterval(Reg);
        if (!MRI->reg_empty(Reg))
          LIS->createAndComputeVirtRegInterval(Reg);
      } else {
        LIS->removeAllRegUnitsForPhysReg(Reg);
      }
    }
  }

  return Changed;
}
