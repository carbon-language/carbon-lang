//===-- SIOptimizeExecMasking.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-optimize-exec-masking"

namespace {

class SIOptimizeExecMasking : public MachineFunctionPass {
public:
  static char ID;

public:
  SIOptimizeExecMasking() : MachineFunctionPass(ID) {
    initializeSIOptimizeExecMaskingPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI optimize exec mask operations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIOptimizeExecMasking, DEBUG_TYPE,
                      "SI optimize exec mask operations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(SIOptimizeExecMasking, DEBUG_TYPE,
                    "SI optimize exec mask operations", false, false)

char SIOptimizeExecMasking::ID = 0;

char &llvm::SIOptimizeExecMaskingID = SIOptimizeExecMasking::ID;

/// If \p MI is a copy from exec, return the register copied to.
static Register isCopyFromExec(const MachineInstr &MI, const GCNSubtarget &ST) {
  switch (MI.getOpcode()) {
  case AMDGPU::COPY:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::S_MOV_B64_term:
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOV_B32_term: {
    const MachineOperand &Src = MI.getOperand(1);
    if (Src.isReg() &&
        Src.getReg() == (ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC))
      return MI.getOperand(0).getReg();
  }
  }

  return AMDGPU::NoRegister;
}

/// If \p MI is a copy to exec, return the register copied from.
static Register isCopyToExec(const MachineInstr &MI, const GCNSubtarget &ST) {
  switch (MI.getOpcode()) {
  case AMDGPU::COPY:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::S_MOV_B32: {
    const MachineOperand &Dst = MI.getOperand(0);
    if (Dst.isReg() &&
        Dst.getReg() == (ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC) &&
        MI.getOperand(1).isReg())
      return MI.getOperand(1).getReg();
    break;
  }
  case AMDGPU::S_MOV_B64_term:
  case AMDGPU::S_MOV_B32_term:
    llvm_unreachable("should have been replaced");
  }

  return Register();
}

/// If \p MI is a logical operation on an exec value,
/// return the register copied to.
static Register isLogicalOpOnExec(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AMDGPU::S_AND_B64:
  case AMDGPU::S_OR_B64:
  case AMDGPU::S_XOR_B64:
  case AMDGPU::S_ANDN2_B64:
  case AMDGPU::S_ORN2_B64:
  case AMDGPU::S_NAND_B64:
  case AMDGPU::S_NOR_B64:
  case AMDGPU::S_XNOR_B64: {
    const MachineOperand &Src1 = MI.getOperand(1);
    if (Src1.isReg() && Src1.getReg() == AMDGPU::EXEC)
      return MI.getOperand(0).getReg();
    const MachineOperand &Src2 = MI.getOperand(2);
    if (Src2.isReg() && Src2.getReg() == AMDGPU::EXEC)
      return MI.getOperand(0).getReg();
    break;
  }
  case AMDGPU::S_AND_B32:
  case AMDGPU::S_OR_B32:
  case AMDGPU::S_XOR_B32:
  case AMDGPU::S_ANDN2_B32:
  case AMDGPU::S_ORN2_B32:
  case AMDGPU::S_NAND_B32:
  case AMDGPU::S_NOR_B32:
  case AMDGPU::S_XNOR_B32: {
    const MachineOperand &Src1 = MI.getOperand(1);
    if (Src1.isReg() && Src1.getReg() == AMDGPU::EXEC_LO)
      return MI.getOperand(0).getReg();
    const MachineOperand &Src2 = MI.getOperand(2);
    if (Src2.isReg() && Src2.getReg() == AMDGPU::EXEC_LO)
      return MI.getOperand(0).getReg();
    break;
  }
  }

  return AMDGPU::NoRegister;
}

static unsigned getSaveExecOp(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::S_AND_B64:
    return AMDGPU::S_AND_SAVEEXEC_B64;
  case AMDGPU::S_OR_B64:
    return AMDGPU::S_OR_SAVEEXEC_B64;
  case AMDGPU::S_XOR_B64:
    return AMDGPU::S_XOR_SAVEEXEC_B64;
  case AMDGPU::S_ANDN2_B64:
    return AMDGPU::S_ANDN2_SAVEEXEC_B64;
  case AMDGPU::S_ORN2_B64:
    return AMDGPU::S_ORN2_SAVEEXEC_B64;
  case AMDGPU::S_NAND_B64:
    return AMDGPU::S_NAND_SAVEEXEC_B64;
  case AMDGPU::S_NOR_B64:
    return AMDGPU::S_NOR_SAVEEXEC_B64;
  case AMDGPU::S_XNOR_B64:
    return AMDGPU::S_XNOR_SAVEEXEC_B64;
  case AMDGPU::S_AND_B32:
    return AMDGPU::S_AND_SAVEEXEC_B32;
  case AMDGPU::S_OR_B32:
    return AMDGPU::S_OR_SAVEEXEC_B32;
  case AMDGPU::S_XOR_B32:
    return AMDGPU::S_XOR_SAVEEXEC_B32;
  case AMDGPU::S_ANDN2_B32:
    return AMDGPU::S_ANDN2_SAVEEXEC_B32;
  case AMDGPU::S_ORN2_B32:
    return AMDGPU::S_ORN2_SAVEEXEC_B32;
  case AMDGPU::S_NAND_B32:
    return AMDGPU::S_NAND_SAVEEXEC_B32;
  case AMDGPU::S_NOR_B32:
    return AMDGPU::S_NOR_SAVEEXEC_B32;
  case AMDGPU::S_XNOR_B32:
    return AMDGPU::S_XNOR_SAVEEXEC_B32;
  default:
    return AMDGPU::INSTRUCTION_LIST_END;
  }
}

// These are only terminators to get correct spill code placement during
// register allocation, so turn them back into normal instructions.
static bool removeTerminatorBit(const SIInstrInfo &TII, MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AMDGPU::S_MOV_B32_term: {
    bool RegSrc = MI.getOperand(1).isReg();
    MI.setDesc(TII.get(RegSrc ? AMDGPU::COPY : AMDGPU::S_MOV_B32));
    return true;
  }
  case AMDGPU::S_MOV_B64_term: {
    bool RegSrc = MI.getOperand(1).isReg();
    MI.setDesc(TII.get(RegSrc ? AMDGPU::COPY : AMDGPU::S_MOV_B64));
    return true;
  }
  case AMDGPU::S_XOR_B64_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(AMDGPU::S_XOR_B64));
    return true;
  }
  case AMDGPU::S_XOR_B32_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(AMDGPU::S_XOR_B32));
    return true;
  }
  case AMDGPU::S_OR_B64_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(AMDGPU::S_OR_B64));
    return true;
  }
  case AMDGPU::S_OR_B32_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(AMDGPU::S_OR_B32));
    return true;
  }
  case AMDGPU::S_ANDN2_B64_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(AMDGPU::S_ANDN2_B64));
    return true;
  }
  case AMDGPU::S_ANDN2_B32_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(AMDGPU::S_ANDN2_B32));
    return true;
  }
  case AMDGPU::S_AND_B64_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(AMDGPU::S_AND_B64));
    return true;
  }
  case AMDGPU::S_AND_B32_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(AMDGPU::S_AND_B32));
    return true;
  }
  default:
    return false;
  }
}

// Turn all pseudoterminators in the block into their equivalent non-terminator
// instructions. Returns the reverse iterator to the first non-terminator
// instruction in the block.
static MachineBasicBlock::reverse_iterator fixTerminators(
  const SIInstrInfo &TII,
  MachineBasicBlock &MBB) {
  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), E = MBB.rend();

  bool Seen = false;
  MachineBasicBlock::reverse_iterator FirstNonTerm = I;
  for (; I != E; ++I) {
    if (!I->isTerminator())
      return Seen ? FirstNonTerm : I;

    if (removeTerminatorBit(TII, *I)) {
      if (!Seen) {
        FirstNonTerm = I;
        Seen = true;
      }
    }
  }

  return FirstNonTerm;
}

static MachineBasicBlock::reverse_iterator findExecCopy(
  const SIInstrInfo &TII,
  const GCNSubtarget &ST,
  MachineBasicBlock &MBB,
  MachineBasicBlock::reverse_iterator I,
  unsigned CopyToExec) {
  const unsigned InstLimit = 25;

  auto E = MBB.rend();
  for (unsigned N = 0; N <= InstLimit && I != E; ++I, ++N) {
    Register CopyFromExec = isCopyFromExec(*I, ST);
    if (CopyFromExec.isValid())
      return I;
  }

  return E;
}

// XXX - Seems LivePhysRegs doesn't work correctly since it will incorrectly
// report the register as unavailable because a super-register with a lane mask
// is unavailable.
static bool isLiveOut(const MachineBasicBlock &MBB, unsigned Reg) {
  for (MachineBasicBlock *Succ : MBB.successors()) {
    if (Succ->isLiveIn(Reg))
      return true;
  }

  return false;
}

// Backwards-iterate from Origin (for n=MaxInstructions iterations) until either
// the beginning of the BB is reached or Pred evaluates to true - which can be
// an arbitrary condition based on the current MachineInstr, for instance an
// target instruction. Breaks prematurely by returning nullptr if  one of the
// registers given in NonModifiableRegs is modified by the current instruction.
static MachineInstr *
findInstrBackwards(MachineInstr &Origin,
                   std::function<bool(MachineInstr *)> Pred,
                   ArrayRef<MCRegister> NonModifiableRegs,
                   const SIRegisterInfo *TRI, unsigned MaxInstructions = 20) {
  MachineBasicBlock::reverse_iterator A = Origin.getReverseIterator(),
                                      E = Origin.getParent()->rend();
  unsigned CurrentIteration = 0;

  for (++A; CurrentIteration < MaxInstructions && A != E; ++A) {
    if (A->isDebugInstr())
      continue;
    
    if (Pred(&*A))
      return &*A;

    for (MCRegister Reg : NonModifiableRegs) {
      if (A->modifiesRegister(Reg, TRI))
        return nullptr;
    }
    
    ++CurrentIteration;
  }

  return nullptr;
}


// Determine if a register Reg is not re-defined and still in use
// in the range (Stop..Start].
// It does so by backwards calculating liveness from the end of the BB until
// either Stop or the beginning of the BB is reached.
// After liveness is calculated, we can determine if Reg is still in use and not
// defined inbetween the instructions.
static bool isRegisterInUseBetween(MachineInstr &Stop, MachineInstr &Start,
                                   MCRegister Reg, const SIRegisterInfo *TRI,
                                   MachineRegisterInfo &MRI,
                                   bool useLiveOuts = false,
                                   bool ignoreStart = false) {
  LivePhysRegs LR(*TRI);
  if (useLiveOuts)
    LR.addLiveOuts(*Stop.getParent());

  MachineBasicBlock::reverse_iterator A(Start);
  MachineBasicBlock::reverse_iterator E(Stop);

  if (ignoreStart)
    ++A;

  for (; A != Stop.getParent()->rend() && A != Stop; ++A) {
    LR.stepBackward(*A);
  }

  return !LR.available(MRI, Reg);
}

// Determine if a register Reg is not re-defined and still in use
// in the range (Stop..BB.end].
static bool isRegisterInUseAfter(MachineInstr &Stop, MCRegister Reg,
                                 const SIRegisterInfo *TRI,
                                 MachineRegisterInfo &MRI) {
  return isRegisterInUseBetween(Stop, *Stop.getParent()->rbegin(), Reg, TRI,
                                MRI, true);
}

// Tries to find a possibility to optimize a v_cmp ..., s_and_saveexec sequence
// by looking at an instance of a s_and_saveexec instruction. Returns a pointer
// to the v_cmp instruction if it is safe to replace the sequence (see the
// conditions in the function body). This is after register allocation, so some
// checks on operand dependencies need to be considered.
static MachineInstr *findPossibleVCMPVCMPXOptimization(
    MachineInstr &SaveExec, MCRegister Exec, const SIRegisterInfo *TRI,
    const SIInstrInfo *TII, MachineRegisterInfo &MRI) {

  MachineInstr *VCmp = nullptr;

  Register SaveExecDest = SaveExec.getOperand(0).getReg();
  if (!TRI->isSGPRReg(MRI, SaveExecDest))
    return nullptr;

  MachineOperand *SaveExecSrc0 =
      TII->getNamedOperand(SaveExec, AMDGPU::OpName::src0);
  if (!SaveExecSrc0->isReg())
    return nullptr;

  // Try to find the last v_cmp instruction that defs the saveexec input
  // operand without any write to Exec or the saveexec input operand inbetween.
  VCmp = findInstrBackwards(
      SaveExec,
      [&](MachineInstr *Check) {
        return AMDGPU::getVCMPXOpFromVCMP(Check->getOpcode()) != -1 &&
               Check->modifiesRegister(SaveExecSrc0->getReg(), TRI);
      },
      {Exec, SaveExecSrc0->getReg()}, TRI);

  if (!VCmp)
    return nullptr;

  MachineOperand *VCmpDest = TII->getNamedOperand(*VCmp, AMDGPU::OpName::sdst);
  assert(VCmpDest && "Should have an sdst operand!");

  // Check if any of the v_cmp source operands is written by the saveexec.
  MachineOperand *Src0 = TII->getNamedOperand(*VCmp, AMDGPU::OpName::src0);
  if (Src0->isReg() && TRI->isSGPRReg(MRI, Src0->getReg()) &&
      SaveExec.modifiesRegister(Src0->getReg(), TRI))
    return nullptr;

  MachineOperand *Src1 = TII->getNamedOperand(*VCmp, AMDGPU::OpName::src1);
  if (Src1->isReg() && TRI->isSGPRReg(MRI, Src1->getReg()) &&
      SaveExec.modifiesRegister(Src1->getReg(), TRI))
    return nullptr;

  // Don't do the transformation if the destination operand is included in
  // it's MBB Live-outs, meaning it's used in any of it's successors, leading
  // to incorrect code if the v_cmp and therefore the def of
  // the dest operand is removed.
  if (isLiveOut(*VCmp->getParent(), VCmpDest->getReg()))
    return nullptr;

  // If the v_cmp target is in use between v_cmp and s_and_saveexec or after the
  // s_and_saveexec, skip the optimization.
  if (isRegisterInUseBetween(*VCmp, SaveExec, VCmpDest->getReg(), TRI, MRI,
                             false, true) ||
      isRegisterInUseAfter(SaveExec, VCmpDest->getReg(), TRI, MRI))
    return nullptr;

  // Try to determine if there is a write to any of the VCmp
  // operands between the saveexec and the vcmp.
  // If yes, additional VGPR spilling might need to be inserted. In this case,
  // it's not worth replacing the instruction sequence.
  SmallVector<MCRegister, 2> NonDefRegs;
  if (Src0->isReg())
    NonDefRegs.push_back(Src0->getReg());

  if (Src1->isReg())
    NonDefRegs.push_back(Src1->getReg());

  if (!findInstrBackwards(
          SaveExec, [&](MachineInstr *Check) { return Check == VCmp; },
          NonDefRegs, TRI))
    return nullptr;

  return VCmp;
}

// Inserts the optimized s_mov_b32 / v_cmpx sequence based on the
// operands extracted from a v_cmp ..., s_and_saveexec pattern.
static bool optimizeVCMPSaveExecSequence(MachineInstr &SaveExecInstr,
                                         MachineInstr &VCmp, MCRegister Exec,
                                         const SIInstrInfo *TII,
                                         const SIRegisterInfo *TRI,
                                         MachineRegisterInfo &MRI) {
  const int NewOpcode = AMDGPU::getVCMPXOpFromVCMP(VCmp.getOpcode());

  if (NewOpcode == -1)
    return false;

  MachineOperand *Src0 = TII->getNamedOperand(VCmp, AMDGPU::OpName::src0);
  MachineOperand *Src1 = TII->getNamedOperand(VCmp, AMDGPU::OpName::src1);

  Register MoveDest = SaveExecInstr.getOperand(0).getReg();

  MachineBasicBlock::instr_iterator InsertPosIt = SaveExecInstr.getIterator();
  if (!SaveExecInstr.uses().empty()) {
    bool isSGPR32 = TRI->getRegSizeInBits(MoveDest, MRI) == 32;
    unsigned MovOpcode = isSGPR32 ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;
    BuildMI(*SaveExecInstr.getParent(), InsertPosIt,
            SaveExecInstr.getDebugLoc(), TII->get(MovOpcode), MoveDest)
        .addReg(Exec);
  }

  // Omit dst as V_CMPX is implicitly writing to EXEC.
  // Add dummy src and clamp modifiers, if needed.
  auto Builder = BuildMI(*VCmp.getParent(), std::next(InsertPosIt),
                         VCmp.getDebugLoc(), TII->get(NewOpcode));

  auto TryAddImmediateValueFromNamedOperand =
      [&](unsigned OperandName) -> void {
    if (auto *Mod = TII->getNamedOperand(VCmp, OperandName))
      Builder.addImm(Mod->getImm());
  };

  TryAddImmediateValueFromNamedOperand(AMDGPU::OpName::src0_modifiers);
  Builder.add(*Src0);

  TryAddImmediateValueFromNamedOperand(AMDGPU::OpName::src1_modifiers);
  Builder.add(*Src1);

  TryAddImmediateValueFromNamedOperand(AMDGPU::OpName::clamp);

  return true;
}

bool SIOptimizeExecMasking::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  MachineRegisterInfo *MRI = &MF.getRegInfo();
  MCRegister Exec = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;

  // Optimize sequences emitted for control flow lowering. They are originally
  // emitted as the separate operations because spill code may need to be
  // inserted for the saved copy of exec.
  //
  //     x = copy exec
  //     z = s_<op>_b64 x, y
  //     exec = copy z
  // =>
  //     x = s_<op>_saveexec_b64 y
  //

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock::reverse_iterator I = fixTerminators(*TII, MBB);
    MachineBasicBlock::reverse_iterator E = MBB.rend();
    if (I == E)
      continue;

    // It's possible to see other terminator copies after the exec copy. This
    // can happen if control flow pseudos had their outputs used by phis.
    Register CopyToExec;

    unsigned SearchCount = 0;
    const unsigned SearchLimit = 5;
    while (I != E && SearchCount++ < SearchLimit) {
      CopyToExec = isCopyToExec(*I, ST);
      if (CopyToExec)
        break;
      ++I;
    }

    if (!CopyToExec)
      continue;

    // Scan backwards to find the def.
    auto CopyToExecInst = &*I;
    auto CopyFromExecInst = findExecCopy(*TII, ST, MBB, I, CopyToExec);
    if (CopyFromExecInst == E) {
      auto PrepareExecInst = std::next(I);
      if (PrepareExecInst == E)
        continue;
      // Fold exec = COPY (S_AND_B64 reg, exec) -> exec = S_AND_B64 reg, exec
      if (CopyToExecInst->getOperand(1).isKill() &&
          isLogicalOpOnExec(*PrepareExecInst) == CopyToExec) {
        LLVM_DEBUG(dbgs() << "Fold exec copy: " << *PrepareExecInst);

        PrepareExecInst->getOperand(0).setReg(Exec);

        LLVM_DEBUG(dbgs() << "into: " << *PrepareExecInst << '\n');

        CopyToExecInst->eraseFromParent();
        Changed = true;
      }

      continue;
    }

    if (isLiveOut(MBB, CopyToExec)) {
      // The copied register is live out and has a second use in another block.
      LLVM_DEBUG(dbgs() << "Exec copy source register is live out\n");
      continue;
    }

    Register CopyFromExec = CopyFromExecInst->getOperand(0).getReg();
    MachineInstr *SaveExecInst = nullptr;
    SmallVector<MachineInstr *, 4> OtherUseInsts;

    for (MachineBasicBlock::iterator J
           = std::next(CopyFromExecInst->getIterator()), JE = I->getIterator();
         J != JE; ++J) {
      if (SaveExecInst && J->readsRegister(Exec, TRI)) {
        LLVM_DEBUG(dbgs() << "exec read prevents saveexec: " << *J << '\n');
        // Make sure this is inserted after any VALU ops that may have been
        // scheduled in between.
        SaveExecInst = nullptr;
        break;
      }

      bool ReadsCopyFromExec = J->readsRegister(CopyFromExec, TRI);

      if (J->modifiesRegister(CopyToExec, TRI)) {
        if (SaveExecInst) {
          LLVM_DEBUG(dbgs() << "Multiple instructions modify "
                            << printReg(CopyToExec, TRI) << '\n');
          SaveExecInst = nullptr;
          break;
        }

        unsigned SaveExecOp = getSaveExecOp(J->getOpcode());
        if (SaveExecOp == AMDGPU::INSTRUCTION_LIST_END)
          break;

        if (ReadsCopyFromExec) {
          SaveExecInst = &*J;
          LLVM_DEBUG(dbgs() << "Found save exec op: " << *SaveExecInst << '\n');
          continue;
        } else {
          LLVM_DEBUG(dbgs()
                     << "Instruction does not read exec copy: " << *J << '\n');
          break;
        }
      } else if (ReadsCopyFromExec && !SaveExecInst) {
        // Make sure no other instruction is trying to use this copy, before it
        // will be rewritten by the saveexec, i.e. hasOneUse. There may have
        // been another use, such as an inserted spill. For example:
        //
        // %sgpr0_sgpr1 = COPY %exec
        // spill %sgpr0_sgpr1
        // %sgpr2_sgpr3 = S_AND_B64 %sgpr0_sgpr1
        //
        LLVM_DEBUG(dbgs() << "Found second use of save inst candidate: " << *J
                          << '\n');
        break;
      }

      if (SaveExecInst && J->readsRegister(CopyToExec, TRI)) {
        assert(SaveExecInst != &*J);
        OtherUseInsts.push_back(&*J);
      }
    }

    if (!SaveExecInst)
      continue;

    LLVM_DEBUG(dbgs() << "Insert save exec op: " << *SaveExecInst << '\n');

    MachineOperand &Src0 = SaveExecInst->getOperand(1);
    MachineOperand &Src1 = SaveExecInst->getOperand(2);

    MachineOperand *OtherOp = nullptr;

    if (Src0.isReg() && Src0.getReg() == CopyFromExec) {
      OtherOp = &Src1;
    } else if (Src1.isReg() && Src1.getReg() == CopyFromExec) {
      if (!SaveExecInst->isCommutable())
        break;

      OtherOp = &Src0;
    } else
      llvm_unreachable("unexpected");

    CopyFromExecInst->eraseFromParent();

    auto InsPt = SaveExecInst->getIterator();
    const DebugLoc &DL = SaveExecInst->getDebugLoc();

    BuildMI(MBB, InsPt, DL, TII->get(getSaveExecOp(SaveExecInst->getOpcode())),
            CopyFromExec)
      .addReg(OtherOp->getReg());
    SaveExecInst->eraseFromParent();

    CopyToExecInst->eraseFromParent();

    for (MachineInstr *OtherInst : OtherUseInsts) {
      OtherInst->substituteRegister(CopyToExec, Exec,
                                    AMDGPU::NoSubRegister, *TRI);
    }

    Changed = true;
  }

  // After all s_op_saveexec instructions are inserted,
  // replace (on GFX10.3 and later)
  // v_cmp_* SGPR, IMM, VGPR
  // s_and_saveexec_b32 EXEC_SGPR_DEST, SGPR
  // with
  // s_mov_b32 EXEC_SGPR_DEST, exec_lo
  // v_cmpx_* IMM, VGPR
  // to reduce pipeline stalls.
  if (ST.hasGFX10_3Insts()) {
    DenseMap<MachineInstr *, MachineInstr *> SaveExecVCmpMapping;
    const unsigned AndSaveExecOpcode =
        ST.isWave32() ? AMDGPU::S_AND_SAVEEXEC_B32 : AMDGPU::S_AND_SAVEEXEC_B64;

    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        // Record relevant v_cmp / s_and_saveexec instruction pairs for
        // replacement.
        if (MI.getOpcode() != AndSaveExecOpcode)
          continue;

        if (MachineInstr *VCmp =
                findPossibleVCMPVCMPXOptimization(MI, Exec, TRI, TII, *MRI))
          SaveExecVCmpMapping[&MI] = VCmp;
      }
    }

    for (const auto &Entry : SaveExecVCmpMapping) {
      MachineInstr *SaveExecInstr = Entry.getFirst();
      MachineInstr *VCmpInstr = Entry.getSecond();

      if (optimizeVCMPSaveExecSequence(*SaveExecInstr, *VCmpInstr, Exec, TII,
                                       TRI, *MRI)) {
        SaveExecInstr->eraseFromParent();
        VCmpInstr->eraseFromParent();

        Changed = true;
      }
    }
  }

  return Changed;
}
