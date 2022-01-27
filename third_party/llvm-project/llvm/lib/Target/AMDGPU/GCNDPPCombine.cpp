//=======- GCNDPPCombine.cpp - optimization for DPP instructions ---==========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The pass combines V_MOV_B32_dpp instruction with its VALU uses as a DPP src0
// operand. If any of the use instruction cannot be combined with the mov the
// whole sequence is reverted.
//
// $old = ...
// $dpp_value = V_MOV_B32_dpp $old, $vgpr_to_be_read_from_other_lane,
//                            dpp_controls..., $row_mask, $bank_mask, $bound_ctrl
// $res = VALU $dpp_value [, src1]
//
// to
//
// $res = VALU_DPP $combined_old, $vgpr_to_be_read_from_other_lane, [src1,]
//                 dpp_controls..., $row_mask, $bank_mask, $combined_bound_ctrl
//
// Combining rules :
//
// if $row_mask and $bank_mask are fully enabled (0xF) and
//    $bound_ctrl==DPP_BOUND_ZERO or $old==0
// -> $combined_old = undef,
//    $combined_bound_ctrl = DPP_BOUND_ZERO
//
// if the VALU op is binary and
//    $bound_ctrl==DPP_BOUND_OFF and
//    $old==identity value (immediate) for the VALU op
// -> $combined_old = src1,
//    $combined_bound_ctrl = DPP_BOUND_OFF
//
// Otherwise cancel.
//
// The mov_dpp instruction should reside in the same BB as all its uses
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "gcn-dpp-combine"

STATISTIC(NumDPPMovsCombined, "Number of DPP moves combined.");

namespace {

class GCNDPPCombine : public MachineFunctionPass {
  MachineRegisterInfo *MRI;
  const SIInstrInfo *TII;
  const GCNSubtarget *ST;

  using RegSubRegPair = TargetInstrInfo::RegSubRegPair;

  MachineOperand *getOldOpndValue(MachineOperand &OldOpnd) const;

  MachineInstr *createDPPInst(MachineInstr &OrigMI, MachineInstr &MovMI,
                              RegSubRegPair CombOldVGPR,
                              MachineOperand *OldOpnd, bool CombBCZ,
                              bool IsShrinkable) const;

  MachineInstr *createDPPInst(MachineInstr &OrigMI, MachineInstr &MovMI,
                              RegSubRegPair CombOldVGPR, bool CombBCZ,
                              bool IsShrinkable) const;

  bool hasNoImmOrEqual(MachineInstr &MI,
                       unsigned OpndName,
                       int64_t Value,
                       int64_t Mask = -1) const;

  bool combineDPPMov(MachineInstr &MI) const;

public:
  static char ID;

  GCNDPPCombine() : MachineFunctionPass(ID) {
    initializeGCNDPPCombinePass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "GCN DPP Combine"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties()
      .set(MachineFunctionProperties::Property::IsSSA);
  }

private:
  int getDPPOp(unsigned Op, bool IsShrinkable) const;
  bool isShrinkable(MachineInstr &MI) const;
};

} // end anonymous namespace

INITIALIZE_PASS(GCNDPPCombine, DEBUG_TYPE, "GCN DPP Combine", false, false)

char GCNDPPCombine::ID = 0;

char &llvm::GCNDPPCombineID = GCNDPPCombine::ID;

FunctionPass *llvm::createGCNDPPCombinePass() {
  return new GCNDPPCombine();
}

bool GCNDPPCombine::isShrinkable(MachineInstr &MI) const {
  unsigned Op = MI.getOpcode();
  if (!TII->isVOP3(Op)) {
    return false;
  }
  if (!TII->hasVALU32BitEncoding(Op)) {
    LLVM_DEBUG(dbgs() << "  Inst hasn't e32 equivalent\n");
    return false;
  }
  if (const auto *SDst = TII->getNamedOperand(MI, AMDGPU::OpName::sdst)) {
    // Give up if there are any uses of the carry-out from instructions like
    // V_ADD_CO_U32. The shrunken form of the instruction would write it to vcc
    // instead of to a virtual register.
    if (!MRI->use_nodbg_empty(SDst->getReg()))
      return false;
  }
  // check if other than abs|neg modifiers are set (opsel for example)
  const int64_t Mask = ~(SISrcMods::ABS | SISrcMods::NEG);
  if (!hasNoImmOrEqual(MI, AMDGPU::OpName::src0_modifiers, 0, Mask) ||
      !hasNoImmOrEqual(MI, AMDGPU::OpName::src1_modifiers, 0, Mask) ||
      !hasNoImmOrEqual(MI, AMDGPU::OpName::clamp, 0) ||
      !hasNoImmOrEqual(MI, AMDGPU::OpName::omod, 0)) {
    LLVM_DEBUG(dbgs() << "  Inst has non-default modifiers\n");
    return false;
  }
  return true;
}

int GCNDPPCombine::getDPPOp(unsigned Op, bool IsShrinkable) const {
  auto DPP32 = AMDGPU::getDPPOp32(Op);
  if (IsShrinkable) {
    assert(DPP32 == -1);
    auto E32 = AMDGPU::getVOPe32(Op);
    DPP32 = (E32 == -1) ? -1 : AMDGPU::getDPPOp32(E32);
  }
  return (DPP32 == -1 || TII->pseudoToMCOpcode(DPP32) == -1) ? -1 : DPP32;
}

// tracks the register operand definition and returns:
//   1. immediate operand used to initialize the register if found
//   2. nullptr if the register operand is undef
//   3. the operand itself otherwise
MachineOperand *GCNDPPCombine::getOldOpndValue(MachineOperand &OldOpnd) const {
  auto *Def = getVRegSubRegDef(getRegSubRegPair(OldOpnd), *MRI);
  if (!Def)
    return nullptr;

  switch(Def->getOpcode()) {
  default: break;
  case AMDGPU::IMPLICIT_DEF:
    return nullptr;
  case AMDGPU::COPY:
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B64_PSEUDO: {
    auto &Op1 = Def->getOperand(1);
    if (Op1.isImm())
      return &Op1;
    break;
  }
  }
  return &OldOpnd;
}

MachineInstr *GCNDPPCombine::createDPPInst(MachineInstr &OrigMI,
                                           MachineInstr &MovMI,
                                           RegSubRegPair CombOldVGPR,
                                           bool CombBCZ,
                                           bool IsShrinkable) const {
  assert(MovMI.getOpcode() == AMDGPU::V_MOV_B32_dpp ||
         MovMI.getOpcode() == AMDGPU::V_MOV_B64_DPP_PSEUDO);

  auto OrigOp = OrigMI.getOpcode();
  auto DPPOp = getDPPOp(OrigOp, IsShrinkable);
  if (DPPOp == -1) {
    LLVM_DEBUG(dbgs() << "  failed: no DPP opcode\n");
    return nullptr;
  }

  auto DPPInst = BuildMI(*OrigMI.getParent(), OrigMI,
                         OrigMI.getDebugLoc(), TII->get(DPPOp))
    .setMIFlags(OrigMI.getFlags());

  bool Fail = false;
  do {
    auto *Dst = TII->getNamedOperand(OrigMI, AMDGPU::OpName::vdst);
    assert(Dst);
    DPPInst.add(*Dst);
    int NumOperands = 1;

    const int OldIdx = AMDGPU::getNamedOperandIdx(DPPOp, AMDGPU::OpName::old);
    if (OldIdx != -1) {
      assert(OldIdx == NumOperands);
      assert(isOfRegClass(
          CombOldVGPR,
          *MRI->getRegClass(
              TII->getNamedOperand(MovMI, AMDGPU::OpName::vdst)->getReg()),
          *MRI));
      auto *Def = getVRegSubRegDef(CombOldVGPR, *MRI);
      DPPInst.addReg(CombOldVGPR.Reg, Def ? 0 : RegState::Undef,
                     CombOldVGPR.SubReg);
      ++NumOperands;
    } else {
      // TODO: this discards MAC/FMA instructions for now, let's add it later
      LLVM_DEBUG(dbgs() << "  failed: no old operand in DPP instruction,"
                           " TBD\n");
      Fail = true;
      break;
    }

    if (auto *Mod0 = TII->getNamedOperand(OrigMI,
                                          AMDGPU::OpName::src0_modifiers)) {
      assert(NumOperands == AMDGPU::getNamedOperandIdx(DPPOp,
                                          AMDGPU::OpName::src0_modifiers));
      assert(0LL == (Mod0->getImm() & ~(SISrcMods::ABS | SISrcMods::NEG)));
      DPPInst.addImm(Mod0->getImm());
      ++NumOperands;
    } else if (AMDGPU::getNamedOperandIdx(DPPOp,
                   AMDGPU::OpName::src0_modifiers) != -1) {
      DPPInst.addImm(0);
      ++NumOperands;
    }
    auto *Src0 = TII->getNamedOperand(MovMI, AMDGPU::OpName::src0);
    assert(Src0);
    if (!TII->isOperandLegal(*DPPInst.getInstr(), NumOperands, Src0)) {
      LLVM_DEBUG(dbgs() << "  failed: src0 is illegal\n");
      Fail = true;
      break;
    }
    DPPInst.add(*Src0);
    DPPInst->getOperand(NumOperands).setIsKill(false);
    ++NumOperands;

    if (auto *Mod1 = TII->getNamedOperand(OrigMI,
                                          AMDGPU::OpName::src1_modifiers)) {
      assert(NumOperands == AMDGPU::getNamedOperandIdx(DPPOp,
                                          AMDGPU::OpName::src1_modifiers));
      assert(0LL == (Mod1->getImm() & ~(SISrcMods::ABS | SISrcMods::NEG)));
      DPPInst.addImm(Mod1->getImm());
      ++NumOperands;
    } else if (AMDGPU::getNamedOperandIdx(DPPOp,
                   AMDGPU::OpName::src1_modifiers) != -1) {
      DPPInst.addImm(0);
      ++NumOperands;
    }
    if (auto *Src1 = TII->getNamedOperand(OrigMI, AMDGPU::OpName::src1)) {
      if (!TII->isOperandLegal(*DPPInst.getInstr(), NumOperands, Src1)) {
        LLVM_DEBUG(dbgs() << "  failed: src1 is illegal\n");
        Fail = true;
        break;
      }
      DPPInst.add(*Src1);
      ++NumOperands;
    }

    if (auto *Src2 = TII->getNamedOperand(OrigMI, AMDGPU::OpName::src2)) {
      if (!TII->getNamedOperand(*DPPInst.getInstr(), AMDGPU::OpName::src2) ||
          !TII->isOperandLegal(*DPPInst.getInstr(), NumOperands, Src2)) {
        LLVM_DEBUG(dbgs() << "  failed: src2 is illegal\n");
        Fail = true;
        break;
      }
      DPPInst.add(*Src2);
    }

    DPPInst.add(*TII->getNamedOperand(MovMI, AMDGPU::OpName::dpp_ctrl));
    DPPInst.add(*TII->getNamedOperand(MovMI, AMDGPU::OpName::row_mask));
    DPPInst.add(*TII->getNamedOperand(MovMI, AMDGPU::OpName::bank_mask));
    DPPInst.addImm(CombBCZ ? 1 : 0);
  } while (false);

  if (Fail) {
    DPPInst.getInstr()->eraseFromParent();
    return nullptr;
  }
  LLVM_DEBUG(dbgs() << "  combined:  " << *DPPInst.getInstr());
  return DPPInst.getInstr();
}

static bool isIdentityValue(unsigned OrigMIOp, MachineOperand *OldOpnd) {
  assert(OldOpnd->isImm());
  switch (OrigMIOp) {
  default: break;
  case AMDGPU::V_ADD_U32_e32:
  case AMDGPU::V_ADD_U32_e64:
  case AMDGPU::V_ADD_CO_U32_e32:
  case AMDGPU::V_ADD_CO_U32_e64:
  case AMDGPU::V_OR_B32_e32:
  case AMDGPU::V_OR_B32_e64:
  case AMDGPU::V_SUBREV_U32_e32:
  case AMDGPU::V_SUBREV_U32_e64:
  case AMDGPU::V_SUBREV_CO_U32_e32:
  case AMDGPU::V_SUBREV_CO_U32_e64:
  case AMDGPU::V_MAX_U32_e32:
  case AMDGPU::V_MAX_U32_e64:
  case AMDGPU::V_XOR_B32_e32:
  case AMDGPU::V_XOR_B32_e64:
    if (OldOpnd->getImm() == 0)
      return true;
    break;
  case AMDGPU::V_AND_B32_e32:
  case AMDGPU::V_AND_B32_e64:
  case AMDGPU::V_MIN_U32_e32:
  case AMDGPU::V_MIN_U32_e64:
    if (static_cast<uint32_t>(OldOpnd->getImm()) ==
        std::numeric_limits<uint32_t>::max())
      return true;
    break;
  case AMDGPU::V_MIN_I32_e32:
  case AMDGPU::V_MIN_I32_e64:
    if (static_cast<int32_t>(OldOpnd->getImm()) ==
        std::numeric_limits<int32_t>::max())
      return true;
    break;
  case AMDGPU::V_MAX_I32_e32:
  case AMDGPU::V_MAX_I32_e64:
    if (static_cast<int32_t>(OldOpnd->getImm()) ==
        std::numeric_limits<int32_t>::min())
      return true;
    break;
  case AMDGPU::V_MUL_I32_I24_e32:
  case AMDGPU::V_MUL_I32_I24_e64:
  case AMDGPU::V_MUL_U32_U24_e32:
  case AMDGPU::V_MUL_U32_U24_e64:
    if (OldOpnd->getImm() == 1)
      return true;
    break;
  }
  return false;
}

MachineInstr *GCNDPPCombine::createDPPInst(
    MachineInstr &OrigMI, MachineInstr &MovMI, RegSubRegPair CombOldVGPR,
    MachineOperand *OldOpndValue, bool CombBCZ, bool IsShrinkable) const {
  assert(CombOldVGPR.Reg);
  if (!CombBCZ && OldOpndValue && OldOpndValue->isImm()) {
    auto *Src1 = TII->getNamedOperand(OrigMI, AMDGPU::OpName::src1);
    if (!Src1 || !Src1->isReg()) {
      LLVM_DEBUG(dbgs() << "  failed: no src1 or it isn't a register\n");
      return nullptr;
    }
    if (!isIdentityValue(OrigMI.getOpcode(), OldOpndValue)) {
      LLVM_DEBUG(dbgs() << "  failed: old immediate isn't an identity\n");
      return nullptr;
    }
    CombOldVGPR = getRegSubRegPair(*Src1);
    auto MovDst = TII->getNamedOperand(MovMI, AMDGPU::OpName::vdst);
    const TargetRegisterClass *RC = MRI->getRegClass(MovDst->getReg());
    if (!isOfRegClass(CombOldVGPR, *RC, *MRI)) {
      LLVM_DEBUG(dbgs() << "  failed: src1 has wrong register class\n");
      return nullptr;
    }
  }
  return createDPPInst(OrigMI, MovMI, CombOldVGPR, CombBCZ, IsShrinkable);
}

// returns true if MI doesn't have OpndName immediate operand or the
// operand has Value
bool GCNDPPCombine::hasNoImmOrEqual(MachineInstr &MI, unsigned OpndName,
                                    int64_t Value, int64_t Mask) const {
  auto *Imm = TII->getNamedOperand(MI, OpndName);
  if (!Imm)
    return true;

  assert(Imm->isImm());
  return (Imm->getImm() & Mask) == Value;
}

bool GCNDPPCombine::combineDPPMov(MachineInstr &MovMI) const {
  assert(MovMI.getOpcode() == AMDGPU::V_MOV_B32_dpp ||
         MovMI.getOpcode() == AMDGPU::V_MOV_B64_DPP_PSEUDO);
  LLVM_DEBUG(dbgs() << "\nDPP combine: " << MovMI);

  auto *DstOpnd = TII->getNamedOperand(MovMI, AMDGPU::OpName::vdst);
  assert(DstOpnd && DstOpnd->isReg());
  auto DPPMovReg = DstOpnd->getReg();
  if (DPPMovReg.isPhysical()) {
    LLVM_DEBUG(dbgs() << "  failed: dpp move writes physreg\n");
    return false;
  }
  if (execMayBeModifiedBeforeAnyUse(*MRI, DPPMovReg, MovMI)) {
    LLVM_DEBUG(dbgs() << "  failed: EXEC mask should remain the same"
                         " for all uses\n");
    return false;
  }

  if (MovMI.getOpcode() == AMDGPU::V_MOV_B64_DPP_PSEUDO) {
    auto *DppCtrl = TII->getNamedOperand(MovMI, AMDGPU::OpName::dpp_ctrl);
    assert(DppCtrl && DppCtrl->isImm());
    if (!AMDGPU::isLegal64BitDPPControl(DppCtrl->getImm())) {
      LLVM_DEBUG(dbgs() << "  failed: 64 bit dpp move uses unsupported"
                           " control value\n");
      // Let it split, then control may become legal.
      return false;
    }
  }

  auto *RowMaskOpnd = TII->getNamedOperand(MovMI, AMDGPU::OpName::row_mask);
  assert(RowMaskOpnd && RowMaskOpnd->isImm());
  auto *BankMaskOpnd = TII->getNamedOperand(MovMI, AMDGPU::OpName::bank_mask);
  assert(BankMaskOpnd && BankMaskOpnd->isImm());
  const bool MaskAllLanes = RowMaskOpnd->getImm() == 0xF &&
                            BankMaskOpnd->getImm() == 0xF;

  auto *BCZOpnd = TII->getNamedOperand(MovMI, AMDGPU::OpName::bound_ctrl);
  assert(BCZOpnd && BCZOpnd->isImm());
  bool BoundCtrlZero = BCZOpnd->getImm();

  auto *OldOpnd = TII->getNamedOperand(MovMI, AMDGPU::OpName::old);
  auto *SrcOpnd = TII->getNamedOperand(MovMI, AMDGPU::OpName::src0);
  assert(OldOpnd && OldOpnd->isReg());
  assert(SrcOpnd && SrcOpnd->isReg());
  if (OldOpnd->getReg().isPhysical() || SrcOpnd->getReg().isPhysical()) {
    LLVM_DEBUG(dbgs() << "  failed: dpp move reads physreg\n");
    return false;
  }

  auto * const OldOpndValue = getOldOpndValue(*OldOpnd);
  // OldOpndValue is either undef (IMPLICIT_DEF) or immediate or something else
  // We could use: assert(!OldOpndValue || OldOpndValue->isImm())
  // but the third option is used to distinguish undef from non-immediate
  // to reuse IMPLICIT_DEF instruction later
  assert(!OldOpndValue || OldOpndValue->isImm() || OldOpndValue == OldOpnd);

  bool CombBCZ = false;

  if (MaskAllLanes && BoundCtrlZero) { // [1]
    CombBCZ = true;
  } else {
    if (!OldOpndValue || !OldOpndValue->isImm()) {
      LLVM_DEBUG(dbgs() << "  failed: the DPP mov isn't combinable\n");
      return false;
    }

    if (OldOpndValue->getParent()->getParent() != MovMI.getParent()) {
      LLVM_DEBUG(dbgs() <<
        "  failed: old reg def and mov should be in the same BB\n");
      return false;
    }

    if (OldOpndValue->getImm() == 0) {
      if (MaskAllLanes) {
        assert(!BoundCtrlZero); // by check [1]
        CombBCZ = true;
      }
    } else if (BoundCtrlZero) {
      assert(!MaskAllLanes); // by check [1]
      LLVM_DEBUG(dbgs() <<
        "  failed: old!=0 and bctrl:0 and not all lanes isn't combinable\n");
      return false;
    }
  }

  LLVM_DEBUG(dbgs() << "  old=";
    if (!OldOpndValue)
      dbgs() << "undef";
    else
      dbgs() << *OldOpndValue;
    dbgs() << ", bound_ctrl=" << CombBCZ << '\n');

  SmallVector<MachineInstr*, 4> OrigMIs, DPPMIs;
  DenseMap<MachineInstr*, SmallVector<unsigned, 4>> RegSeqWithOpNos;
  auto CombOldVGPR = getRegSubRegPair(*OldOpnd);
  // try to reuse previous old reg if its undefined (IMPLICIT_DEF)
  if (CombBCZ && OldOpndValue) { // CombOldVGPR should be undef
    const TargetRegisterClass *RC = MRI->getRegClass(DPPMovReg);
    CombOldVGPR = RegSubRegPair(
      MRI->createVirtualRegister(RC));
    auto UndefInst = BuildMI(*MovMI.getParent(), MovMI, MovMI.getDebugLoc(),
                             TII->get(AMDGPU::IMPLICIT_DEF), CombOldVGPR.Reg);
    DPPMIs.push_back(UndefInst.getInstr());
  }

  OrigMIs.push_back(&MovMI);
  bool Rollback = true;
  SmallVector<MachineOperand*, 16> Uses;

  for (auto &Use : MRI->use_nodbg_operands(DPPMovReg)) {
    Uses.push_back(&Use);
  }

  while (!Uses.empty()) {
    MachineOperand *Use = Uses.pop_back_val();
    Rollback = true;

    auto &OrigMI = *Use->getParent();
    LLVM_DEBUG(dbgs() << "  try: " << OrigMI);

    auto OrigOp = OrigMI.getOpcode();
    if (OrigOp == AMDGPU::REG_SEQUENCE) {
      Register FwdReg = OrigMI.getOperand(0).getReg();
      unsigned FwdSubReg = 0;

      if (execMayBeModifiedBeforeAnyUse(*MRI, FwdReg, OrigMI)) {
        LLVM_DEBUG(dbgs() << "  failed: EXEC mask should remain the same"
                             " for all uses\n");
        break;
      }

      unsigned OpNo, E = OrigMI.getNumOperands();
      for (OpNo = 1; OpNo < E; OpNo += 2) {
        if (OrigMI.getOperand(OpNo).getReg() == DPPMovReg) {
          FwdSubReg = OrigMI.getOperand(OpNo + 1).getImm();
          break;
        }
      }

      if (!FwdSubReg)
        break;

      for (auto &Op : MRI->use_nodbg_operands(FwdReg)) {
        if (Op.getSubReg() == FwdSubReg)
          Uses.push_back(&Op);
      }
      RegSeqWithOpNos[&OrigMI].push_back(OpNo);
      continue;
    }

    bool IsShrinkable = isShrinkable(OrigMI);
    if (!(IsShrinkable || TII->isVOP1(OrigOp) || TII->isVOP2(OrigOp))) {
      LLVM_DEBUG(dbgs() << "  failed: not VOP1/2/3\n");
      break;
    }

    auto *Src0 = TII->getNamedOperand(OrigMI, AMDGPU::OpName::src0);
    auto *Src1 = TII->getNamedOperand(OrigMI, AMDGPU::OpName::src1);
    if (Use != Src0 && !(Use == Src1 && OrigMI.isCommutable())) { // [1]
      LLVM_DEBUG(dbgs() << "  failed: no suitable operands\n");
      break;
    }

    assert(Src0 && "Src1 without Src0?");
    if (Src1 && Src1->isIdenticalTo(*Src0)) {
      assert(Src1->isReg());
      LLVM_DEBUG(
          dbgs()
          << "  " << OrigMI
          << "  failed: DPP register is used more than once per instruction\n");
      break;
    }

    LLVM_DEBUG(dbgs() << "  combining: " << OrigMI);
    if (Use == Src0) {
      if (auto *DPPInst = createDPPInst(OrigMI, MovMI, CombOldVGPR,
                                        OldOpndValue, CombBCZ, IsShrinkable)) {
        DPPMIs.push_back(DPPInst);
        Rollback = false;
      }
    } else {
      assert(Use == Src1 && OrigMI.isCommutable()); // by check [1]
      auto *BB = OrigMI.getParent();
      auto *NewMI = BB->getParent()->CloneMachineInstr(&OrigMI);
      BB->insert(OrigMI, NewMI);
      if (TII->commuteInstruction(*NewMI)) {
        LLVM_DEBUG(dbgs() << "  commuted:  " << *NewMI);
        if (auto *DPPInst =
                createDPPInst(*NewMI, MovMI, CombOldVGPR, OldOpndValue, CombBCZ,
                              IsShrinkable)) {
          DPPMIs.push_back(DPPInst);
          Rollback = false;
        }
      } else
        LLVM_DEBUG(dbgs() << "  failed: cannot be commuted\n");
      NewMI->eraseFromParent();
    }
    if (Rollback)
      break;
    OrigMIs.push_back(&OrigMI);
  }

  Rollback |= !Uses.empty();

  for (auto *MI : *(Rollback? &DPPMIs : &OrigMIs))
    MI->eraseFromParent();

  if (!Rollback) {
    for (auto &S : RegSeqWithOpNos) {
      if (MRI->use_nodbg_empty(S.first->getOperand(0).getReg())) {
        S.first->eraseFromParent();
        continue;
      }
      while (!S.second.empty())
        S.first->getOperand(S.second.pop_back_val()).setIsUndef(true);
    }
  }

  return !Rollback;
}

bool GCNDPPCombine::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  if (!ST->hasDPP() || skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  TII = ST->getInstrInfo();

  bool Changed = false;
  for (auto &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(llvm::reverse(MBB))) {
      if (MI.getOpcode() == AMDGPU::V_MOV_B32_dpp && combineDPPMov(MI)) {
        Changed = true;
        ++NumDPPMovsCombined;
      } else if (MI.getOpcode() == AMDGPU::V_MOV_B64_DPP_PSEUDO) {
        if (ST->has64BitDPP() && combineDPPMov(MI)) {
          Changed = true;
          ++NumDPPMovsCombined;
        } else {
          auto Split = TII->expandMovDPP64(MI);
          for (auto M : { Split.first, Split.second }) {
            if (M && combineDPPMov(*M))
              ++NumDPPMovsCombined;
          }
          Changed = true;
        }
      }
    }
  }
  return Changed;
}
