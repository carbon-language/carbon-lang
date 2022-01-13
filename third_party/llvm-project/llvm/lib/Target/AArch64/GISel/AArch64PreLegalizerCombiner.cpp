//=== lib/CodeGen/GlobalISel/AArch64PreLegalizerCombiner.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass does combining of machine instructions at the generic MI level,
// before the legalizer.
//
//===----------------------------------------------------------------------===//

#include "AArch64GlobalISelUtils.h"
#include "AArch64TargetMachine.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aarch64-prelegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

/// Return true if a G_FCONSTANT instruction is known to be better-represented
/// as a G_CONSTANT.
static bool matchFConstantToConstant(MachineInstr &MI,
                                     MachineRegisterInfo &MRI) {
  assert(MI.getOpcode() == TargetOpcode::G_FCONSTANT);
  Register DstReg = MI.getOperand(0).getReg();
  const unsigned DstSize = MRI.getType(DstReg).getSizeInBits();
  if (DstSize != 32 && DstSize != 64)
    return false;

  // When we're storing a value, it doesn't matter what register bank it's on.
  // Since not all floating point constants can be materialized using a fmov,
  // it makes more sense to just use a GPR.
  return all_of(MRI.use_nodbg_instructions(DstReg),
                [](const MachineInstr &Use) { return Use.mayStore(); });
}

/// Change a G_FCONSTANT into a G_CONSTANT.
static void applyFConstantToConstant(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_FCONSTANT);
  MachineIRBuilder MIB(MI);
  const APFloat &ImmValAPF = MI.getOperand(1).getFPImm()->getValueAPF();
  MIB.buildConstant(MI.getOperand(0).getReg(), ImmValAPF.bitcastToAPInt());
  MI.eraseFromParent();
}

/// Try to match a G_ICMP of a G_TRUNC with zero, in which the truncated bits
/// are sign bits. In this case, we can transform the G_ICMP to directly compare
/// the wide value with a zero.
static bool matchICmpRedundantTrunc(MachineInstr &MI, MachineRegisterInfo &MRI,
                                    GISelKnownBits *KB, Register &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_ICMP && KB);

  auto Pred = (CmpInst::Predicate)MI.getOperand(1).getPredicate();
  if (!ICmpInst::isEquality(Pred))
    return false;

  Register LHS = MI.getOperand(2).getReg();
  LLT LHSTy = MRI.getType(LHS);
  if (!LHSTy.isScalar())
    return false;

  Register RHS = MI.getOperand(3).getReg();
  Register WideReg;

  if (!mi_match(LHS, MRI, m_GTrunc(m_Reg(WideReg))) ||
      !mi_match(RHS, MRI, m_SpecificICst(0)))
    return false;

  LLT WideTy = MRI.getType(WideReg);
  if (KB->computeNumSignBits(WideReg) <=
      WideTy.getSizeInBits() - LHSTy.getSizeInBits())
    return false;

  MatchInfo = WideReg;
  return true;
}

static bool applyICmpRedundantTrunc(MachineInstr &MI, MachineRegisterInfo &MRI,
                                    MachineIRBuilder &Builder,
                                    GISelChangeObserver &Observer,
                                    Register &WideReg) {
  assert(MI.getOpcode() == TargetOpcode::G_ICMP);

  LLT WideTy = MRI.getType(WideReg);
  // We're going to directly use the wide register as the LHS, and then use an
  // equivalent size zero for RHS.
  Builder.setInstrAndDebugLoc(MI);
  auto WideZero = Builder.buildConstant(WideTy, 0);
  Observer.changingInstr(MI);
  MI.getOperand(2).setReg(WideReg);
  MI.getOperand(3).setReg(WideZero.getReg(0));
  Observer.changedInstr(MI);
  return true;
}

/// \returns true if it is possible to fold a constant into a G_GLOBAL_VALUE.
///
/// e.g.
///
/// %g = G_GLOBAL_VALUE @x -> %g = G_GLOBAL_VALUE @x + cst
static bool matchFoldGlobalOffset(MachineInstr &MI, MachineRegisterInfo &MRI,
                                  std::pair<uint64_t, uint64_t> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_GLOBAL_VALUE);
  MachineFunction &MF = *MI.getMF();
  auto &GlobalOp = MI.getOperand(1);
  auto *GV = GlobalOp.getGlobal();
  if (GV->isThreadLocal())
    return false;

  // Don't allow anything that could represent offsets etc.
  if (MF.getSubtarget<AArch64Subtarget>().ClassifyGlobalReference(
          GV, MF.getTarget()) != AArch64II::MO_NO_FLAG)
    return false;

  // Look for a G_GLOBAL_VALUE only used by G_PTR_ADDs against constants:
  //
  //  %g = G_GLOBAL_VALUE @x
  //  %ptr1 = G_PTR_ADD %g, cst1
  //  %ptr2 = G_PTR_ADD %g, cst2
  //  ...
  //  %ptrN = G_PTR_ADD %g, cstN
  //
  // Identify the *smallest* constant. We want to be able to form this:
  //
  //  %offset_g = G_GLOBAL_VALUE @x + min_cst
  //  %g = G_PTR_ADD %offset_g, -min_cst
  //  %ptr1 = G_PTR_ADD %g, cst1
  //  ...
  Register Dst = MI.getOperand(0).getReg();
  uint64_t MinOffset = -1ull;
  for (auto &UseInstr : MRI.use_nodbg_instructions(Dst)) {
    if (UseInstr.getOpcode() != TargetOpcode::G_PTR_ADD)
      return false;
    auto Cst = getIConstantVRegValWithLookThrough(
        UseInstr.getOperand(2).getReg(), MRI);
    if (!Cst)
      return false;
    MinOffset = std::min(MinOffset, Cst->Value.getZExtValue());
  }

  // Require that the new offset is larger than the existing one to avoid
  // infinite loops.
  uint64_t CurrOffset = GlobalOp.getOffset();
  uint64_t NewOffset = MinOffset + CurrOffset;
  if (NewOffset <= CurrOffset)
    return false;

  // Check whether folding this offset is legal. It must not go out of bounds of
  // the referenced object to avoid violating the code model, and must be
  // smaller than 2^21 because this is the largest offset expressible in all
  // object formats.
  //
  // This check also prevents us from folding negative offsets, which will end
  // up being treated in the same way as large positive ones. They could also
  // cause code model violations, and aren't really common enough to matter.
  if (NewOffset >= (1 << 21))
    return false;

  Type *T = GV->getValueType();
  if (!T->isSized() ||
      NewOffset > GV->getParent()->getDataLayout().getTypeAllocSize(T))
    return false;
  MatchInfo = std::make_pair(NewOffset, MinOffset);
  return true;
}

static bool applyFoldGlobalOffset(MachineInstr &MI, MachineRegisterInfo &MRI,
                                  MachineIRBuilder &B,
                                  GISelChangeObserver &Observer,
                                  std::pair<uint64_t, uint64_t> &MatchInfo) {
  // Change:
  //
  //  %g = G_GLOBAL_VALUE @x
  //  %ptr1 = G_PTR_ADD %g, cst1
  //  %ptr2 = G_PTR_ADD %g, cst2
  //  ...
  //  %ptrN = G_PTR_ADD %g, cstN
  //
  // To:
  //
  //  %offset_g = G_GLOBAL_VALUE @x + min_cst
  //  %g = G_PTR_ADD %offset_g, -min_cst
  //  %ptr1 = G_PTR_ADD %g, cst1
  //  ...
  //  %ptrN = G_PTR_ADD %g, cstN
  //
  // Then, the original G_PTR_ADDs should be folded later on so that they look
  // like this:
  //
  //  %ptrN = G_PTR_ADD %offset_g, cstN - min_cst
  uint64_t Offset, MinOffset;
  std::tie(Offset, MinOffset) = MatchInfo;
  B.setInstrAndDebugLoc(MI);
  Observer.changingInstr(MI);
  auto &GlobalOp = MI.getOperand(1);
  auto *GV = GlobalOp.getGlobal();
  GlobalOp.ChangeToGA(GV, Offset, GlobalOp.getTargetFlags());
  Register Dst = MI.getOperand(0).getReg();
  Register NewGVDst = MRI.cloneVirtualRegister(Dst);
  MI.getOperand(0).setReg(NewGVDst);
  Observer.changedInstr(MI);
  B.buildPtrAdd(
      Dst, NewGVDst,
      B.buildConstant(LLT::scalar(64), -static_cast<int64_t>(MinOffset)));
  return true;
}

static bool tryToSimplifyUADDO(MachineInstr &MI, MachineIRBuilder &B,
                               CombinerHelper &Helper,
                               GISelChangeObserver &Observer) {
  // Try simplify G_UADDO with 8 or 16 bit operands to wide G_ADD and TBNZ if
  // result is only used in the no-overflow case. It is restricted to cases
  // where we know that the high-bits of the operands are 0. If there's an
  // overflow, then the the 9th or 17th bit must be set, which can be checked
  // using TBNZ.
  //
  // Change (for UADDOs on 8 and 16 bits):
  //
  //   %z0 = G_ASSERT_ZEXT _
  //   %op0 = G_TRUNC %z0
  //   %z1 = G_ASSERT_ZEXT _
  //   %op1 = G_TRUNC %z1
  //   %val, %cond = G_UADDO %op0, %op1
  //   G_BRCOND %cond, %error.bb
  //
  // error.bb:
  //   (no successors and no uses of %val)
  //
  // To:
  //
  //   %z0 = G_ASSERT_ZEXT _
  //   %z1 = G_ASSERT_ZEXT _
  //   %add = G_ADD %z0, %z1
  //   %val = G_TRUNC %add
  //   %bit = G_AND %add, 1 << scalar-size-in-bits(%op1)
  //   %cond = G_ICMP NE, %bit, 0
  //   G_BRCOND %cond, %error.bb

  auto &MRI = *B.getMRI();

  MachineOperand *DefOp0 = MRI.getOneDef(MI.getOperand(2).getReg());
  MachineOperand *DefOp1 = MRI.getOneDef(MI.getOperand(3).getReg());
  Register Op0Wide;
  Register Op1Wide;
  if (!mi_match(DefOp0->getParent(), MRI, m_GTrunc(m_Reg(Op0Wide))) ||
      !mi_match(DefOp1->getParent(), MRI, m_GTrunc(m_Reg(Op1Wide))))
    return false;
  LLT WideTy0 = MRI.getType(Op0Wide);
  LLT WideTy1 = MRI.getType(Op1Wide);
  Register ResVal = MI.getOperand(0).getReg();
  LLT OpTy = MRI.getType(ResVal);
  MachineInstr *Op0WideDef = MRI.getVRegDef(Op0Wide);
  MachineInstr *Op1WideDef = MRI.getVRegDef(Op1Wide);

  unsigned OpTySize = OpTy.getScalarSizeInBits();
  // First check that the G_TRUNC feeding the G_UADDO are no-ops, because the
  // inputs have been zero-extended.
  if (Op0WideDef->getOpcode() != TargetOpcode::G_ASSERT_ZEXT ||
      Op1WideDef->getOpcode() != TargetOpcode::G_ASSERT_ZEXT ||
      OpTySize != Op0WideDef->getOperand(2).getImm() ||
      OpTySize != Op1WideDef->getOperand(2).getImm())
    return false;

  // Only scalar UADDO with either 8 or 16 bit operands are handled.
  if (!WideTy0.isScalar() || !WideTy1.isScalar() || WideTy0 != WideTy1 ||
      OpTySize >= WideTy0.getScalarSizeInBits() ||
      (OpTySize != 8 && OpTySize != 16))
    return false;

  // The overflow-status result must be used by a branch only.
  Register ResStatus = MI.getOperand(1).getReg();
  if (!MRI.hasOneNonDBGUse(ResStatus))
    return false;
  MachineInstr *CondUser = &*MRI.use_instr_nodbg_begin(ResStatus);
  if (CondUser->getOpcode() != TargetOpcode::G_BRCOND)
    return false;

  // Make sure the computed result is only used in the no-overflow blocks.
  MachineBasicBlock *CurrentMBB = MI.getParent();
  MachineBasicBlock *FailMBB = CondUser->getOperand(1).getMBB();
  if (!FailMBB->succ_empty() || CondUser->getParent() != CurrentMBB)
    return false;
  if (any_of(MRI.use_nodbg_instructions(ResVal),
             [&MI, FailMBB, CurrentMBB](MachineInstr &I) {
               return &MI != &I &&
                      (I.getParent() == FailMBB || I.getParent() == CurrentMBB);
             }))
    return false;

  // Remove G_ADDO.
  B.setInstrAndDebugLoc(*MI.getNextNode());
  MI.eraseFromParent();

  // Emit wide add.
  Register AddDst = MRI.cloneVirtualRegister(Op0Wide);
  B.buildInstr(TargetOpcode::G_ADD, {AddDst}, {Op0Wide, Op1Wide});

  // Emit check of the 9th or 17th bit and update users (the branch). This will
  // later be folded to TBNZ.
  Register CondBit = MRI.cloneVirtualRegister(Op0Wide);
  B.buildAnd(
      CondBit, AddDst,
      B.buildConstant(LLT::scalar(32), OpTySize == 8 ? 1 << 8 : 1 << 16));
  B.buildICmp(CmpInst::ICMP_NE, ResStatus, CondBit,
              B.buildConstant(LLT::scalar(32), 0));

  // Update ZEXts users of the result value. Because all uses are in the
  // no-overflow case, we know that the top bits are 0 and we can ignore ZExts.
  B.buildZExtOrTrunc(ResVal, AddDst);
  for (MachineOperand &U : make_early_inc_range(MRI.use_operands(ResVal))) {
    Register WideReg;
    if (mi_match(U.getParent(), MRI, m_GZExt(m_Reg(WideReg)))) {
      auto OldR = U.getParent()->getOperand(0).getReg();
      Observer.erasingInstr(*U.getParent());
      U.getParent()->eraseFromParent();
      Helper.replaceRegWith(MRI, OldR, AddDst);
    }
  }

  return true;
}

class AArch64PreLegalizerCombinerHelperState {
protected:
  CombinerHelper &Helper;

public:
  AArch64PreLegalizerCombinerHelperState(CombinerHelper &Helper)
      : Helper(Helper) {}
};

#define AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "AArch64GenPreLegalizeGICombiner.inc"
#undef AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
#include "AArch64GenPreLegalizeGICombiner.inc"
#undef AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

class AArch64PreLegalizerCombinerInfo : public CombinerInfo {
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;
  AArch64GenPreLegalizerCombinerHelperRuleConfig GeneratedRuleCfg;

public:
  AArch64PreLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
                                  GISelKnownBits *KB, MachineDominatorTree *MDT)
      : CombinerInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, OptSize, MinSize),
        KB(KB), MDT(MDT) {
    if (!GeneratedRuleCfg.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  virtual bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
                       MachineIRBuilder &B) const override;
};

bool AArch64PreLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                              MachineInstr &MI,
                                              MachineIRBuilder &B) const {
  CombinerHelper Helper(Observer, B, KB, MDT);
  AArch64GenPreLegalizerCombinerHelper Generated(GeneratedRuleCfg, Helper);

  if (Generated.tryCombineAll(Observer, MI, B))
    return true;

  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case TargetOpcode::G_CONCAT_VECTORS:
    return Helper.tryCombineConcatVectors(MI);
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return Helper.tryCombineShuffleVector(MI);
  case TargetOpcode::G_UADDO:
    return tryToSimplifyUADDO(MI, B, Helper, Observer);
  case TargetOpcode::G_MEMCPY_INLINE:
    return Helper.tryEmitMemcpyInline(MI);
  case TargetOpcode::G_MEMCPY:
  case TargetOpcode::G_MEMMOVE:
  case TargetOpcode::G_MEMSET: {
    // If we're at -O0 set a maxlen of 32 to inline, otherwise let the other
    // heuristics decide.
    unsigned MaxLen = EnableOpt ? 0 : 32;
    // Try to inline memcpy type calls if optimizations are enabled.
    if (Helper.tryCombineMemCpyFamily(MI, MaxLen))
      return true;
    if (Opc == TargetOpcode::G_MEMSET)
      return llvm::AArch64GISelUtils::tryEmitBZero(MI, B, EnableMinSize);
    return false;
  }
  }

  return false;
}

#define AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP
#include "AArch64GenPreLegalizeGICombiner.inc"
#undef AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP

// Pass boilerplate
// ================

class AArch64PreLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  AArch64PreLegalizerCombiner();

  StringRef getPassName() const override { return "AArch64PreLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

void AArch64PreLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelKnownBitsAnalysis>();
  AU.addPreserved<GISelKnownBitsAnalysis>();
  AU.addRequired<MachineDominatorTree>();
  AU.addPreserved<MachineDominatorTree>();
  AU.addRequired<GISelCSEAnalysisWrapperPass>();
  AU.addPreserved<GISelCSEAnalysisWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

AArch64PreLegalizerCombiner::AArch64PreLegalizerCombiner()
    : MachineFunctionPass(ID) {
  initializeAArch64PreLegalizerCombinerPass(*PassRegistry::getPassRegistry());
}

bool AArch64PreLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  auto &TPC = getAnalysis<TargetPassConfig>();

  // Enable CSE.
  GISelCSEAnalysisWrapper &Wrapper =
      getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
  auto *CSEInfo = &Wrapper.get(TPC.getCSEConfig());

  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOpt::None && !skipFunction(F);
  GISelKnownBits *KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);
  MachineDominatorTree *MDT = &getAnalysis<MachineDominatorTree>();
  AArch64PreLegalizerCombinerInfo PCInfo(EnableOpt, F.hasOptSize(),
                                         F.hasMinSize(), KB, MDT);
  Combiner C(PCInfo, &TPC);
  return C.combineMachineInstrs(MF, CSEInfo);
}

char AArch64PreLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(AArch64PreLegalizerCombiner, DEBUG_TYPE,
                      "Combine AArch64 machine instrs before legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_END(AArch64PreLegalizerCombiner, DEBUG_TYPE,
                    "Combine AArch64 machine instrs before legalization", false,
                    false)


namespace llvm {
FunctionPass *createAArch64PreLegalizerCombiner() {
  return new AArch64PreLegalizerCombiner();
}
} // end namespace llvm
