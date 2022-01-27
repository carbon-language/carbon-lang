//=== AArch64PostLegalizerLowering.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Post-legalization lowering for instructions.
///
/// This is used to offload pattern matching from the selector.
///
/// For example, this combiner will notice that a G_SHUFFLE_VECTOR is actually
/// a G_ZIP, G_UZP, etc.
///
/// General optimization combines should be handled by either the
/// AArch64PostLegalizerCombiner or the AArch64PreLegalizerCombiner.
///
//===----------------------------------------------------------------------===//

#include "AArch64GlobalISelUtils.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "GISel/AArch64LegalizerInfo.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "TargetInfo/AArch64TargetInfo.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "aarch64-postlegalizer-lowering"

using namespace llvm;
using namespace MIPatternMatch;
using namespace AArch64GISelUtils;

/// Represents a pseudo instruction which replaces a G_SHUFFLE_VECTOR.
///
/// Used for matching target-supported shuffles before codegen.
struct ShuffleVectorPseudo {
  unsigned Opc; ///< Opcode for the instruction. (E.g. G_ZIP1)
  Register Dst; ///< Destination register.
  SmallVector<SrcOp, 2> SrcOps; ///< Source registers.
  ShuffleVectorPseudo(unsigned Opc, Register Dst,
                      std::initializer_list<SrcOp> SrcOps)
      : Opc(Opc), Dst(Dst), SrcOps(SrcOps){};
  ShuffleVectorPseudo() {}
};

/// Check if a vector shuffle corresponds to a REV instruction with the
/// specified blocksize.
static bool isREVMask(ArrayRef<int> M, unsigned EltSize, unsigned NumElts,
                      unsigned BlockSize) {
  assert((BlockSize == 16 || BlockSize == 32 || BlockSize == 64) &&
         "Only possible block sizes for REV are: 16, 32, 64");
  assert(EltSize != 64 && "EltSize cannot be 64 for REV mask.");

  unsigned BlockElts = M[0] + 1;

  // If the first shuffle index is UNDEF, be optimistic.
  if (M[0] < 0)
    BlockElts = BlockSize / EltSize;

  if (BlockSize <= EltSize || BlockSize != BlockElts * EltSize)
    return false;

  for (unsigned i = 0; i < NumElts; ++i) {
    // Ignore undef indices.
    if (M[i] < 0)
      continue;
    if (static_cast<unsigned>(M[i]) !=
        (i - i % BlockElts) + (BlockElts - 1 - i % BlockElts))
      return false;
  }

  return true;
}

/// Determines if \p M is a shuffle vector mask for a TRN of \p NumElts.
/// Whether or not G_TRN1 or G_TRN2 should be used is stored in \p WhichResult.
static bool isTRNMask(ArrayRef<int> M, unsigned NumElts,
                      unsigned &WhichResult) {
  if (NumElts % 2 != 0)
    return false;
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i < NumElts; i += 2) {
    if ((M[i] >= 0 && static_cast<unsigned>(M[i]) != i + WhichResult) ||
        (M[i + 1] >= 0 &&
         static_cast<unsigned>(M[i + 1]) != i + NumElts + WhichResult))
      return false;
  }
  return true;
}

/// Check if a G_EXT instruction can handle a shuffle mask \p M when the vector
/// sources of the shuffle are different.
static Optional<std::pair<bool, uint64_t>> getExtMask(ArrayRef<int> M,
                                                      unsigned NumElts) {
  // Look for the first non-undef element.
  auto FirstRealElt = find_if(M, [](int Elt) { return Elt >= 0; });
  if (FirstRealElt == M.end())
    return None;

  // Use APInt to handle overflow when calculating expected element.
  unsigned MaskBits = APInt(32, NumElts * 2).logBase2();
  APInt ExpectedElt = APInt(MaskBits, *FirstRealElt + 1);

  // The following shuffle indices must be the successive elements after the
  // first real element.
  if (any_of(
          make_range(std::next(FirstRealElt), M.end()),
          [&ExpectedElt](int Elt) { return Elt != ExpectedElt++ && Elt >= 0; }))
    return None;

  // The index of an EXT is the first element if it is not UNDEF.
  // Watch out for the beginning UNDEFs. The EXT index should be the expected
  // value of the first element.  E.g.
  // <-1, -1, 3, ...> is treated as <1, 2, 3, ...>.
  // <-1, -1, 0, 1, ...> is treated as <2*NumElts-2, 2*NumElts-1, 0, 1, ...>.
  // ExpectedElt is the last mask index plus 1.
  uint64_t Imm = ExpectedElt.getZExtValue();
  bool ReverseExt = false;

  // There are two difference cases requiring to reverse input vectors.
  // For example, for vector <4 x i32> we have the following cases,
  // Case 1: shufflevector(<4 x i32>,<4 x i32>,<-1, -1, -1, 0>)
  // Case 2: shufflevector(<4 x i32>,<4 x i32>,<-1, -1, 7, 0>)
  // For both cases, we finally use mask <5, 6, 7, 0>, which requires
  // to reverse two input vectors.
  if (Imm < NumElts)
    ReverseExt = true;
  else
    Imm -= NumElts;
  return std::make_pair(ReverseExt, Imm);
}

/// Determines if \p M is a shuffle vector mask for a UZP of \p NumElts.
/// Whether or not G_UZP1 or G_UZP2 should be used is stored in \p WhichResult.
static bool isUZPMask(ArrayRef<int> M, unsigned NumElts,
                      unsigned &WhichResult) {
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i != NumElts; ++i) {
    // Skip undef indices.
    if (M[i] < 0)
      continue;
    if (static_cast<unsigned>(M[i]) != 2 * i + WhichResult)
      return false;
  }
  return true;
}

/// \return true if \p M is a zip mask for a shuffle vector of \p NumElts.
/// Whether or not G_ZIP1 or G_ZIP2 should be used is stored in \p WhichResult.
static bool isZipMask(ArrayRef<int> M, unsigned NumElts,
                      unsigned &WhichResult) {
  if (NumElts % 2 != 0)
    return false;

  // 0 means use ZIP1, 1 means use ZIP2.
  WhichResult = (M[0] == 0 ? 0 : 1);
  unsigned Idx = WhichResult * NumElts / 2;
  for (unsigned i = 0; i != NumElts; i += 2) {
      if ((M[i] >= 0 && static_cast<unsigned>(M[i]) != Idx) ||
          (M[i + 1] >= 0 && static_cast<unsigned>(M[i + 1]) != Idx + NumElts))
        return false;
    Idx += 1;
  }
  return true;
}

/// Helper function for matchINS.
///
/// \returns a value when \p M is an ins mask for \p NumInputElements.
///
/// First element of the returned pair is true when the produced
/// G_INSERT_VECTOR_ELT destination should be the LHS of the G_SHUFFLE_VECTOR.
///
/// Second element is the destination lane for the G_INSERT_VECTOR_ELT.
static Optional<std::pair<bool, int>> isINSMask(ArrayRef<int> M,
                                                int NumInputElements) {
  if (M.size() != static_cast<size_t>(NumInputElements))
    return None;
  int NumLHSMatch = 0, NumRHSMatch = 0;
  int LastLHSMismatch = -1, LastRHSMismatch = -1;
  for (int Idx = 0; Idx < NumInputElements; ++Idx) {
    if (M[Idx] == -1) {
      ++NumLHSMatch;
      ++NumRHSMatch;
      continue;
    }
    M[Idx] == Idx ? ++NumLHSMatch : LastLHSMismatch = Idx;
    M[Idx] == Idx + NumInputElements ? ++NumRHSMatch : LastRHSMismatch = Idx;
  }
  const int NumNeededToMatch = NumInputElements - 1;
  if (NumLHSMatch == NumNeededToMatch)
    return std::make_pair(true, LastLHSMismatch);
  if (NumRHSMatch == NumNeededToMatch)
    return std::make_pair(false, LastRHSMismatch);
  return None;
}

/// \return true if a G_SHUFFLE_VECTOR instruction \p MI can be replaced with a
/// G_REV instruction. Returns the appropriate G_REV opcode in \p Opc.
static bool matchREV(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  ArrayRef<int> ShuffleMask = MI.getOperand(3).getShuffleMask();
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  LLT Ty = MRI.getType(Dst);
  unsigned EltSize = Ty.getScalarSizeInBits();

  // Element size for a rev cannot be 64.
  if (EltSize == 64)
    return false;

  unsigned NumElts = Ty.getNumElements();

  // Try to produce G_REV64
  if (isREVMask(ShuffleMask, EltSize, NumElts, 64)) {
    MatchInfo = ShuffleVectorPseudo(AArch64::G_REV64, Dst, {Src});
    return true;
  }

  // TODO: Produce G_REV32 and G_REV16 once we have proper legalization support.
  // This should be identical to above, but with a constant 32 and constant
  // 16.
  return false;
}

/// \return true if a G_SHUFFLE_VECTOR instruction \p MI can be replaced with
/// a G_TRN1 or G_TRN2 instruction.
static bool matchTRN(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  unsigned WhichResult;
  ArrayRef<int> ShuffleMask = MI.getOperand(3).getShuffleMask();
  Register Dst = MI.getOperand(0).getReg();
  unsigned NumElts = MRI.getType(Dst).getNumElements();
  if (!isTRNMask(ShuffleMask, NumElts, WhichResult))
    return false;
  unsigned Opc = (WhichResult == 0) ? AArch64::G_TRN1 : AArch64::G_TRN2;
  Register V1 = MI.getOperand(1).getReg();
  Register V2 = MI.getOperand(2).getReg();
  MatchInfo = ShuffleVectorPseudo(Opc, Dst, {V1, V2});
  return true;
}

/// \return true if a G_SHUFFLE_VECTOR instruction \p MI can be replaced with
/// a G_UZP1 or G_UZP2 instruction.
///
/// \param [in] MI - The shuffle vector instruction.
/// \param [out] MatchInfo - Either G_UZP1 or G_UZP2 on success.
static bool matchUZP(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  unsigned WhichResult;
  ArrayRef<int> ShuffleMask = MI.getOperand(3).getShuffleMask();
  Register Dst = MI.getOperand(0).getReg();
  unsigned NumElts = MRI.getType(Dst).getNumElements();
  if (!isUZPMask(ShuffleMask, NumElts, WhichResult))
    return false;
  unsigned Opc = (WhichResult == 0) ? AArch64::G_UZP1 : AArch64::G_UZP2;
  Register V1 = MI.getOperand(1).getReg();
  Register V2 = MI.getOperand(2).getReg();
  MatchInfo = ShuffleVectorPseudo(Opc, Dst, {V1, V2});
  return true;
}

static bool matchZip(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  unsigned WhichResult;
  ArrayRef<int> ShuffleMask = MI.getOperand(3).getShuffleMask();
  Register Dst = MI.getOperand(0).getReg();
  unsigned NumElts = MRI.getType(Dst).getNumElements();
  if (!isZipMask(ShuffleMask, NumElts, WhichResult))
    return false;
  unsigned Opc = (WhichResult == 0) ? AArch64::G_ZIP1 : AArch64::G_ZIP2;
  Register V1 = MI.getOperand(1).getReg();
  Register V2 = MI.getOperand(2).getReg();
  MatchInfo = ShuffleVectorPseudo(Opc, Dst, {V1, V2});
  return true;
}

/// Helper function for matchDup.
static bool matchDupFromInsertVectorElt(int Lane, MachineInstr &MI,
                                        MachineRegisterInfo &MRI,
                                        ShuffleVectorPseudo &MatchInfo) {
  if (Lane != 0)
    return false;

  // Try to match a vector splat operation into a dup instruction.
  // We're looking for this pattern:
  //
  // %scalar:gpr(s64) = COPY $x0
  // %undef:fpr(<2 x s64>) = G_IMPLICIT_DEF
  // %cst0:gpr(s32) = G_CONSTANT i32 0
  // %zerovec:fpr(<2 x s32>) = G_BUILD_VECTOR %cst0(s32), %cst0(s32)
  // %ins:fpr(<2 x s64>) = G_INSERT_VECTOR_ELT %undef, %scalar(s64), %cst0(s32)
  // %splat:fpr(<2 x s64>) = G_SHUFFLE_VECTOR %ins(<2 x s64>), %undef, %zerovec(<2 x s32>)
  //
  // ...into:
  // %splat = G_DUP %scalar

  // Begin matching the insert.
  auto *InsMI = getOpcodeDef(TargetOpcode::G_INSERT_VECTOR_ELT,
                             MI.getOperand(1).getReg(), MRI);
  if (!InsMI)
    return false;
  // Match the undef vector operand.
  if (!getOpcodeDef(TargetOpcode::G_IMPLICIT_DEF, InsMI->getOperand(1).getReg(),
                    MRI))
    return false;

  // Match the index constant 0.
  if (!mi_match(InsMI->getOperand(3).getReg(), MRI, m_ZeroInt()))
    return false;

  MatchInfo = ShuffleVectorPseudo(AArch64::G_DUP, MI.getOperand(0).getReg(),
                                  {InsMI->getOperand(2).getReg()});
  return true;
}

/// Helper function for matchDup.
static bool matchDupFromBuildVector(int Lane, MachineInstr &MI,
                                    MachineRegisterInfo &MRI,
                                    ShuffleVectorPseudo &MatchInfo) {
  assert(Lane >= 0 && "Expected positive lane?");
  // Test if the LHS is a BUILD_VECTOR. If it is, then we can just reference the
  // lane's definition directly.
  auto *BuildVecMI = getOpcodeDef(TargetOpcode::G_BUILD_VECTOR,
                                  MI.getOperand(1).getReg(), MRI);
  if (!BuildVecMI)
    return false;
  Register Reg = BuildVecMI->getOperand(Lane + 1).getReg();
  MatchInfo =
      ShuffleVectorPseudo(AArch64::G_DUP, MI.getOperand(0).getReg(), {Reg});
  return true;
}

static bool matchDup(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  auto MaybeLane = getSplatIndex(MI);
  if (!MaybeLane)
    return false;
  int Lane = *MaybeLane;
  // If this is undef splat, generate it via "just" vdup, if possible.
  if (Lane < 0)
    Lane = 0;
  if (matchDupFromInsertVectorElt(Lane, MI, MRI, MatchInfo))
    return true;
  if (matchDupFromBuildVector(Lane, MI, MRI, MatchInfo))
    return true;
  return false;
}

static bool matchEXT(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  Register Dst = MI.getOperand(0).getReg();
  auto ExtInfo = getExtMask(MI.getOperand(3).getShuffleMask(),
                            MRI.getType(Dst).getNumElements());
  if (!ExtInfo)
    return false;
  bool ReverseExt;
  uint64_t Imm;
  std::tie(ReverseExt, Imm) = *ExtInfo;
  Register V1 = MI.getOperand(1).getReg();
  Register V2 = MI.getOperand(2).getReg();
  if (ReverseExt)
    std::swap(V1, V2);
  uint64_t ExtFactor = MRI.getType(V1).getScalarSizeInBits() / 8;
  Imm *= ExtFactor;
  MatchInfo = ShuffleVectorPseudo(AArch64::G_EXT, Dst, {V1, V2, Imm});
  return true;
}

/// Replace a G_SHUFFLE_VECTOR instruction with a pseudo.
/// \p Opc is the opcode to use. \p MI is the G_SHUFFLE_VECTOR.
static bool applyShuffleVectorPseudo(MachineInstr &MI,
                                     ShuffleVectorPseudo &MatchInfo) {
  MachineIRBuilder MIRBuilder(MI);
  MIRBuilder.buildInstr(MatchInfo.Opc, {MatchInfo.Dst}, MatchInfo.SrcOps);
  MI.eraseFromParent();
  return true;
}

/// Replace a G_SHUFFLE_VECTOR instruction with G_EXT.
/// Special-cased because the constant operand must be emitted as a G_CONSTANT
/// for the imported tablegen patterns to work.
static bool applyEXT(MachineInstr &MI, ShuffleVectorPseudo &MatchInfo) {
  MachineIRBuilder MIRBuilder(MI);
  // Tablegen patterns expect an i32 G_CONSTANT as the final op.
  auto Cst =
      MIRBuilder.buildConstant(LLT::scalar(32), MatchInfo.SrcOps[2].getImm());
  MIRBuilder.buildInstr(MatchInfo.Opc, {MatchInfo.Dst},
                        {MatchInfo.SrcOps[0], MatchInfo.SrcOps[1], Cst});
  MI.eraseFromParent();
  return true;
}

/// Match a G_SHUFFLE_VECTOR with a mask which corresponds to a
/// G_INSERT_VECTOR_ELT and G_EXTRACT_VECTOR_ELT pair.
///
/// e.g.
///   %shuf = G_SHUFFLE_VECTOR %left, %right, shufflemask(0, 0)
///
/// Can be represented as
///
///   %extract = G_EXTRACT_VECTOR_ELT %left, 0
///   %ins = G_INSERT_VECTOR_ELT %left, %extract, 1
///
static bool matchINS(MachineInstr &MI, MachineRegisterInfo &MRI,
                     std::tuple<Register, int, Register, int> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  ArrayRef<int> ShuffleMask = MI.getOperand(3).getShuffleMask();
  Register Dst = MI.getOperand(0).getReg();
  int NumElts = MRI.getType(Dst).getNumElements();
  auto DstIsLeftAndDstLane = isINSMask(ShuffleMask, NumElts);
  if (!DstIsLeftAndDstLane)
    return false;
  bool DstIsLeft;
  int DstLane;
  std::tie(DstIsLeft, DstLane) = *DstIsLeftAndDstLane;
  Register Left = MI.getOperand(1).getReg();
  Register Right = MI.getOperand(2).getReg();
  Register DstVec = DstIsLeft ? Left : Right;
  Register SrcVec = Left;

  int SrcLane = ShuffleMask[DstLane];
  if (SrcLane >= NumElts) {
    SrcVec = Right;
    SrcLane -= NumElts;
  }

  MatchInfo = std::make_tuple(DstVec, DstLane, SrcVec, SrcLane);
  return true;
}

static bool applyINS(MachineInstr &MI, MachineRegisterInfo &MRI,
                     MachineIRBuilder &Builder,
                     std::tuple<Register, int, Register, int> &MatchInfo) {
  Builder.setInstrAndDebugLoc(MI);
  Register Dst = MI.getOperand(0).getReg();
  auto ScalarTy = MRI.getType(Dst).getElementType();
  Register DstVec, SrcVec;
  int DstLane, SrcLane;
  std::tie(DstVec, DstLane, SrcVec, SrcLane) = MatchInfo;
  auto SrcCst = Builder.buildConstant(LLT::scalar(64), SrcLane);
  auto Extract = Builder.buildExtractVectorElement(ScalarTy, SrcVec, SrcCst);
  auto DstCst = Builder.buildConstant(LLT::scalar(64), DstLane);
  Builder.buildInsertVectorElement(Dst, DstVec, Extract, DstCst);
  MI.eraseFromParent();
  return true;
}

/// isVShiftRImm - Check if this is a valid vector for the immediate
/// operand of a vector shift right operation. The value must be in the range:
///   1 <= Value <= ElementBits for a right shift.
static bool isVShiftRImm(Register Reg, MachineRegisterInfo &MRI, LLT Ty,
                         int64_t &Cnt) {
  assert(Ty.isVector() && "vector shift count is not a vector type");
  MachineInstr *MI = MRI.getVRegDef(Reg);
  auto Cst = getAArch64VectorSplatScalar(*MI, MRI);
  if (!Cst)
    return false;
  Cnt = *Cst;
  int64_t ElementBits = Ty.getScalarSizeInBits();
  return Cnt >= 1 && Cnt <= ElementBits;
}

/// Match a vector G_ASHR or G_LSHR with a valid immediate shift.
static bool matchVAshrLshrImm(MachineInstr &MI, MachineRegisterInfo &MRI,
                              int64_t &Imm) {
  assert(MI.getOpcode() == TargetOpcode::G_ASHR ||
         MI.getOpcode() == TargetOpcode::G_LSHR);
  LLT Ty = MRI.getType(MI.getOperand(1).getReg());
  if (!Ty.isVector())
    return false;
  return isVShiftRImm(MI.getOperand(2).getReg(), MRI, Ty, Imm);
}

static bool applyVAshrLshrImm(MachineInstr &MI, MachineRegisterInfo &MRI,
                              int64_t &Imm) {
  unsigned Opc = MI.getOpcode();
  assert(Opc == TargetOpcode::G_ASHR || Opc == TargetOpcode::G_LSHR);
  unsigned NewOpc =
      Opc == TargetOpcode::G_ASHR ? AArch64::G_VASHR : AArch64::G_VLSHR;
  MachineIRBuilder MIB(MI);
  auto ImmDef = MIB.buildConstant(LLT::scalar(32), Imm);
  MIB.buildInstr(NewOpc, {MI.getOperand(0)}, {MI.getOperand(1), ImmDef});
  MI.eraseFromParent();
  return true;
}

/// Determine if it is possible to modify the \p RHS and predicate \p P of a
/// G_ICMP instruction such that the right-hand side is an arithmetic immediate.
///
/// \returns A pair containing the updated immediate and predicate which may
/// be used to optimize the instruction.
///
/// \note This assumes that the comparison has been legalized.
Optional<std::pair<uint64_t, CmpInst::Predicate>>
tryAdjustICmpImmAndPred(Register RHS, CmpInst::Predicate P,
                          const MachineRegisterInfo &MRI) {
  const auto &Ty = MRI.getType(RHS);
  if (Ty.isVector())
    return None;
  unsigned Size = Ty.getSizeInBits();
  assert((Size == 32 || Size == 64) && "Expected 32 or 64 bit compare only?");

  // If the RHS is not a constant, or the RHS is already a valid arithmetic
  // immediate, then there is nothing to change.
  auto ValAndVReg = getIConstantVRegValWithLookThrough(RHS, MRI);
  if (!ValAndVReg)
    return None;
  uint64_t C = ValAndVReg->Value.getZExtValue();
  if (isLegalArithImmed(C))
    return None;

  // We have a non-arithmetic immediate. Check if adjusting the immediate and
  // adjusting the predicate will result in a legal arithmetic immediate.
  switch (P) {
  default:
    return None;
  case CmpInst::ICMP_SLT:
  case CmpInst::ICMP_SGE:
    // Check for
    //
    // x slt c => x sle c - 1
    // x sge c => x sgt c - 1
    //
    // When c is not the smallest possible negative number.
    if ((Size == 64 && static_cast<int64_t>(C) == INT64_MIN) ||
        (Size == 32 && static_cast<int32_t>(C) == INT32_MIN))
      return None;
    P = (P == CmpInst::ICMP_SLT) ? CmpInst::ICMP_SLE : CmpInst::ICMP_SGT;
    C -= 1;
    break;
  case CmpInst::ICMP_ULT:
  case CmpInst::ICMP_UGE:
    // Check for
    //
    // x ult c => x ule c - 1
    // x uge c => x ugt c - 1
    //
    // When c is not zero.
    if (C == 0)
      return None;
    P = (P == CmpInst::ICMP_ULT) ? CmpInst::ICMP_ULE : CmpInst::ICMP_UGT;
    C -= 1;
    break;
  case CmpInst::ICMP_SLE:
  case CmpInst::ICMP_SGT:
    // Check for
    //
    // x sle c => x slt c + 1
    // x sgt c => s sge c + 1
    //
    // When c is not the largest possible signed integer.
    if ((Size == 32 && static_cast<int32_t>(C) == INT32_MAX) ||
        (Size == 64 && static_cast<int64_t>(C) == INT64_MAX))
      return None;
    P = (P == CmpInst::ICMP_SLE) ? CmpInst::ICMP_SLT : CmpInst::ICMP_SGE;
    C += 1;
    break;
  case CmpInst::ICMP_ULE:
  case CmpInst::ICMP_UGT:
    // Check for
    //
    // x ule c => x ult c + 1
    // x ugt c => s uge c + 1
    //
    // When c is not the largest possible unsigned integer.
    if ((Size == 32 && static_cast<uint32_t>(C) == UINT32_MAX) ||
        (Size == 64 && C == UINT64_MAX))
      return None;
    P = (P == CmpInst::ICMP_ULE) ? CmpInst::ICMP_ULT : CmpInst::ICMP_UGE;
    C += 1;
    break;
  }

  // Check if the new constant is valid, and return the updated constant and
  // predicate if it is.
  if (Size == 32)
    C = static_cast<uint32_t>(C);
  if (!isLegalArithImmed(C))
    return None;
  return {{C, P}};
}

/// Determine whether or not it is possible to update the RHS and predicate of
/// a G_ICMP instruction such that the RHS will be selected as an arithmetic
/// immediate.
///
/// \p MI - The G_ICMP instruction
/// \p MatchInfo - The new RHS immediate and predicate on success
///
/// See tryAdjustICmpImmAndPred for valid transformations.
bool matchAdjustICmpImmAndPred(
    MachineInstr &MI, const MachineRegisterInfo &MRI,
    std::pair<uint64_t, CmpInst::Predicate> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_ICMP);
  Register RHS = MI.getOperand(3).getReg();
  auto Pred = static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
  if (auto MaybeNewImmAndPred = tryAdjustICmpImmAndPred(RHS, Pred, MRI)) {
    MatchInfo = *MaybeNewImmAndPred;
    return true;
  }
  return false;
}

bool applyAdjustICmpImmAndPred(
    MachineInstr &MI, std::pair<uint64_t, CmpInst::Predicate> &MatchInfo,
    MachineIRBuilder &MIB, GISelChangeObserver &Observer) {
  MIB.setInstrAndDebugLoc(MI);
  MachineOperand &RHS = MI.getOperand(3);
  MachineRegisterInfo &MRI = *MIB.getMRI();
  auto Cst = MIB.buildConstant(MRI.cloneVirtualRegister(RHS.getReg()),
                               MatchInfo.first);
  Observer.changingInstr(MI);
  RHS.setReg(Cst->getOperand(0).getReg());
  MI.getOperand(1).setPredicate(MatchInfo.second);
  Observer.changedInstr(MI);
  return true;
}

bool matchDupLane(MachineInstr &MI, MachineRegisterInfo &MRI,
                  std::pair<unsigned, int> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  Register Src1Reg = MI.getOperand(1).getReg();
  const LLT SrcTy = MRI.getType(Src1Reg);
  const LLT DstTy = MRI.getType(MI.getOperand(0).getReg());

  auto LaneIdx = getSplatIndex(MI);
  if (!LaneIdx)
    return false;

  // The lane idx should be within the first source vector.
  if (*LaneIdx >= SrcTy.getNumElements())
    return false;

  if (DstTy != SrcTy)
    return false;

  LLT ScalarTy = SrcTy.getElementType();
  unsigned ScalarSize = ScalarTy.getSizeInBits();

  unsigned Opc = 0;
  switch (SrcTy.getNumElements()) {
  case 2:
    if (ScalarSize == 64)
      Opc = AArch64::G_DUPLANE64;
    else if (ScalarSize == 32)
      Opc = AArch64::G_DUPLANE32;
    break;
  case 4:
    if (ScalarSize == 32)
      Opc = AArch64::G_DUPLANE32;
    break;
  case 8:
    if (ScalarSize == 16)
      Opc = AArch64::G_DUPLANE16;
    break;
  case 16:
    if (ScalarSize == 8)
      Opc = AArch64::G_DUPLANE8;
    break;
  default:
    break;
  }
  if (!Opc)
    return false;

  MatchInfo.first = Opc;
  MatchInfo.second = *LaneIdx;
  return true;
}

bool applyDupLane(MachineInstr &MI, MachineRegisterInfo &MRI,
                  MachineIRBuilder &B, std::pair<unsigned, int> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  Register Src1Reg = MI.getOperand(1).getReg();
  const LLT SrcTy = MRI.getType(Src1Reg);

  B.setInstrAndDebugLoc(MI);
  auto Lane = B.buildConstant(LLT::scalar(64), MatchInfo.second);

  Register DupSrc = MI.getOperand(1).getReg();
  // For types like <2 x s32>, we can use G_DUPLANE32, with a <4 x s32> source.
  // To do this, we can use a G_CONCAT_VECTORS to do the widening.
  if (SrcTy == LLT::fixed_vector(2, LLT::scalar(32))) {
    assert(MRI.getType(MI.getOperand(0).getReg()).getNumElements() == 2 &&
           "Unexpected dest elements");
    auto Undef = B.buildUndef(SrcTy);
    DupSrc = B.buildConcatVectors(
                  SrcTy.changeElementCount(ElementCount::getFixed(4)),
                  {Src1Reg, Undef.getReg(0)})
                 .getReg(0);
  }
  B.buildInstr(MatchInfo.first, {MI.getOperand(0).getReg()}, {DupSrc, Lane});
  MI.eraseFromParent();
  return true;
}

static bool matchBuildVectorToDup(MachineInstr &MI, MachineRegisterInfo &MRI) {
  assert(MI.getOpcode() == TargetOpcode::G_BUILD_VECTOR);
  auto Splat = getAArch64VectorSplat(MI, MRI);
  if (!Splat)
    return false;
  if (Splat->isReg())
    return true;
  // Later, during selection, we'll try to match imported patterns using
  // immAllOnesV and immAllZerosV. These require G_BUILD_VECTOR. Don't lower
  // G_BUILD_VECTORs which could match those patterns.
  int64_t Cst = Splat->getCst();
  return (Cst != 0 && Cst != -1);
}

static bool applyBuildVectorToDup(MachineInstr &MI, MachineRegisterInfo &MRI,
                                  MachineIRBuilder &B) {
  B.setInstrAndDebugLoc(MI);
  B.buildInstr(AArch64::G_DUP, {MI.getOperand(0).getReg()},
               {MI.getOperand(1).getReg()});
  MI.eraseFromParent();
  return true;
}

/// \returns how many instructions would be saved by folding a G_ICMP's shift
/// and/or extension operations.
static unsigned getCmpOperandFoldingProfit(Register CmpOp,
                                           const MachineRegisterInfo &MRI) {
  // No instructions to save if there's more than one use or no uses.
  if (!MRI.hasOneNonDBGUse(CmpOp))
    return 0;

  // FIXME: This is duplicated with the selector. (See: selectShiftedRegister)
  auto IsSupportedExtend = [&](const MachineInstr &MI) {
    if (MI.getOpcode() == TargetOpcode::G_SEXT_INREG)
      return true;
    if (MI.getOpcode() != TargetOpcode::G_AND)
      return false;
    auto ValAndVReg =
        getIConstantVRegValWithLookThrough(MI.getOperand(2).getReg(), MRI);
    if (!ValAndVReg)
      return false;
    uint64_t Mask = ValAndVReg->Value.getZExtValue();
    return (Mask == 0xFF || Mask == 0xFFFF || Mask == 0xFFFFFFFF);
  };

  MachineInstr *Def = getDefIgnoringCopies(CmpOp, MRI);
  if (IsSupportedExtend(*Def))
    return 1;

  unsigned Opc = Def->getOpcode();
  if (Opc != TargetOpcode::G_SHL && Opc != TargetOpcode::G_ASHR &&
      Opc != TargetOpcode::G_LSHR)
    return 0;

  auto MaybeShiftAmt =
      getIConstantVRegValWithLookThrough(Def->getOperand(2).getReg(), MRI);
  if (!MaybeShiftAmt)
    return 0;
  uint64_t ShiftAmt = MaybeShiftAmt->Value.getZExtValue();
  MachineInstr *ShiftLHS =
      getDefIgnoringCopies(Def->getOperand(1).getReg(), MRI);

  // Check if we can fold an extend and a shift.
  // FIXME: This is duplicated with the selector. (See:
  // selectArithExtendedRegister)
  if (IsSupportedExtend(*ShiftLHS))
    return (ShiftAmt <= 4) ? 2 : 1;

  LLT Ty = MRI.getType(Def->getOperand(0).getReg());
  if (Ty.isVector())
    return 0;
  unsigned ShiftSize = Ty.getSizeInBits();
  if ((ShiftSize == 32 && ShiftAmt <= 31) ||
      (ShiftSize == 64 && ShiftAmt <= 63))
    return 1;
  return 0;
}

/// \returns true if it would be profitable to swap the LHS and RHS of a G_ICMP
/// instruction \p MI.
static bool trySwapICmpOperands(MachineInstr &MI,
                                 const MachineRegisterInfo &MRI) {
  assert(MI.getOpcode() == TargetOpcode::G_ICMP);
  // Swap the operands if it would introduce a profitable folding opportunity.
  // (e.g. a shift + extend).
  //
  //  For example:
  //    lsl     w13, w11, #1
  //    cmp     w13, w12
  // can be turned into:
  //    cmp     w12, w11, lsl #1

  // Don't swap if there's a constant on the RHS, because we know we can fold
  // that.
  Register RHS = MI.getOperand(3).getReg();
  auto RHSCst = getIConstantVRegValWithLookThrough(RHS, MRI);
  if (RHSCst && isLegalArithImmed(RHSCst->Value.getSExtValue()))
    return false;

  Register LHS = MI.getOperand(2).getReg();
  auto Pred = static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
  auto GetRegForProfit = [&](Register Reg) {
    MachineInstr *Def = getDefIgnoringCopies(Reg, MRI);
    return isCMN(Def, Pred, MRI) ? Def->getOperand(2).getReg() : Reg;
  };

  // Don't have a constant on the RHS. If we swap the LHS and RHS of the
  // compare, would we be able to fold more instructions?
  Register TheLHS = GetRegForProfit(LHS);
  Register TheRHS = GetRegForProfit(RHS);

  // If the LHS is more likely to give us a folding opportunity, then swap the
  // LHS and RHS.
  return (getCmpOperandFoldingProfit(TheLHS, MRI) >
          getCmpOperandFoldingProfit(TheRHS, MRI));
}

static bool applySwapICmpOperands(MachineInstr &MI,
                                   GISelChangeObserver &Observer) {
  auto Pred = static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
  Register LHS = MI.getOperand(2).getReg();
  Register RHS = MI.getOperand(3).getReg();
  Observer.changedInstr(MI);
  MI.getOperand(1).setPredicate(CmpInst::getSwappedPredicate(Pred));
  MI.getOperand(2).setReg(RHS);
  MI.getOperand(3).setReg(LHS);
  Observer.changedInstr(MI);
  return true;
}

/// \returns a function which builds a vector floating point compare instruction
/// for a condition code \p CC.
/// \param [in] IsZero - True if the comparison is against 0.
/// \param [in] NoNans - True if the target has NoNansFPMath.
static std::function<Register(MachineIRBuilder &)>
getVectorFCMP(AArch64CC::CondCode CC, Register LHS, Register RHS, bool IsZero,
              bool NoNans, MachineRegisterInfo &MRI) {
  LLT DstTy = MRI.getType(LHS);
  assert(DstTy.isVector() && "Expected vector types only?");
  assert(DstTy == MRI.getType(RHS) && "Src and Dst types must match!");
  switch (CC) {
  default:
    llvm_unreachable("Unexpected condition code!");
  case AArch64CC::NE:
    return [LHS, RHS, IsZero, DstTy](MachineIRBuilder &MIB) {
      auto FCmp = IsZero
                      ? MIB.buildInstr(AArch64::G_FCMEQZ, {DstTy}, {LHS})
                      : MIB.buildInstr(AArch64::G_FCMEQ, {DstTy}, {LHS, RHS});
      return MIB.buildNot(DstTy, FCmp).getReg(0);
    };
  case AArch64CC::EQ:
    return [LHS, RHS, IsZero, DstTy](MachineIRBuilder &MIB) {
      return IsZero
                 ? MIB.buildInstr(AArch64::G_FCMEQZ, {DstTy}, {LHS}).getReg(0)
                 : MIB.buildInstr(AArch64::G_FCMEQ, {DstTy}, {LHS, RHS})
                       .getReg(0);
    };
  case AArch64CC::GE:
    return [LHS, RHS, IsZero, DstTy](MachineIRBuilder &MIB) {
      return IsZero
                 ? MIB.buildInstr(AArch64::G_FCMGEZ, {DstTy}, {LHS}).getReg(0)
                 : MIB.buildInstr(AArch64::G_FCMGE, {DstTy}, {LHS, RHS})
                       .getReg(0);
    };
  case AArch64CC::GT:
    return [LHS, RHS, IsZero, DstTy](MachineIRBuilder &MIB) {
      return IsZero
                 ? MIB.buildInstr(AArch64::G_FCMGTZ, {DstTy}, {LHS}).getReg(0)
                 : MIB.buildInstr(AArch64::G_FCMGT, {DstTy}, {LHS, RHS})
                       .getReg(0);
    };
  case AArch64CC::LS:
    return [LHS, RHS, IsZero, DstTy](MachineIRBuilder &MIB) {
      return IsZero
                 ? MIB.buildInstr(AArch64::G_FCMLEZ, {DstTy}, {LHS}).getReg(0)
                 : MIB.buildInstr(AArch64::G_FCMGE, {DstTy}, {RHS, LHS})
                       .getReg(0);
    };
  case AArch64CC::MI:
    return [LHS, RHS, IsZero, DstTy](MachineIRBuilder &MIB) {
      return IsZero
                 ? MIB.buildInstr(AArch64::G_FCMLTZ, {DstTy}, {LHS}).getReg(0)
                 : MIB.buildInstr(AArch64::G_FCMGT, {DstTy}, {RHS, LHS})
                       .getReg(0);
    };
  }
}

/// Try to lower a vector G_FCMP \p MI into an AArch64-specific pseudo.
static bool lowerVectorFCMP(MachineInstr &MI, MachineRegisterInfo &MRI,
                            MachineIRBuilder &MIB) {
  assert(MI.getOpcode() == TargetOpcode::G_FCMP);
  const auto &ST = MI.getMF()->getSubtarget<AArch64Subtarget>();
  Register Dst = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(Dst);
  if (!DstTy.isVector() || !ST.hasNEON())
    return false;
  const auto Pred =
      static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
  Register LHS = MI.getOperand(2).getReg();
  // TODO: Handle v4s16 case.
  unsigned EltSize = MRI.getType(LHS).getScalarSizeInBits();
  if (EltSize != 32 && EltSize != 64)
    return false;
  Register RHS = MI.getOperand(3).getReg();
  auto Splat = getAArch64VectorSplat(*MRI.getVRegDef(RHS), MRI);

  // Compares against 0 have special target-specific pseudos.
  bool IsZero = Splat && Splat->isCst() && Splat->getCst() == 0;
  bool Invert;
  AArch64CC::CondCode CC, CC2;
  changeVectorFCMPPredToAArch64CC(Pred, CC, CC2, Invert);
  bool NoNans = ST.getTargetLowering()->getTargetMachine().Options.NoNaNsFPMath;

  // Instead of having an apply function, just build here to simplify things.
  MIB.setInstrAndDebugLoc(MI);
  auto Cmp = getVectorFCMP(CC, LHS, RHS, IsZero, NoNans, MRI);
  Register CmpRes;
  if (CC2 == AArch64CC::AL)
    CmpRes = Cmp(MIB);
  else {
    auto Cmp2 = getVectorFCMP(CC2, LHS, RHS, IsZero, NoNans, MRI);
    auto Cmp2Dst = Cmp2(MIB);
    auto Cmp1Dst = Cmp(MIB);
    CmpRes = MIB.buildOr(DstTy, Cmp1Dst, Cmp2Dst).getReg(0);
  }
  if (Invert)
    CmpRes = MIB.buildNot(DstTy, CmpRes).getReg(0);
  MRI.replaceRegWith(Dst, CmpRes);
  MI.eraseFromParent();
  return false;
}

static bool matchFormTruncstore(MachineInstr &MI, MachineRegisterInfo &MRI,
                                Register &SrcReg) {
  assert(MI.getOpcode() == TargetOpcode::G_STORE);
  Register DstReg = MI.getOperand(0).getReg();
  if (MRI.getType(DstReg).isVector())
    return false;
  // Match a store of a truncate.
  if (!mi_match(DstReg, MRI, m_GTrunc(m_Reg(SrcReg))))
    return false;
  // Only form truncstores for value types of max 64b.
  return MRI.getType(SrcReg).getSizeInBits() <= 64;
}

static bool applyFormTruncstore(MachineInstr &MI, MachineRegisterInfo &MRI,
                                MachineIRBuilder &B,
                                GISelChangeObserver &Observer,
                                Register &SrcReg) {
  assert(MI.getOpcode() == TargetOpcode::G_STORE);
  Observer.changingInstr(MI);
  MI.getOperand(0).setReg(SrcReg);
  Observer.changedInstr(MI);
  return true;
}

#define AARCH64POSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_DEPS
#include "AArch64GenPostLegalizeGILowering.inc"
#undef AARCH64POSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define AARCH64POSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_H
#include "AArch64GenPostLegalizeGILowering.inc"
#undef AARCH64POSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_H

class AArch64PostLegalizerLoweringInfo : public CombinerInfo {
public:
  AArch64GenPostLegalizerLoweringHelperRuleConfig GeneratedRuleCfg;

  AArch64PostLegalizerLoweringInfo(bool OptSize, bool MinSize)
      : CombinerInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, /*OptEnabled = */ true, OptSize,
                     MinSize) {
    if (!GeneratedRuleCfg.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  virtual bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
                       MachineIRBuilder &B) const override;
};

bool AArch64PostLegalizerLoweringInfo::combine(GISelChangeObserver &Observer,
                                               MachineInstr &MI,
                                               MachineIRBuilder &B) const {
  CombinerHelper Helper(Observer, B);
  AArch64GenPostLegalizerLoweringHelper Generated(GeneratedRuleCfg);
  return Generated.tryCombineAll(Observer, MI, B, Helper);
}

#define AARCH64POSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_CPP
#include "AArch64GenPostLegalizeGILowering.inc"
#undef AARCH64POSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_CPP

class AArch64PostLegalizerLowering : public MachineFunctionPass {
public:
  static char ID;

  AArch64PostLegalizerLowering();

  StringRef getPassName() const override {
    return "AArch64PostLegalizerLowering";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

void AArch64PostLegalizerLowering::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

AArch64PostLegalizerLowering::AArch64PostLegalizerLowering()
    : MachineFunctionPass(ID) {
  initializeAArch64PostLegalizerLoweringPass(*PassRegistry::getPassRegistry());
}

bool AArch64PostLegalizerLowering::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  assert(MF.getProperties().hasProperty(
             MachineFunctionProperties::Property::Legalized) &&
         "Expected a legalized function?");
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  AArch64PostLegalizerLoweringInfo PCInfo(F.hasOptSize(), F.hasMinSize());
  Combiner C(PCInfo, TPC);
  return C.combineMachineInstrs(MF, /*CSEInfo*/ nullptr);
}

char AArch64PostLegalizerLowering::ID = 0;
INITIALIZE_PASS_BEGIN(AArch64PostLegalizerLowering, DEBUG_TYPE,
                      "Lower AArch64 MachineInstrs after legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AArch64PostLegalizerLowering, DEBUG_TYPE,
                    "Lower AArch64 MachineInstrs after legalization", false,
                    false)

namespace llvm {
FunctionPass *createAArch64PostLegalizerLowering() {
  return new AArch64PostLegalizerLowering();
}
} // end namespace llvm
