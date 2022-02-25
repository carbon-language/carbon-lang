//===-- llvm/CodeGen/GlobalISel/CombinerHelper.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
/// \file
/// This contains common combine transformations that may be used in a combine
/// pass,or by the target elsewhere.
/// Targets can pick individual opcode transformations from the helper or use
/// tryCombine which invokes all transformations. All of the transformations
/// return true if the MachineInstruction changed and false otherwise.
///
//===--------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_COMBINERHELPER_H
#define LLVM_CODEGEN_GLOBALISEL_COMBINERHELPER_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/Support/Alignment.h"

namespace llvm {

class GISelChangeObserver;
class MachineIRBuilder;
class MachineInstrBuilder;
class MachineRegisterInfo;
class MachineInstr;
class MachineOperand;
class GISelKnownBits;
class MachineDominatorTree;
class LegalizerInfo;
struct LegalityQuery;
class RegisterBank;
class RegisterBankInfo;
class TargetLowering;
class TargetRegisterInfo;

struct PreferredTuple {
  LLT Ty;                // The result type of the extend.
  unsigned ExtendOpcode; // G_ANYEXT/G_SEXT/G_ZEXT
  MachineInstr *MI;
};

struct IndexedLoadStoreMatchInfo {
  Register Addr;
  Register Base;
  Register Offset;
  bool IsPre;
};

struct PtrAddChain {
  int64_t Imm;
  Register Base;
  const RegisterBank *Bank;
};

struct RegisterImmPair {
  Register Reg;
  int64_t Imm;
};

struct ShiftOfShiftedLogic {
  MachineInstr *Logic;
  MachineInstr *Shift2;
  Register LogicNonShiftReg;
  uint64_t ValSum;
};

using OperandBuildSteps =
    SmallVector<std::function<void(MachineInstrBuilder &)>, 4>;
struct InstructionBuildSteps {
  unsigned Opcode = 0;          /// The opcode for the produced instruction.
  OperandBuildSteps OperandFns; /// Operands to be added to the instruction.
  InstructionBuildSteps() = default;
  InstructionBuildSteps(unsigned Opcode, const OperandBuildSteps &OperandFns)
      : Opcode(Opcode), OperandFns(OperandFns) {}
};

struct InstructionStepsMatchInfo {
  /// Describes instructions to be built during a combine.
  SmallVector<InstructionBuildSteps, 2> InstrsToBuild;
  InstructionStepsMatchInfo() = default;
  InstructionStepsMatchInfo(
      std::initializer_list<InstructionBuildSteps> InstrsToBuild)
      : InstrsToBuild(InstrsToBuild) {}
};

class CombinerHelper {
protected:
  MachineIRBuilder &Builder;
  MachineRegisterInfo &MRI;
  GISelChangeObserver &Observer;
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;
  const LegalizerInfo *LI;
  const RegisterBankInfo *RBI;
  const TargetRegisterInfo *TRI;

public:
  CombinerHelper(GISelChangeObserver &Observer, MachineIRBuilder &B,
                 GISelKnownBits *KB = nullptr,
                 MachineDominatorTree *MDT = nullptr,
                 const LegalizerInfo *LI = nullptr);

  GISelKnownBits *getKnownBits() const {
    return KB;
  }

  const TargetLowering &getTargetLowering() const;

  /// \return true if the combine is running prior to legalization, or if \p
  /// Query is legal on the target.
  bool isLegalOrBeforeLegalizer(const LegalityQuery &Query) const;

  /// MachineRegisterInfo::replaceRegWith() and inform the observer of the changes
  void replaceRegWith(MachineRegisterInfo &MRI, Register FromReg, Register ToReg) const;

  /// Replace a single register operand with a new register and inform the
  /// observer of the changes.
  void replaceRegOpWith(MachineRegisterInfo &MRI, MachineOperand &FromRegOp,
                        Register ToReg) const;

  /// Get the register bank of \p Reg.
  /// If Reg has not been assigned a register, a register class,
  /// or a register bank, then this returns nullptr.
  ///
  /// \pre Reg.isValid()
  const RegisterBank *getRegBank(Register Reg) const;

  /// Set the register bank of \p Reg.
  /// Does nothing if the RegBank is null.
  /// This is the counterpart to getRegBank.
  void setRegBank(Register Reg, const RegisterBank *RegBank);

  /// If \p MI is COPY, try to combine it.
  /// Returns true if MI changed.
  bool tryCombineCopy(MachineInstr &MI);
  bool matchCombineCopy(MachineInstr &MI);
  void applyCombineCopy(MachineInstr &MI);

  /// Returns true if \p DefMI precedes \p UseMI or they are the same
  /// instruction. Both must be in the same basic block.
  bool isPredecessor(const MachineInstr &DefMI, const MachineInstr &UseMI);

  /// Returns true if \p DefMI dominates \p UseMI. By definition an
  /// instruction dominates itself.
  ///
  /// If we haven't been provided with a MachineDominatorTree during
  /// construction, this function returns a conservative result that tracks just
  /// a single basic block.
  bool dominates(const MachineInstr &DefMI, const MachineInstr &UseMI);

  /// If \p MI is extend that consumes the result of a load, try to combine it.
  /// Returns true if MI changed.
  bool tryCombineExtendingLoads(MachineInstr &MI);
  bool matchCombineExtendingLoads(MachineInstr &MI, PreferredTuple &MatchInfo);
  void applyCombineExtendingLoads(MachineInstr &MI, PreferredTuple &MatchInfo);

  /// Combine \p MI into a pre-indexed or post-indexed load/store operation if
  /// legal and the surrounding code makes it useful.
  bool tryCombineIndexedLoadStore(MachineInstr &MI);
  bool matchCombineIndexedLoadStore(MachineInstr &MI, IndexedLoadStoreMatchInfo &MatchInfo);
  void applyCombineIndexedLoadStore(MachineInstr &MI, IndexedLoadStoreMatchInfo &MatchInfo);

  bool matchSextTruncSextLoad(MachineInstr &MI);
  void applySextTruncSextLoad(MachineInstr &MI);

  /// Match sext_inreg(load p), imm -> sextload p
  bool matchSextInRegOfLoad(MachineInstr &MI, std::tuple<Register, unsigned> &MatchInfo);
  void applySextInRegOfLoad(MachineInstr &MI, std::tuple<Register, unsigned> &MatchInfo);

  /// Try to combine G_[SU]DIV and G_[SU]REM into a single G_[SU]DIVREM
  /// when their source operands are identical.
  bool matchCombineDivRem(MachineInstr &MI, MachineInstr *&OtherMI);
  void applyCombineDivRem(MachineInstr &MI, MachineInstr *&OtherMI);

  /// If a brcond's true block is not the fallthrough, make it so by inverting
  /// the condition and swapping operands.
  bool matchOptBrCondByInvertingCond(MachineInstr &MI, MachineInstr *&BrCond);
  void applyOptBrCondByInvertingCond(MachineInstr &MI, MachineInstr *&BrCond);

  /// If \p MI is G_CONCAT_VECTORS, try to combine it.
  /// Returns true if MI changed.
  /// Right now, we support:
  /// - concat_vector(undef, undef) => undef
  /// - concat_vector(build_vector(A, B), build_vector(C, D)) =>
  ///   build_vector(A, B, C, D)
  ///
  /// \pre MI.getOpcode() == G_CONCAT_VECTORS.
  bool tryCombineConcatVectors(MachineInstr &MI);
  /// Check if the G_CONCAT_VECTORS \p MI is undef or if it
  /// can be flattened into a build_vector.
  /// In the first case \p IsUndef will be true.
  /// In the second case \p Ops will contain the operands needed
  /// to produce the flattened build_vector.
  ///
  /// \pre MI.getOpcode() == G_CONCAT_VECTORS.
  bool matchCombineConcatVectors(MachineInstr &MI, bool &IsUndef,
                                 SmallVectorImpl<Register> &Ops);
  /// Replace \p MI with a flattened build_vector with \p Ops or an
  /// implicit_def if IsUndef is true.
  void applyCombineConcatVectors(MachineInstr &MI, bool IsUndef,
                                 const ArrayRef<Register> Ops);

  /// Try to combine G_SHUFFLE_VECTOR into G_CONCAT_VECTORS.
  /// Returns true if MI changed.
  ///
  /// \pre MI.getOpcode() == G_SHUFFLE_VECTOR.
  bool tryCombineShuffleVector(MachineInstr &MI);
  /// Check if the G_SHUFFLE_VECTOR \p MI can be replaced by a
  /// concat_vectors.
  /// \p Ops will contain the operands needed to produce the flattened
  /// concat_vectors.
  ///
  /// \pre MI.getOpcode() == G_SHUFFLE_VECTOR.
  bool matchCombineShuffleVector(MachineInstr &MI,
                                 SmallVectorImpl<Register> &Ops);
  /// Replace \p MI with a concat_vectors with \p Ops.
  void applyCombineShuffleVector(MachineInstr &MI,
                                 const ArrayRef<Register> Ops);

  /// Optimize memcpy intrinsics et al, e.g. constant len calls.
  /// /p MaxLen if non-zero specifies the max length of a mem libcall to inline.
  ///
  /// For example (pre-indexed):
  ///
  ///     $addr = G_PTR_ADD $base, $offset
  ///     [...]
  ///     $val = G_LOAD $addr
  ///     [...]
  ///     $whatever = COPY $addr
  ///
  /// -->
  ///
  ///     $val, $addr = G_INDEXED_LOAD $base, $offset, 1 (IsPre)
  ///     [...]
  ///     $whatever = COPY $addr
  ///
  /// or (post-indexed):
  ///
  ///     G_STORE $val, $base
  ///     [...]
  ///     $addr = G_PTR_ADD $base, $offset
  ///     [...]
  ///     $whatever = COPY $addr
  ///
  /// -->
  ///
  ///     $addr = G_INDEXED_STORE $val, $base, $offset
  ///     [...]
  ///     $whatever = COPY $addr
  bool tryCombineMemCpyFamily(MachineInstr &MI, unsigned MaxLen = 0);

  bool matchPtrAddImmedChain(MachineInstr &MI, PtrAddChain &MatchInfo);
  void applyPtrAddImmedChain(MachineInstr &MI, PtrAddChain &MatchInfo);

  /// Fold (shift (shift base, x), y) -> (shift base (x+y))
  bool matchShiftImmedChain(MachineInstr &MI, RegisterImmPair &MatchInfo);
  void applyShiftImmedChain(MachineInstr &MI, RegisterImmPair &MatchInfo);

  /// If we have a shift-by-constant of a bitwise logic op that itself has a
  /// shift-by-constant operand with identical opcode, we may be able to convert
  /// that into 2 independent shifts followed by the logic op.
  bool matchShiftOfShiftedLogic(MachineInstr &MI,
                                ShiftOfShiftedLogic &MatchInfo);
  void applyShiftOfShiftedLogic(MachineInstr &MI,
                                ShiftOfShiftedLogic &MatchInfo);

  /// Transform a multiply by a power-of-2 value to a left shift.
  bool matchCombineMulToShl(MachineInstr &MI, unsigned &ShiftVal);
  void applyCombineMulToShl(MachineInstr &MI, unsigned &ShiftVal);

  // Transform a G_SHL with an extended source into a narrower shift if
  // possible.
  bool matchCombineShlOfExtend(MachineInstr &MI, RegisterImmPair &MatchData);
  void applyCombineShlOfExtend(MachineInstr &MI,
                               const RegisterImmPair &MatchData);

  /// Fold away a merge of an unmerge of the corresponding values.
  bool matchCombineMergeUnmerge(MachineInstr &MI, Register &MatchInfo);

  /// Reduce a shift by a constant to an unmerge and a shift on a half sized
  /// type. This will not produce a shift smaller than \p TargetShiftSize.
  bool matchCombineShiftToUnmerge(MachineInstr &MI, unsigned TargetShiftSize,
                                 unsigned &ShiftVal);
  void applyCombineShiftToUnmerge(MachineInstr &MI, const unsigned &ShiftVal);
  bool tryCombineShiftToUnmerge(MachineInstr &MI, unsigned TargetShiftAmount);

  /// Transform <ty,...> G_UNMERGE(G_MERGE ty X, Y, Z) -> ty X, Y, Z.
  bool
  matchCombineUnmergeMergeToPlainValues(MachineInstr &MI,
                                        SmallVectorImpl<Register> &Operands);
  void
  applyCombineUnmergeMergeToPlainValues(MachineInstr &MI,
                                        SmallVectorImpl<Register> &Operands);

  /// Transform G_UNMERGE Constant -> Constant1, Constant2, ...
  bool matchCombineUnmergeConstant(MachineInstr &MI,
                                   SmallVectorImpl<APInt> &Csts);
  void applyCombineUnmergeConstant(MachineInstr &MI,
                                   SmallVectorImpl<APInt> &Csts);

  /// Transform X, Y<dead> = G_UNMERGE Z -> X = G_TRUNC Z.
  bool matchCombineUnmergeWithDeadLanesToTrunc(MachineInstr &MI);
  void applyCombineUnmergeWithDeadLanesToTrunc(MachineInstr &MI);

  /// Transform X, Y = G_UNMERGE(G_ZEXT(Z)) -> X = G_ZEXT(Z); Y = G_CONSTANT 0
  bool matchCombineUnmergeZExtToZExt(MachineInstr &MI);
  void applyCombineUnmergeZExtToZExt(MachineInstr &MI);

  /// Transform fp_instr(cst) to constant result of the fp operation.
  bool matchCombineConstantFoldFpUnary(MachineInstr &MI,
                                       Optional<APFloat> &Cst);
  void applyCombineConstantFoldFpUnary(MachineInstr &MI,
                                       Optional<APFloat> &Cst);

  /// Transform IntToPtr(PtrToInt(x)) to x if cast is in the same address space.
  bool matchCombineI2PToP2I(MachineInstr &MI, Register &Reg);
  void applyCombineI2PToP2I(MachineInstr &MI, Register &Reg);

  /// Transform PtrToInt(IntToPtr(x)) to x.
  bool matchCombineP2IToI2P(MachineInstr &MI, Register &Reg);
  void applyCombineP2IToI2P(MachineInstr &MI, Register &Reg);

  /// Transform G_ADD (G_PTRTOINT x), y -> G_PTRTOINT (G_PTR_ADD x, y)
  /// Transform G_ADD y, (G_PTRTOINT x) -> G_PTRTOINT (G_PTR_ADD x, y)
  bool matchCombineAddP2IToPtrAdd(MachineInstr &MI,
                                  std::pair<Register, bool> &PtrRegAndCommute);
  void applyCombineAddP2IToPtrAdd(MachineInstr &MI,
                                  std::pair<Register, bool> &PtrRegAndCommute);

  // Transform G_PTR_ADD (G_PTRTOINT C1), C2 -> C1 + C2
  bool matchCombineConstPtrAddToI2P(MachineInstr &MI, int64_t &NewCst);
  void applyCombineConstPtrAddToI2P(MachineInstr &MI, int64_t &NewCst);

  /// Transform anyext(trunc(x)) to x.
  bool matchCombineAnyExtTrunc(MachineInstr &MI, Register &Reg);
  void applyCombineAnyExtTrunc(MachineInstr &MI, Register &Reg);

  /// Transform zext(trunc(x)) to x.
  bool matchCombineZextTrunc(MachineInstr &MI, Register &Reg);

  /// Transform [asz]ext([asz]ext(x)) to [asz]ext x.
  bool matchCombineExtOfExt(MachineInstr &MI,
                            std::tuple<Register, unsigned> &MatchInfo);
  void applyCombineExtOfExt(MachineInstr &MI,
                            std::tuple<Register, unsigned> &MatchInfo);

  /// Transform fneg(fneg(x)) to x.
  bool matchCombineFNegOfFNeg(MachineInstr &MI, Register &Reg);

  /// Match fabs(fabs(x)) to fabs(x).
  bool matchCombineFAbsOfFAbs(MachineInstr &MI, Register &Src);
  void applyCombineFAbsOfFAbs(MachineInstr &MI, Register &Src);

  /// Transform trunc ([asz]ext x) to x or ([asz]ext x) or (trunc x).
  bool matchCombineTruncOfExt(MachineInstr &MI,
                              std::pair<Register, unsigned> &MatchInfo);
  void applyCombineTruncOfExt(MachineInstr &MI,
                              std::pair<Register, unsigned> &MatchInfo);

  /// Transform trunc (shl x, K) to shl (trunc x),
  /// K => K < VT.getScalarSizeInBits().
  bool matchCombineTruncOfShl(MachineInstr &MI,
                              std::pair<Register, Register> &MatchInfo);
  void applyCombineTruncOfShl(MachineInstr &MI,
                              std::pair<Register, Register> &MatchInfo);

  /// Transform G_MUL(x, -1) to G_SUB(0, x)
  void applyCombineMulByNegativeOne(MachineInstr &MI);

  /// Return true if any explicit use operand on \p MI is defined by a
  /// G_IMPLICIT_DEF.
  bool matchAnyExplicitUseIsUndef(MachineInstr &MI);

  /// Return true if all register explicit use operands on \p MI are defined by
  /// a G_IMPLICIT_DEF.
  bool matchAllExplicitUsesAreUndef(MachineInstr &MI);

  /// Return true if a G_SHUFFLE_VECTOR instruction \p MI has an undef mask.
  bool matchUndefShuffleVectorMask(MachineInstr &MI);

  /// Return true if a G_STORE instruction \p MI is storing an undef value.
  bool matchUndefStore(MachineInstr &MI);

  /// Return true if a G_SELECT instruction \p MI has an undef comparison.
  bool matchUndefSelectCmp(MachineInstr &MI);

  /// Return true if a G_SELECT instruction \p MI has a constant comparison. If
  /// true, \p OpIdx will store the operand index of the known selected value.
  bool matchConstantSelectCmp(MachineInstr &MI, unsigned &OpIdx);

  /// Replace an instruction with a G_FCONSTANT with value \p C.
  bool replaceInstWithFConstant(MachineInstr &MI, double C);

  /// Replace an instruction with a G_CONSTANT with value \p C.
  bool replaceInstWithConstant(MachineInstr &MI, int64_t C);

  /// Replace an instruction with a G_CONSTANT with value \p C.
  bool replaceInstWithConstant(MachineInstr &MI, APInt C);

  /// Replace an instruction with a G_IMPLICIT_DEF.
  bool replaceInstWithUndef(MachineInstr &MI);

  /// Delete \p MI and replace all of its uses with its \p OpIdx-th operand.
  bool replaceSingleDefInstWithOperand(MachineInstr &MI, unsigned OpIdx);

  /// Delete \p MI and replace all of its uses with \p Replacement.
  bool replaceSingleDefInstWithReg(MachineInstr &MI, Register Replacement);

  /// Return true if \p MOP1 and \p MOP2 are register operands are defined by
  /// equivalent instructions.
  bool matchEqualDefs(const MachineOperand &MOP1, const MachineOperand &MOP2);

  /// Return true if \p MOP is defined by a G_CONSTANT with a value equal to
  /// \p C.
  bool matchConstantOp(const MachineOperand &MOP, int64_t C);

  /// Optimize (cond ? x : x) -> x
  bool matchSelectSameVal(MachineInstr &MI);

  /// Optimize (x op x) -> x
  bool matchBinOpSameVal(MachineInstr &MI);

  /// Check if operand \p OpIdx is zero.
  bool matchOperandIsZero(MachineInstr &MI, unsigned OpIdx);

  /// Check if operand \p OpIdx is undef.
  bool matchOperandIsUndef(MachineInstr &MI, unsigned OpIdx);

  /// Check if operand \p OpIdx is known to be a power of 2.
  bool matchOperandIsKnownToBeAPowerOfTwo(MachineInstr &MI, unsigned OpIdx);

  /// Erase \p MI
  bool eraseInst(MachineInstr &MI);

  /// Return true if MI is a G_ADD which can be simplified to a G_SUB.
  bool matchSimplifyAddToSub(MachineInstr &MI,
                             std::tuple<Register, Register> &MatchInfo);
  void applySimplifyAddToSub(MachineInstr &MI,
                             std::tuple<Register, Register> &MatchInfo);

  /// Match (logic_op (op x...), (op y...)) -> (op (logic_op x, y))
  bool
  matchHoistLogicOpWithSameOpcodeHands(MachineInstr &MI,
                                       InstructionStepsMatchInfo &MatchInfo);

  /// Replace \p MI with a series of instructions described in \p MatchInfo.
  void applyBuildInstructionSteps(MachineInstr &MI,
                                  InstructionStepsMatchInfo &MatchInfo);

  /// Match ashr (shl x, C), C -> sext_inreg (C)
  bool matchAshrShlToSextInreg(MachineInstr &MI,
                               std::tuple<Register, int64_t> &MatchInfo);
  void applyAshShlToSextInreg(MachineInstr &MI,
                              std::tuple<Register, int64_t> &MatchInfo);

  /// Fold and(and(x, C1), C2) -> C1&C2 ? and(x, C1&C2) : 0
  bool matchOverlappingAnd(MachineInstr &MI,
                           std::function<void(MachineIRBuilder &)> &MatchInfo);

  /// \return true if \p MI is a G_AND instruction whose operands are x and y
  /// where x & y == x or x & y == y. (E.g., one of operands is all-ones value.)
  ///
  /// \param [in] MI - The G_AND instruction.
  /// \param [out] Replacement - A register the G_AND should be replaced with on
  /// success.
  bool matchRedundantAnd(MachineInstr &MI, Register &Replacement);

  /// \return true if \p MI is a G_OR instruction whose operands are x and y
  /// where x | y == x or x | y == y. (E.g., one of operands is all-zeros
  /// value.)
  ///
  /// \param [in] MI - The G_OR instruction.
  /// \param [out] Replacement - A register the G_OR should be replaced with on
  /// success.
  bool matchRedundantOr(MachineInstr &MI, Register &Replacement);

  /// \return true if \p MI is a G_SEXT_INREG that can be erased.
  bool matchRedundantSExtInReg(MachineInstr &MI);

  /// Combine inverting a result of a compare into the opposite cond code.
  bool matchNotCmp(MachineInstr &MI, SmallVectorImpl<Register> &RegsToNegate);
  void applyNotCmp(MachineInstr &MI, SmallVectorImpl<Register> &RegsToNegate);

  /// Fold (xor (and x, y), y) -> (and (not x), y)
  ///{
  bool matchXorOfAndWithSameReg(MachineInstr &MI,
                                std::pair<Register, Register> &MatchInfo);
  void applyXorOfAndWithSameReg(MachineInstr &MI,
                                std::pair<Register, Register> &MatchInfo);
  ///}

  /// Combine G_PTR_ADD with nullptr to G_INTTOPTR
  bool matchPtrAddZero(MachineInstr &MI);
  void applyPtrAddZero(MachineInstr &MI);

  /// Combine G_UREM x, (known power of 2) to an add and bitmasking.
  void applySimplifyURemByPow2(MachineInstr &MI);

  bool matchCombineInsertVecElts(MachineInstr &MI,
                                 SmallVectorImpl<Register> &MatchInfo);

  void applyCombineInsertVecElts(MachineInstr &MI,
                             SmallVectorImpl<Register> &MatchInfo);

  /// Match expression trees of the form
  ///
  /// \code
  ///  sN *a = ...
  ///  sM val = a[0] | (a[1] << N) | (a[2] << 2N) | (a[3] << 3N) ...
  /// \endcode
  ///
  /// And check if the tree can be replaced with a M-bit load + possibly a
  /// bswap.
  bool matchLoadOrCombine(MachineInstr &MI,
                          std::function<void(MachineIRBuilder &)> &MatchInfo);

  bool matchExtendThroughPhis(MachineInstr &MI, MachineInstr *&ExtMI);
  void applyExtendThroughPhis(MachineInstr &MI, MachineInstr *&ExtMI);

  bool matchExtractVecEltBuildVec(MachineInstr &MI, Register &Reg);
  void applyExtractVecEltBuildVec(MachineInstr &MI, Register &Reg);

  bool matchExtractAllEltsFromBuildVector(
      MachineInstr &MI,
      SmallVectorImpl<std::pair<Register, MachineInstr *>> &MatchInfo);
  void applyExtractAllEltsFromBuildVector(
      MachineInstr &MI,
      SmallVectorImpl<std::pair<Register, MachineInstr *>> &MatchInfo);

  /// Use a function which takes in a MachineIRBuilder to perform a combine.
  /// By default, it erases the instruction \p MI from the function.
  void applyBuildFn(MachineInstr &MI,
                    std::function<void(MachineIRBuilder &)> &MatchInfo);
  /// Use a function which takes in a MachineIRBuilder to perform a combine.
  /// This variant does not erase \p MI after calling the build function.
  void applyBuildFnNoErase(MachineInstr &MI,
                           std::function<void(MachineIRBuilder &)> &MatchInfo);

  bool matchFunnelShiftToRotate(MachineInstr &MI);
  void applyFunnelShiftToRotate(MachineInstr &MI);
  bool matchRotateOutOfRange(MachineInstr &MI);
  void applyRotateOutOfRange(MachineInstr &MI);

  /// \returns true if a G_ICMP instruction \p MI can be replaced with a true
  /// or false constant based off of KnownBits information.
  bool matchICmpToTrueFalseKnownBits(MachineInstr &MI, int64_t &MatchInfo);

  bool matchBitfieldExtractFromSExtInReg(
      MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo);
  /// Match: and (lshr x, cst), mask -> ubfx x, cst, width
  bool matchBitfieldExtractFromAnd(
      MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo);

  /// Match: shr (shl x, n), k -> sbfx/ubfx x, pos, width
  bool matchBitfieldExtractFromShr(
      MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo);

  /// Reassociate pointer calculations with G_ADD involved, to allow better
  /// addressing mode usage.
  bool matchReassocPtrAdd(MachineInstr &MI,
                          std::function<void(MachineIRBuilder &)> &MatchInfo);


  /// Do constant folding when opportunities are exposed after MIR building.
  bool matchConstantFold(MachineInstr &MI, APInt &MatchInfo);

  /// \returns true if it is possible to narrow the width of a scalar binop
  /// feeding a G_AND instruction \p MI.
  bool matchNarrowBinopFeedingAnd(
      MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo);

  /// Try to transform \p MI by using all of the above
  /// combine functions. Returns true if changed.
  bool tryCombine(MachineInstr &MI);

  /// Emit loads and stores that perform the given memcpy.
  /// Assumes \p MI is a G_MEMCPY_INLINE
  /// TODO: implement dynamically sized inline memcpy,
  ///       and rename: s/bool tryEmit/void emit/
  bool tryEmitMemcpyInline(MachineInstr &MI);

private:
  // Memcpy family optimization helpers.
  bool tryEmitMemcpyInline(MachineInstr &MI, Register Dst, Register Src,
                           uint64_t KnownLen, Align DstAlign, Align SrcAlign,
                           bool IsVolatile);
  bool optimizeMemcpy(MachineInstr &MI, Register Dst, Register Src,
                      uint64_t KnownLen, uint64_t Limit, Align DstAlign,
                      Align SrcAlign, bool IsVolatile);
  bool optimizeMemmove(MachineInstr &MI, Register Dst, Register Src,
                       uint64_t KnownLen, Align DstAlign, Align SrcAlign,
                       bool IsVolatile);
  bool optimizeMemset(MachineInstr &MI, Register Dst, Register Val,
                      uint64_t KnownLen, Align DstAlign, bool IsVolatile);

  /// Given a non-indexed load or store instruction \p MI, find an offset that
  /// can be usefully and legally folded into it as a post-indexing operation.
  ///
  /// \returns true if a candidate is found.
  bool findPostIndexCandidate(MachineInstr &MI, Register &Addr, Register &Base,
                              Register &Offset);

  /// Given a non-indexed load or store instruction \p MI, find an offset that
  /// can be usefully and legally folded into it as a pre-indexing operation.
  ///
  /// \returns true if a candidate is found.
  bool findPreIndexCandidate(MachineInstr &MI, Register &Addr, Register &Base,
                             Register &Offset);

  /// Helper function for matchLoadOrCombine. Searches for Registers
  /// which may have been produced by a load instruction + some arithmetic.
  ///
  /// \param [in] Root - The search root.
  ///
  /// \returns The Registers found during the search.
  Optional<SmallVector<Register, 8>>
  findCandidatesForLoadOrCombine(const MachineInstr *Root) const;

  /// Helper function for matchLoadOrCombine.
  ///
  /// Checks if every register in \p RegsToVisit is defined by a load
  /// instruction + some arithmetic.
  ///
  /// \param [out] MemOffset2Idx - Maps the byte positions each load ends up
  /// at to the index of the load.
  /// \param [in] MemSizeInBits - The number of bits each load should produce.
  ///
  /// \returns On success, a 3-tuple containing lowest-index load found, the
  /// lowest index, and the last load in the sequence.
  Optional<std::tuple<GZExtLoad *, int64_t, GZExtLoad *>>
  findLoadOffsetsForLoadOrCombine(
      SmallDenseMap<int64_t, int64_t, 8> &MemOffset2Idx,
      const SmallVector<Register, 8> &RegsToVisit,
      const unsigned MemSizeInBits);

  /// Examines the G_PTR_ADD instruction \p PtrAdd and determines if performing
  /// a re-association of its operands would break an existing legal addressing
  /// mode that the address computation currently represents.
  bool reassociationCanBreakAddressingModePattern(MachineInstr &PtrAdd);
};
} // namespace llvm

#endif
