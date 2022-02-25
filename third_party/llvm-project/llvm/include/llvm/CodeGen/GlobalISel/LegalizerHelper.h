//== llvm/CodeGen/GlobalISel/LegalizerHelper.h ---------------- -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file A pass to convert the target-illegal operations created by IR -> MIR
/// translation into ones the target expects to be able to select. This may
/// occur in multiple phases, for example G_ADD <2 x i8> -> G_ADD <2 x i16> ->
/// G_ADD <4 x i16>.
///
/// The LegalizerHelper class is where most of the work happens, and is
/// designed to be callable from other passes that find themselves with an
/// illegal instruction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_LEGALIZERHELPER_H
#define LLVM_CODEGEN_GLOBALISEL_LEGALIZERHELPER_H

#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"

namespace llvm {
// Forward declarations.
class LegalizerInfo;
class Legalizer;
class MachineRegisterInfo;
class GISelChangeObserver;
class LostDebugLocObserver;
class TargetLowering;

class LegalizerHelper {
public:
  /// Expose MIRBuilder so clients can set their own RecordInsertInstruction
  /// functions
  MachineIRBuilder &MIRBuilder;

  /// To keep track of changes made by the LegalizerHelper.
  GISelChangeObserver &Observer;

private:
  MachineRegisterInfo &MRI;
  const LegalizerInfo &LI;
  const TargetLowering &TLI;

public:
  enum LegalizeResult {
    /// Instruction was already legal and no change was made to the
    /// MachineFunction.
    AlreadyLegal,

    /// Instruction has been legalized and the MachineFunction changed.
    Legalized,

    /// Some kind of error has occurred and we could not legalize this
    /// instruction.
    UnableToLegalize,
  };

  /// Expose LegalizerInfo so the clients can re-use.
  const LegalizerInfo &getLegalizerInfo() const { return LI; }
  const TargetLowering &getTargetLowering() const { return TLI; }

  LegalizerHelper(MachineFunction &MF, GISelChangeObserver &Observer,
                  MachineIRBuilder &B);
  LegalizerHelper(MachineFunction &MF, const LegalizerInfo &LI,
                  GISelChangeObserver &Observer, MachineIRBuilder &B);

  /// Replace \p MI by a sequence of legal instructions that can implement the
  /// same operation. Note that this means \p MI may be deleted, so any iterator
  /// steps should be performed before calling this function. \p Helper should
  /// be initialized to the MachineFunction containing \p MI.
  ///
  /// Considered as an opaque blob, the legal code will use and define the same
  /// registers as \p MI.
  LegalizeResult legalizeInstrStep(MachineInstr &MI,
                                   LostDebugLocObserver &LocObserver);

  /// Legalize an instruction by emiting a runtime library call instead.
  LegalizeResult libcall(MachineInstr &MI, LostDebugLocObserver &LocObserver);

  /// Legalize an instruction by reducing the width of the underlying scalar
  /// type.
  LegalizeResult narrowScalar(MachineInstr &MI, unsigned TypeIdx, LLT NarrowTy);

  /// Legalize an instruction by performing the operation on a wider scalar type
  /// (for example a 16-bit addition can be safely performed at 32-bits
  /// precision, ignoring the unused bits).
  LegalizeResult widenScalar(MachineInstr &MI, unsigned TypeIdx, LLT WideTy);

  /// Legalize an instruction by replacing the value type
  LegalizeResult bitcast(MachineInstr &MI, unsigned TypeIdx, LLT Ty);

  /// Legalize an instruction by splitting it into simpler parts, hopefully
  /// understood by the target.
  LegalizeResult lower(MachineInstr &MI, unsigned TypeIdx, LLT Ty);

  /// Legalize a vector instruction by splitting into multiple components, each
  /// acting on the same scalar type as the original but with fewer elements.
  LegalizeResult fewerElementsVector(MachineInstr &MI, unsigned TypeIdx,
                                     LLT NarrowTy);

  /// Legalize a vector instruction by increasing the number of vector elements
  /// involved and ignoring the added elements later.
  LegalizeResult moreElementsVector(MachineInstr &MI, unsigned TypeIdx,
                                    LLT MoreTy);

  /// Cast the given value to an LLT::scalar with an equivalent size. Returns
  /// the register to use if an instruction was inserted. Returns the original
  /// register if no coercion was necessary.
  //
  // This may also fail and return Register() if there is no legal way to cast.
  Register coerceToScalar(Register Val);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Use by extending the operand's type to \p WideTy using the specified \p
  /// ExtOpcode for the extension instruction, and replacing the vreg of the
  /// operand in place.
  void widenScalarSrc(MachineInstr &MI, LLT WideTy, unsigned OpIdx,
                      unsigned ExtOpcode);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Use by truncating the operand's type to \p NarrowTy using G_TRUNC, and
  /// replacing the vreg of the operand in place.
  void narrowScalarSrc(MachineInstr &MI, LLT NarrowTy, unsigned OpIdx);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Def by extending the operand's type to \p WideTy and truncating it back
  /// with the \p TruncOpcode, and replacing the vreg of the operand in place.
  void widenScalarDst(MachineInstr &MI, LLT WideTy, unsigned OpIdx = 0,
                      unsigned TruncOpcode = TargetOpcode::G_TRUNC);

  // Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  // Def by truncating the operand's type to \p NarrowTy, replacing in place and
  // extending back with \p ExtOpcode.
  void narrowScalarDst(MachineInstr &MI, LLT NarrowTy, unsigned OpIdx,
                       unsigned ExtOpcode);
  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Def by performing it with additional vector elements and extracting the
  /// result elements, and replacing the vreg of the operand in place.
  void moreElementsVectorDst(MachineInstr &MI, LLT MoreTy, unsigned OpIdx);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Use by producing a vector with undefined high elements, extracting the
  /// original vector type, and replacing the vreg of the operand in place.
  void moreElementsVectorSrc(MachineInstr &MI, LLT MoreTy, unsigned OpIdx);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// use by inserting a G_BITCAST to \p CastTy
  void bitcastSrc(MachineInstr &MI, LLT CastTy, unsigned OpIdx);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// def by inserting a G_BITCAST from \p CastTy
  void bitcastDst(MachineInstr &MI, LLT CastTy, unsigned OpIdx);

  /// Widen \p OrigReg to \p WideTy by merging to a wider type, padding with
  /// G_IMPLICIT_DEF, and producing dead results.
  Register widenWithUnmerge(LLT WideTy, Register OrigReg);

private:
  LegalizeResult
  widenScalarMergeValues(MachineInstr &MI, unsigned TypeIdx, LLT WideTy);
  LegalizeResult
  widenScalarUnmergeValues(MachineInstr &MI, unsigned TypeIdx, LLT WideTy);
  LegalizeResult
  widenScalarExtract(MachineInstr &MI, unsigned TypeIdx, LLT WideTy);
  LegalizeResult
  widenScalarInsert(MachineInstr &MI, unsigned TypeIdx, LLT WideTy);
  LegalizeResult widenScalarAddSubOverflow(MachineInstr &MI, unsigned TypeIdx,
                                           LLT WideTy);
  LegalizeResult widenScalarAddSubShlSat(MachineInstr &MI, unsigned TypeIdx,
                                         LLT WideTy);
  LegalizeResult widenScalarMulo(MachineInstr &MI, unsigned TypeIdx,
                                 LLT WideTy);

  /// Helper function to split a wide generic register into bitwise blocks with
  /// the given Type (which implies the number of blocks needed). The generic
  /// registers created are appended to Ops, starting at bit 0 of Reg.
  void extractParts(Register Reg, LLT Ty, int NumParts,
                    SmallVectorImpl<Register> &VRegs);

  /// Version which handles irregular splits.
  bool extractParts(Register Reg, LLT RegTy, LLT MainTy,
                    LLT &LeftoverTy,
                    SmallVectorImpl<Register> &VRegs,
                    SmallVectorImpl<Register> &LeftoverVRegs);

  /// Helper function to build a wide generic register \p DstReg of type \p
  /// RegTy from smaller parts. This will produce a G_MERGE_VALUES,
  /// G_BUILD_VECTOR, G_CONCAT_VECTORS, or sequence of G_INSERT as appropriate
  /// for the types.
  ///
  /// \p PartRegs must be registers of type \p PartTy.
  ///
  /// If \p ResultTy does not evenly break into \p PartTy sized pieces, the
  /// remainder must be specified with \p LeftoverRegs of type \p LeftoverTy.
  void insertParts(Register DstReg, LLT ResultTy,
                   LLT PartTy, ArrayRef<Register> PartRegs,
                   LLT LeftoverTy = LLT(), ArrayRef<Register> LeftoverRegs = {});

  /// Unmerge \p SrcReg into smaller sized values, and append them to \p
  /// Parts. The elements of \p Parts will be the greatest common divisor type
  /// of \p DstTy, \p NarrowTy and the type of \p SrcReg. This will compute and
  /// return the GCD type.
  LLT extractGCDType(SmallVectorImpl<Register> &Parts, LLT DstTy,
                     LLT NarrowTy, Register SrcReg);

  /// Unmerge \p SrcReg into \p GCDTy typed registers. This will append all of
  /// the unpacked registers to \p Parts. This version is if the common unmerge
  /// type is already known.
  void extractGCDType(SmallVectorImpl<Register> &Parts, LLT GCDTy,
                      Register SrcReg);

  /// Produce a merge of values in \p VRegs to define \p DstReg. Perform a merge
  /// from the least common multiple type, and convert as appropriate to \p
  /// DstReg.
  ///
  /// \p VRegs should each have type \p GCDTy. This type should be greatest
  /// common divisor type of \p DstReg, \p NarrowTy, and an undetermined source
  /// type.
  ///
  /// \p NarrowTy is the desired result merge source type. If the source value
  /// needs to be widened to evenly cover \p DstReg, inserts high bits
  /// corresponding to the extension opcode \p PadStrategy.
  ///
  /// \p VRegs will be cleared, and the the result \p NarrowTy register pieces
  /// will replace it. Returns The complete LCMTy that \p VRegs will cover when
  /// merged.
  LLT buildLCMMergePieces(LLT DstTy, LLT NarrowTy, LLT GCDTy,
                          SmallVectorImpl<Register> &VRegs,
                          unsigned PadStrategy = TargetOpcode::G_ANYEXT);

  /// Merge the values in \p RemergeRegs to an \p LCMTy typed value. Extract the
  /// low bits into \p DstReg. This is intended to use the outputs from
  /// buildLCMMergePieces after processing.
  void buildWidenedRemergeToDst(Register DstReg, LLT LCMTy,
                                ArrayRef<Register> RemergeRegs);

  /// Perform generic multiplication of values held in multiple registers.
  /// Generated instructions use only types NarrowTy and i1.
  /// Destination can be same or two times size of the source.
  void multiplyRegisters(SmallVectorImpl<Register> &DstRegs,
                         ArrayRef<Register> Src1Regs,
                         ArrayRef<Register> Src2Regs, LLT NarrowTy);

  void changeOpcode(MachineInstr &MI, unsigned NewOpcode);

  LegalizeResult tryNarrowPow2Reduction(MachineInstr &MI, Register SrcReg,
                                        LLT SrcTy, LLT NarrowTy,
                                        unsigned ScalarOpc);

public:
  /// Return the alignment to use for a stack temporary object with the given
  /// type.
  Align getStackTemporaryAlignment(LLT Type, Align MinAlign = Align()) const;

  /// Create a stack temporary based on the size in bytes and the alignment
  MachineInstrBuilder createStackTemporary(TypeSize Bytes, Align Alignment,
                                           MachinePointerInfo &PtrInfo);

  /// Get a pointer to vector element \p Index located in memory for a vector of
  /// type \p VecTy starting at a base address of \p VecPtr. If \p Index is out
  /// of bounds the returned pointer is unspecified, but will be within the
  /// vector bounds.
  Register getVectorElementPointer(Register VecPtr, LLT VecTy, Register Index);

  LegalizeResult fewerElementsVectorImplicitDef(MachineInstr &MI,
                                                unsigned TypeIdx, LLT NarrowTy);

  /// Legalize a instruction with a vector type where each operand may have a
  /// different element type. All type indexes must have the same number of
  /// elements.
  LegalizeResult fewerElementsVectorMultiEltType(MachineInstr &MI,
                                                 unsigned TypeIdx, LLT NarrowTy);

  LegalizeResult fewerElementsVectorCasts(MachineInstr &MI, unsigned TypeIdx,
                                          LLT NarrowTy);

  LegalizeResult
  fewerElementsVectorCmp(MachineInstr &MI, unsigned TypeIdx, LLT NarrowTy);

  LegalizeResult
  fewerElementsVectorSelect(MachineInstr &MI, unsigned TypeIdx, LLT NarrowTy);

  LegalizeResult fewerElementsVectorPhi(MachineInstr &MI,
                                        unsigned TypeIdx, LLT NarrowTy);

  LegalizeResult moreElementsVectorPhi(MachineInstr &MI, unsigned TypeIdx,
                                       LLT MoreTy);
  LegalizeResult moreElementsVectorShuffle(MachineInstr &MI, unsigned TypeIdx,
                                           LLT MoreTy);

  LegalizeResult fewerElementsVectorUnmergeValues(MachineInstr &MI,
                                                  unsigned TypeIdx,
                                                  LLT NarrowTy);
  LegalizeResult fewerElementsVectorMerge(MachineInstr &MI, unsigned TypeIdx,
                                          LLT NarrowTy);
  LegalizeResult fewerElementsVectorExtractInsertVectorElt(MachineInstr &MI,
                                                           unsigned TypeIdx,
                                                           LLT NarrowTy);

  LegalizeResult fewerElementsVectorMulo(MachineInstr &MI, unsigned TypeIdx,
                                         LLT NarrowTy);

  LegalizeResult reduceLoadStoreWidth(GLoadStore &MI, unsigned TypeIdx,
                                      LLT NarrowTy);

  /// Legalize an instruction by reducing the operation width, either by
  /// narrowing the type of the operation or by reducing the number of elements
  /// of a vector.
  /// The used strategy (narrow vs. fewerElements) is decided by \p NarrowTy.
  /// Narrow is used if the scalar type of \p NarrowTy and \p DstTy differ,
  /// fewerElements is used when the scalar type is the same but the number of
  /// elements between \p NarrowTy and \p DstTy differ.
  LegalizeResult reduceOperationWidth(MachineInstr &MI, unsigned TypeIdx,
                                      LLT NarrowTy);

  LegalizeResult fewerElementsVectorSextInReg(MachineInstr &MI, unsigned TypeIdx,
                                              LLT NarrowTy);

  LegalizeResult narrowScalarShiftByConstant(MachineInstr &MI, const APInt &Amt,
                                             LLT HalfTy, LLT ShiftAmtTy);

  LegalizeResult fewerElementsVectorReductions(MachineInstr &MI,
                                               unsigned TypeIdx, LLT NarrowTy);

  LegalizeResult fewerElementsVectorShuffle(MachineInstr &MI, unsigned TypeIdx,
                                            LLT NarrowTy);

  LegalizeResult narrowScalarShift(MachineInstr &MI, unsigned TypeIdx, LLT Ty);
  LegalizeResult narrowScalarAddSub(MachineInstr &MI, unsigned TypeIdx,
                                    LLT NarrowTy);
  LegalizeResult narrowScalarMul(MachineInstr &MI, LLT Ty);
  LegalizeResult narrowScalarFPTOI(MachineInstr &MI, unsigned TypeIdx, LLT Ty);
  LegalizeResult narrowScalarExtract(MachineInstr &MI, unsigned TypeIdx, LLT Ty);
  LegalizeResult narrowScalarInsert(MachineInstr &MI, unsigned TypeIdx, LLT Ty);

  LegalizeResult narrowScalarBasic(MachineInstr &MI, unsigned TypeIdx, LLT Ty);
  LegalizeResult narrowScalarExt(MachineInstr &MI, unsigned TypeIdx, LLT Ty);
  LegalizeResult narrowScalarSelect(MachineInstr &MI, unsigned TypeIdx, LLT Ty);
  LegalizeResult narrowScalarCTLZ(MachineInstr &MI, unsigned TypeIdx, LLT Ty);
  LegalizeResult narrowScalarCTTZ(MachineInstr &MI, unsigned TypeIdx, LLT Ty);
  LegalizeResult narrowScalarCTPOP(MachineInstr &MI, unsigned TypeIdx, LLT Ty);

  /// Perform Bitcast legalize action on G_EXTRACT_VECTOR_ELT.
  LegalizeResult bitcastExtractVectorElt(MachineInstr &MI, unsigned TypeIdx,
                                         LLT CastTy);

  /// Perform Bitcast legalize action on G_INSERT_VECTOR_ELT.
  LegalizeResult bitcastInsertVectorElt(MachineInstr &MI, unsigned TypeIdx,
                                        LLT CastTy);

  LegalizeResult lowerBitcast(MachineInstr &MI);
  LegalizeResult lowerLoad(GAnyLoad &MI);
  LegalizeResult lowerStore(GStore &MI);
  LegalizeResult lowerBitCount(MachineInstr &MI);
  LegalizeResult lowerFunnelShiftWithInverse(MachineInstr &MI);
  LegalizeResult lowerFunnelShiftAsShifts(MachineInstr &MI);
  LegalizeResult lowerFunnelShift(MachineInstr &MI);
  LegalizeResult lowerRotateWithReverseRotate(MachineInstr &MI);
  LegalizeResult lowerRotate(MachineInstr &MI);

  LegalizeResult lowerU64ToF32BitOps(MachineInstr &MI);
  LegalizeResult lowerUITOFP(MachineInstr &MI);
  LegalizeResult lowerSITOFP(MachineInstr &MI);
  LegalizeResult lowerFPTOUI(MachineInstr &MI);
  LegalizeResult lowerFPTOSI(MachineInstr &MI);

  LegalizeResult lowerFPTRUNC_F64_TO_F16(MachineInstr &MI);
  LegalizeResult lowerFPTRUNC(MachineInstr &MI);
  LegalizeResult lowerFPOWI(MachineInstr &MI);

  LegalizeResult lowerMinMax(MachineInstr &MI);
  LegalizeResult lowerFCopySign(MachineInstr &MI);
  LegalizeResult lowerFMinNumMaxNum(MachineInstr &MI);
  LegalizeResult lowerFMad(MachineInstr &MI);
  LegalizeResult lowerIntrinsicRound(MachineInstr &MI);
  LegalizeResult lowerFFloor(MachineInstr &MI);
  LegalizeResult lowerMergeValues(MachineInstr &MI);
  LegalizeResult lowerUnmergeValues(MachineInstr &MI);
  LegalizeResult lowerExtractInsertVectorElt(MachineInstr &MI);
  LegalizeResult lowerShuffleVector(MachineInstr &MI);
  LegalizeResult lowerDynStackAlloc(MachineInstr &MI);
  LegalizeResult lowerExtract(MachineInstr &MI);
  LegalizeResult lowerInsert(MachineInstr &MI);
  LegalizeResult lowerSADDO_SSUBO(MachineInstr &MI);
  LegalizeResult lowerAddSubSatToMinMax(MachineInstr &MI);
  LegalizeResult lowerAddSubSatToAddoSubo(MachineInstr &MI);
  LegalizeResult lowerShlSat(MachineInstr &MI);
  LegalizeResult lowerBswap(MachineInstr &MI);
  LegalizeResult lowerBitreverse(MachineInstr &MI);
  LegalizeResult lowerReadWriteRegister(MachineInstr &MI);
  LegalizeResult lowerSMULH_UMULH(MachineInstr &MI);
  LegalizeResult lowerSelect(MachineInstr &MI);
  LegalizeResult lowerDIVREM(MachineInstr &MI);
  LegalizeResult lowerAbsToAddXor(MachineInstr &MI);
  LegalizeResult lowerAbsToMaxNeg(MachineInstr &MI);
  LegalizeResult lowerIsNaN(MachineInstr &MI);
  LegalizeResult lowerVectorReduction(MachineInstr &MI);
};

/// Helper function that creates a libcall to the given \p Name using the given
/// calling convention \p CC.
LegalizerHelper::LegalizeResult
createLibcall(MachineIRBuilder &MIRBuilder, const char *Name,
              const CallLowering::ArgInfo &Result,
              ArrayRef<CallLowering::ArgInfo> Args, CallingConv::ID CC);

/// Helper function that creates the given libcall.
LegalizerHelper::LegalizeResult
createLibcall(MachineIRBuilder &MIRBuilder, RTLIB::Libcall Libcall,
              const CallLowering::ArgInfo &Result,
              ArrayRef<CallLowering::ArgInfo> Args);

/// Create a libcall to memcpy et al.
LegalizerHelper::LegalizeResult
createMemLibcall(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                 MachineInstr &MI, LostDebugLocObserver &LocObserver);

} // End namespace llvm.

#endif
