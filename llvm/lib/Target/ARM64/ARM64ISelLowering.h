//==-- ARM64ISelLowering.h - ARM64 DAG Lowering Interface --------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that ARM64 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ARM64_ISELLOWERING_H
#define LLVM_TARGET_ARM64_ISELLOWERING_H

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {

namespace ARM64ISD {

enum {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  WrapperLarge, // 4-instruction MOVZ/MOVK sequence for 64-bit addresses.
  CALL,         // Function call.

  // Almost the same as a normal call node, except that a TLSDesc relocation is
  // needed so the linker can relax it correctly if possible.
  TLSDESC_CALL,
  ADRP,     // Page address of a TargetGlobalAddress operand.
  ADDlow,   // Add the low 12 bits of a TargetGlobalAddress operand.
  LOADgot,  // Load from automatically generated descriptor (e.g. Global
            // Offset Table, TLS record).
  RET_FLAG, // Return with a flag operand. Operand 0 is the chain operand.
  BRCOND,   // Conditional branch instruction; "b.cond".
  CSEL,
  FCSEL, // Conditional move instruction.
  CSINV, // Conditional select invert.
  CSNEG, // Conditional select negate.
  CSINC, // Conditional select increment.

  // Pointer to the thread's local storage area. Materialised from TPIDR_EL0 on
  // ELF.
  THREAD_POINTER,
  ADC,
  SBC, // adc, sbc instructions

  // Arithmetic instructions which write flags.
  ADDS,
  SUBS,
  ADCS,
  SBCS,
  ANDS,

  // Floating point comparison
  FCMP,

  // Floating point max and min instructions.
  FMAX,
  FMIN,

  // Scalar extract
  EXTR,

  // Scalar-to-vector duplication
  DUP,
  DUPLANE8,
  DUPLANE16,
  DUPLANE32,
  DUPLANE64,

  // Vector immedate moves
  MOVI,
  MOVIshift,
  MOVIedit,
  MOVImsl,
  FMOV,
  MVNIshift,
  MVNImsl,

  // Vector immediate ops
  BICi,
  ORRi,

  // Vector arithmetic negation
  NEG,

  // Vector shuffles
  ZIP1,
  ZIP2,
  UZP1,
  UZP2,
  TRN1,
  TRN2,
  REV16,
  REV32,
  REV64,
  EXT,

  // Vector shift by scalar
  VSHL,
  VLSHR,
  VASHR,

  // Vector shift by scalar (again)
  SQSHL_I,
  UQSHL_I,
  SQSHLU_I,
  SRSHR_I,
  URSHR_I,

  // Vector comparisons
  CMEQ,
  CMGE,
  CMGT,
  CMHI,
  CMHS,
  FCMEQ,
  FCMGE,
  FCMGT,

  // Vector zero comparisons
  CMEQz,
  CMGEz,
  CMGTz,
  CMLEz,
  CMLTz,
  FCMEQz,
  FCMGEz,
  FCMGTz,
  FCMLEz,
  FCMLTz,

  // Vector bitwise negation
  NOT,

  // Vector bitwise selection
  BIT,

  // Compare-and-branch
  CBZ,
  CBNZ,
  TBZ,
  TBNZ,

  // Tail calls
  TC_RETURN,

  // Custom prefetch handling
  PREFETCH,

  // {s|u}int to FP within a FP register.
  SITOF,
  UITOF
};

} // end namespace ARM64ISD

class ARM64Subtarget;
class ARM64TargetMachine;

class ARM64TargetLowering : public TargetLowering {
  bool RequireStrictAlign;

public:
  explicit ARM64TargetLowering(ARM64TargetMachine &TM);

  /// Selects the correct CCAssignFn for a the given CallingConvention
  /// value.
  CCAssignFn *CCAssignFnForCall(CallingConv::ID CC, bool IsVarArg) const;

  /// computeMaskedBitsForTargetNode - Determine which of the bits specified in
  /// Mask are known to be either zero or one and return them in the
  /// KnownZero/KnownOne bitsets.
  void computeMaskedBitsForTargetNode(const SDValue Op, APInt &KnownZero,
                                      APInt &KnownOne, const SelectionDAG &DAG,
                                      unsigned Depth = 0) const;

  virtual MVT getScalarShiftAmountTy(EVT LHSTy) const;

  /// allowsUnalignedMemoryAccesses - Returns true if the target allows
  /// unaligned memory accesses. of the specified type.
  virtual bool allowsUnalignedMemoryAccesses(EVT VT, unsigned AddrSpace = 0,
                                             bool *Fast = 0) const {
    if (RequireStrictAlign)
      return false;
    // FIXME: True for Cyclone, but not necessary others.
    if (Fast)
      *Fast = true;
    return true;
  }

  /// LowerOperation - Provide custom lowering hooks for some operations.
  virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

  virtual const char *getTargetNodeName(unsigned Opcode) const;

  virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

  /// getFunctionAlignment - Return the Log2 alignment of this function.
  virtual unsigned getFunctionAlignment(const Function *F) const;

  /// getMaximalGlobalOffset - Returns the maximal possible offset which can
  /// be used for loads / stores from the global.
  virtual unsigned getMaximalGlobalOffset() const;

  /// Returns true if a cast between SrcAS and DestAS is a noop.
  virtual bool isNoopAddrSpaceCast(unsigned SrcAS, unsigned DestAS) const {
    // Addrspacecasts are always noops.
    return true;
  }

  /// createFastISel - This method returns a target specific FastISel object,
  /// or null if the target does not support "fast" ISel.
  virtual FastISel *createFastISel(FunctionLoweringInfo &funcInfo,
                                   const TargetLibraryInfo *libInfo) const;

  virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

  virtual bool isFPImmLegal(const APFloat &Imm, EVT VT) const;

  /// isShuffleMaskLegal - Return true if the given shuffle mask can be
  /// codegen'd directly, or if it should be stack expanded.
  virtual bool isShuffleMaskLegal(const SmallVectorImpl<int> &M, EVT VT) const;

  /// getSetCCResultType - Return the ISD::SETCC ValueType
  virtual EVT getSetCCResultType(LLVMContext &Context, EVT VT) const;

  SDValue ReconstructShuffle(SDValue Op, SelectionDAG &DAG) const;

  MachineBasicBlock *EmitAtomicBinary(MachineInstr *MI, MachineBasicBlock *BB,
                                      unsigned Size, unsigned BinOpcode) const;
  MachineBasicBlock *EmitAtomicCmpSwap(MachineInstr *MI, MachineBasicBlock *BB,
                                       unsigned Size) const;
  MachineBasicBlock *EmitAtomicBinary128(MachineInstr *MI,
                                         MachineBasicBlock *BB,
                                         unsigned BinOpcodeLo,
                                         unsigned BinOpcodeHi) const;
  MachineBasicBlock *EmitAtomicCmpSwap128(MachineInstr *MI,
                                          MachineBasicBlock *BB) const;
  MachineBasicBlock *EmitAtomicMinMax128(MachineInstr *MI,
                                         MachineBasicBlock *BB,
                                         unsigned CondCode) const;
  MachineBasicBlock *EmitF128CSEL(MachineInstr *MI,
                                  MachineBasicBlock *BB) const;

  virtual MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr *MI, MachineBasicBlock *MBB) const;

  virtual bool getTgtMemIntrinsic(IntrinsicInfo &Info, const CallInst &I,
                                  unsigned Intrinsic) const;

  virtual bool isTruncateFree(Type *Ty1, Type *Ty2) const;
  virtual bool isTruncateFree(EVT VT1, EVT VT2) const;

  virtual bool isZExtFree(Type *Ty1, Type *Ty2) const;
  virtual bool isZExtFree(EVT VT1, EVT VT2) const;
  virtual bool isZExtFree(SDValue Val, EVT VT2) const;

  virtual bool hasPairedLoad(Type *LoadedType,
                             unsigned &RequiredAligment) const;
  virtual bool hasPairedLoad(EVT LoadedType, unsigned &RequiredAligment) const;

  virtual bool isLegalAddImmediate(int64_t) const;
  virtual bool isLegalICmpImmediate(int64_t) const;

  virtual EVT getOptimalMemOpType(uint64_t Size, unsigned DstAlign,
                                  unsigned SrcAlign, bool IsMemset,
                                  bool ZeroMemset, bool MemcpyStrSrc,
                                  MachineFunction &MF) const;

  /// isLegalAddressingMode - Return true if the addressing mode represented
  /// by AM is legal for this target, for a load/store of the specified type.
  virtual bool isLegalAddressingMode(const AddrMode &AM, Type *Ty) const;

  /// \brief Return the cost of the scaling factor used in the addressing
  /// mode represented by AM for this target, for a load/store
  /// of the specified type.
  /// If the AM is supported, the return value must be >= 0.
  /// If the AM is not supported, it returns a negative value.
  virtual int getScalingFactorCost(const AddrMode &AM, Type *Ty) const;

  /// isFMAFasterThanFMulAndFAdd - Return true if an FMA operation is faster
  /// than a pair of fmul and fadd instructions. fmuladd intrinsics will be
  /// expanded to FMAs when this method returns true, otherwise fmuladd is
  /// expanded to fmul + fadd.
  virtual bool isFMAFasterThanFMulAndFAdd(EVT VT) const;

  virtual const uint16_t *getScratchRegisters(CallingConv::ID CC) const;

  virtual bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                                 Type *Ty) const;

private:
  /// Subtarget - Keep a pointer to the ARM64Subtarget around so that we can
  /// make the right decision when generating code for different targets.
  const ARM64Subtarget *Subtarget;

  void addTypeForNEON(EVT VT, EVT PromotedBitwiseVT);
  void addDRTypeForNEON(MVT VT);
  void addQRTypeForNEON(MVT VT);

  virtual SDValue
  LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                       const SmallVectorImpl<ISD::InputArg> &Ins, SDLoc DL,
                       SelectionDAG &DAG,
                       SmallVectorImpl<SDValue> &InVals) const;

  virtual SDValue LowerCall(CallLoweringInfo & /*CLI*/,
                            SmallVectorImpl<SDValue> &InVals) const;

  SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                          CallingConv::ID CallConv, bool isVarArg,
                          const SmallVectorImpl<ISD::InputArg> &Ins, SDLoc DL,
                          SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals,
                          bool isThisReturn, SDValue ThisVal) const;

  bool isEligibleForTailCallOptimization(
      SDValue Callee, CallingConv::ID CalleeCC, bool isVarArg,
      bool isCalleeStructRet, bool isCallerStructRet,
      const SmallVectorImpl<ISD::OutputArg> &Outs,
      const SmallVectorImpl<SDValue> &OutVals,
      const SmallVectorImpl<ISD::InputArg> &Ins, SelectionDAG &DAG) const;

  void saveVarArgRegisters(CCState &CCInfo, SelectionDAG &DAG, SDLoc DL,
                           SDValue &Chain) const;

  virtual bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                              bool isVarArg,
                              const SmallVectorImpl<ISD::OutputArg> &Outs,
                              LLVMContext &Context) const;

  virtual SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                              bool isVarArg,
                              const SmallVectorImpl<ISD::OutputArg> &Outs,
                              const SmallVectorImpl<SDValue> &OutVals, SDLoc DL,
                              SelectionDAG &DAG) const;

  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerDarwinGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerELFGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerELFTLSDescCall(SDValue SymAddr, SDValue DescAddr, SDLoc DL,
                              SelectionDAG &DAG) const;
  SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerAAPCS_VASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerDarwin_VASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVACOPY(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVAARG(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSCALAR_TO_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEXTRACT_SUBVECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVectorSRA_SRL_SHL(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerShiftLeftParts(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerShiftRightParts(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVSETCC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCTPOP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerF128Call(SDValue Op, SelectionDAG &DAG,
                        RTLIB::Libcall Call) const;
  SDValue LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFP_EXTEND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFP_ROUND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVectorAND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVectorOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFSINCOS(SDValue Op, SelectionDAG &DAG) const;

  ConstraintType getConstraintType(const std::string &Constraint) const;

  /// Examine constraint string and operand type and determine a weight value.
  /// The operand object must already have been set up with the operand type.
  ConstraintWeight getSingleConstraintMatchWeight(AsmOperandInfo &info,
                                                  const char *constraint) const;

  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const std::string &Constraint, MVT VT) const;
  void LowerAsmOperandForConstraint(SDValue Op, std::string &Constraint,
                                    std::vector<SDValue> &Ops,
                                    SelectionDAG &DAG) const;

  bool isUsedByReturnOnly(SDNode *N, SDValue &Chain) const;
  bool mayBeEmittedAsTailCall(CallInst *CI) const;
  bool getIndexedAddressParts(SDNode *Op, SDValue &Base, SDValue &Offset,
                              ISD::MemIndexedMode &AM, bool &IsInc,
                              SelectionDAG &DAG) const;
  bool getPreIndexedAddressParts(SDNode *N, SDValue &Base, SDValue &Offset,
                                 ISD::MemIndexedMode &AM,
                                 SelectionDAG &DAG) const;
  bool getPostIndexedAddressParts(SDNode *N, SDNode *Op, SDValue &Base,
                                  SDValue &Offset, ISD::MemIndexedMode &AM,
                                  SelectionDAG &DAG) const;

  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const;
};

namespace ARM64 {
FastISel *createFastISel(FunctionLoweringInfo &funcInfo,
                         const TargetLibraryInfo *libInfo);
} // end namespace ARM64

} // end namespace llvm

#endif // LLVM_TARGET_ARM64_ISELLOWERING_H
