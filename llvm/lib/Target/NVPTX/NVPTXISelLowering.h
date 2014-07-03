//===-- NVPTXISelLowering.h - NVPTX DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that NVPTX uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTXISELLOWERING_H
#define NVPTXISELLOWERING_H

#include "NVPTX.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
namespace NVPTXISD {
enum NodeType {
  // Start the numbering from where ISD NodeType finishes.
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  Wrapper,
  CALL,
  RET_FLAG,
  LOAD_PARAM,
  DeclareParam,
  DeclareScalarParam,
  DeclareRetParam,
  DeclareRet,
  DeclareScalarRet,
  PrintCall,
  PrintCallUni,
  CallArgBegin,
  CallArg,
  LastCallArg,
  CallArgEnd,
  CallVoid,
  CallVal,
  CallSymbol,
  Prototype,
  MoveParam,
  PseudoUseParam,
  RETURN,
  CallSeqBegin,
  CallSeqEnd,
  CallPrototype,
  FUN_SHFL_CLAMP,
  FUN_SHFR_CLAMP,
  MUL_WIDE_SIGNED,
  MUL_WIDE_UNSIGNED,
  IMAD,
  Dummy,

  LoadV2 = ISD::FIRST_TARGET_MEMORY_OPCODE,
  LoadV4,
  LDGV2, // LDG.v2
  LDGV4, // LDG.v4
  LDUV2, // LDU.v2
  LDUV4, // LDU.v4
  StoreV2,
  StoreV4,
  LoadParam,
  LoadParamV2,
  LoadParamV4,
  StoreParam,
  StoreParamV2,
  StoreParamV4,
  StoreParamS32, // to sext and store a <32bit value, not used currently
  StoreParamU32, // to zext and store a <32bit value, not used currently 
  StoreRetval,
  StoreRetvalV2,
  StoreRetvalV4,

  // Texture intrinsics
  Tex1DFloatI32,
  Tex1DFloatFloat,
  Tex1DFloatFloatLevel,
  Tex1DFloatFloatGrad,
  Tex1DI32I32,
  Tex1DI32Float,
  Tex1DI32FloatLevel,
  Tex1DI32FloatGrad,
  Tex1DArrayFloatI32,
  Tex1DArrayFloatFloat,
  Tex1DArrayFloatFloatLevel,
  Tex1DArrayFloatFloatGrad,
  Tex1DArrayI32I32,
  Tex1DArrayI32Float,
  Tex1DArrayI32FloatLevel,
  Tex1DArrayI32FloatGrad,
  Tex2DFloatI32,
  Tex2DFloatFloat,
  Tex2DFloatFloatLevel,
  Tex2DFloatFloatGrad,
  Tex2DI32I32,
  Tex2DI32Float,
  Tex2DI32FloatLevel,
  Tex2DI32FloatGrad,
  Tex2DArrayFloatI32,
  Tex2DArrayFloatFloat,
  Tex2DArrayFloatFloatLevel,
  Tex2DArrayFloatFloatGrad,
  Tex2DArrayI32I32,
  Tex2DArrayI32Float,
  Tex2DArrayI32FloatLevel,
  Tex2DArrayI32FloatGrad,
  Tex3DFloatI32,
  Tex3DFloatFloat,
  Tex3DFloatFloatLevel,
  Tex3DFloatFloatGrad,
  Tex3DI32I32,
  Tex3DI32Float,
  Tex3DI32FloatLevel,
  Tex3DI32FloatGrad,

  // Surface intrinsics
  Suld1DI8Trap,
  Suld1DI16Trap,
  Suld1DI32Trap,
  Suld1DV2I8Trap,
  Suld1DV2I16Trap,
  Suld1DV2I32Trap,
  Suld1DV4I8Trap,
  Suld1DV4I16Trap,
  Suld1DV4I32Trap,

  Suld1DArrayI8Trap,
  Suld1DArrayI16Trap,
  Suld1DArrayI32Trap,
  Suld1DArrayV2I8Trap,
  Suld1DArrayV2I16Trap,
  Suld1DArrayV2I32Trap,
  Suld1DArrayV4I8Trap,
  Suld1DArrayV4I16Trap,
  Suld1DArrayV4I32Trap,

  Suld2DI8Trap,
  Suld2DI16Trap,
  Suld2DI32Trap,
  Suld2DV2I8Trap,
  Suld2DV2I16Trap,
  Suld2DV2I32Trap,
  Suld2DV4I8Trap,
  Suld2DV4I16Trap,
  Suld2DV4I32Trap,

  Suld2DArrayI8Trap,
  Suld2DArrayI16Trap,
  Suld2DArrayI32Trap,
  Suld2DArrayV2I8Trap,
  Suld2DArrayV2I16Trap,
  Suld2DArrayV2I32Trap,
  Suld2DArrayV4I8Trap,
  Suld2DArrayV4I16Trap,
  Suld2DArrayV4I32Trap,

  Suld3DI8Trap,
  Suld3DI16Trap,
  Suld3DI32Trap,
  Suld3DV2I8Trap,
  Suld3DV2I16Trap,
  Suld3DV2I32Trap,
  Suld3DV4I8Trap,
  Suld3DV4I16Trap,
  Suld3DV4I32Trap
};
}

class NVPTXSubtarget;

//===--------------------------------------------------------------------===//
// TargetLowering Implementation
//===--------------------------------------------------------------------===//
class NVPTXTargetLowering : public TargetLowering {
public:
  explicit NVPTXTargetLowering(NVPTXTargetMachine &TM);
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerGlobalAddress(const GlobalValue *GV, int64_t Offset,
                             SelectionDAG &DAG) const;

  const char *getTargetNodeName(unsigned Opcode) const override;

  bool isTypeSupportedInIntrinsic(MVT VT) const;

  bool getTgtMemIntrinsic(IntrinsicInfo &Info, const CallInst &I,
                          unsigned Intrinsic) const override;

  /// isLegalAddressingMode - Return true if the addressing mode represented
  /// by AM is legal for this target, for a load/store of the specified type
  /// Used to guide target specific optimizations, like loop strength
  /// reduction (LoopStrengthReduce.cpp) and memory optimization for
  /// address mode (CodeGenPrepare.cpp)
  bool isLegalAddressingMode(const AddrMode &AM, Type *Ty) const override;

  /// getFunctionAlignment - Return the Log2 alignment of this function.
  unsigned getFunctionAlignment(const Function *F) const;

  EVT getSetCCResultType(LLVMContext &Ctx, EVT VT) const override {
    if (VT.isVector())
      return EVT::getVectorVT(Ctx, MVT::i1, VT.getVectorNumElements());
    return MVT::i1;
  }

  ConstraintType
  getConstraintType(const std::string &Constraint) const override;
  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const std::string &Constraint,
                               MVT VT) const override;

  SDValue LowerFormalArguments(
      SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
      const SmallVectorImpl<ISD::InputArg> &Ins, SDLoc dl, SelectionDAG &DAG,
      SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerCall(CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  std::string getPrototype(Type *, const ArgListTy &,
                           const SmallVectorImpl<ISD::OutputArg> &,
                           unsigned retAlignment,
                           const ImmutableCallSite *CS) const;

  SDValue
  LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
              const SmallVectorImpl<ISD::OutputArg> &Outs,
              const SmallVectorImpl<SDValue> &OutVals, SDLoc dl,
              SelectionDAG &DAG) const override;

  void LowerAsmOperandForConstraint(SDValue Op, std::string &Constraint,
                                    std::vector<SDValue> &Ops,
                                    SelectionDAG &DAG) const override;

  NVPTXTargetMachine *nvTM;

  // PTX always uses 32-bit shift amounts
  MVT getScalarShiftAmountTy(EVT LHSTy) const override { return MVT::i32; }

  TargetLoweringBase::LegalizeTypeAction
  getPreferredVectorAction(EVT VT) const override;

private:
  const NVPTXSubtarget &nvptxSubtarget; // cache the subtarget here

  SDValue getExtSymb(SelectionDAG &DAG, const char *name, int idx,
                     EVT = MVT::i32) const;
  SDValue getParamSymbol(SelectionDAG &DAG, int idx, EVT) const;
  SDValue getParamHelpSymbol(SelectionDAG &DAG, int idx);

  SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOADi1(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTOREi1(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTOREVector(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerShiftRightParts(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerShiftLeftParts(SDValue Op, SelectionDAG &DAG) const;

  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;
  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

  unsigned getArgumentAlignment(SDValue Callee, const ImmutableCallSite *CS,
                                Type *Ty, unsigned Idx) const;
};
} // namespace llvm

#endif // NVPTXISELLOWERING_H
