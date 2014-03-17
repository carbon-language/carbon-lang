//===-- AMDILISelLowering.cpp - AMDIL DAG Lowering Implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// \brief TargetLowering functions borrowed from AMDIL.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUISelLowering.h"
#include "AMDGPURegisterInfo.h"
#include "AMDGPUSubtarget.h"
#include "AMDILIntrinsicInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;
//===----------------------------------------------------------------------===//
// TargetLowering Implementation Help Functions End
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TargetLowering Class Implementation Begins
//===----------------------------------------------------------------------===//
void AMDGPUTargetLowering::InitAMDILLowering() {
  static const int types[] = {
    (int)MVT::i8,
    (int)MVT::i16,
    (int)MVT::i32,
    (int)MVT::f32,
    (int)MVT::f64,
    (int)MVT::i64,
    (int)MVT::v2i8,
    (int)MVT::v4i8,
    (int)MVT::v2i16,
    (int)MVT::v4i16,
    (int)MVT::v4f32,
    (int)MVT::v4i32,
    (int)MVT::v2f32,
    (int)MVT::v2i32,
    (int)MVT::v2f64,
    (int)MVT::v2i64
  };

  static const int IntTypes[] = {
    (int)MVT::i8,
    (int)MVT::i16,
    (int)MVT::i32,
    (int)MVT::i64
  };

  static const int FloatTypes[] = {
    (int)MVT::f32,
    (int)MVT::f64
  };

  static const int VectorTypes[] = {
    (int)MVT::v2i8,
    (int)MVT::v4i8,
    (int)MVT::v2i16,
    (int)MVT::v4i16,
    (int)MVT::v4f32,
    (int)MVT::v4i32,
    (int)MVT::v2f32,
    (int)MVT::v2i32,
    (int)MVT::v2f64,
    (int)MVT::v2i64
  };
  const size_t NumTypes = array_lengthof(types);
  const size_t NumFloatTypes = array_lengthof(FloatTypes);
  const size_t NumIntTypes = array_lengthof(IntTypes);
  const size_t NumVectorTypes = array_lengthof(VectorTypes);

  const AMDGPUSubtarget &STM = getTargetMachine().getSubtarget<AMDGPUSubtarget>();
  // These are the current register classes that are
  // supported

  for (unsigned int x  = 0; x < NumTypes; ++x) {
    MVT::SimpleValueType VT = (MVT::SimpleValueType)types[x];

    setOperationAction(ISD::SUBE, VT, Expand);
    setOperationAction(ISD::SUBC, VT, Expand);
    setOperationAction(ISD::ADDE, VT, Expand);
    setOperationAction(ISD::ADDC, VT, Expand);
    setOperationAction(ISD::BRCOND, VT, Custom);
    setOperationAction(ISD::BR_JT, VT, Expand);
    setOperationAction(ISD::BRIND, VT, Expand);
    // TODO: Implement custom UREM/SREM routines
    setOperationAction(ISD::SREM, VT, Expand);
    setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, VT, Expand);
    if (VT != MVT::i64 && VT != MVT::v2i64) {
      setOperationAction(ISD::SDIV, VT, Custom);
    }
  }
  for (unsigned int x = 0; x < NumFloatTypes; ++x) {
    MVT::SimpleValueType VT = (MVT::SimpleValueType)FloatTypes[x];

    // IL does not have these operations for floating point types
    setOperationAction(ISD::FP_ROUND_INREG, VT, Expand);
    setOperationAction(ISD::SETOLT, VT, Expand);
    setOperationAction(ISD::SETOGE, VT, Expand);
    setOperationAction(ISD::SETOGT, VT, Expand);
    setOperationAction(ISD::SETOLE, VT, Expand);
    setOperationAction(ISD::SETULT, VT, Expand);
    setOperationAction(ISD::SETUGE, VT, Expand);
    setOperationAction(ISD::SETUGT, VT, Expand);
    setOperationAction(ISD::SETULE, VT, Expand);
  }

  for (unsigned int x = 0; x < NumIntTypes; ++x) {
    MVT::SimpleValueType VT = (MVT::SimpleValueType)IntTypes[x];

    // GPU also does not have divrem function for signed or unsigned
    setOperationAction(ISD::SDIVREM, VT, Expand);

    // GPU does not have [S|U]MUL_LOHI functions as a single instruction
    setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, VT, Expand);

    setOperationAction(ISD::BSWAP, VT, Expand);

    // GPU doesn't have any counting operators
    setOperationAction(ISD::CTPOP, VT, Expand);
    setOperationAction(ISD::CTTZ, VT, Expand);
    setOperationAction(ISD::CTLZ, VT, Expand);
  }

  for (unsigned int ii = 0; ii < NumVectorTypes; ++ii) {
    MVT::SimpleValueType VT = (MVT::SimpleValueType)VectorTypes[ii];

    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Expand);
    setOperationAction(ISD::SDIVREM, VT, Expand);
    setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    // setOperationAction(ISD::VSETCC, VT, Expand);
    setOperationAction(ISD::SELECT_CC, VT, Expand);

  }
  setOperationAction(ISD::MULHU, MVT::i64, Expand);
  setOperationAction(ISD::MULHU, MVT::v2i64, Expand);
  setOperationAction(ISD::MULHS, MVT::i64, Expand);
  setOperationAction(ISD::MULHS, MVT::v2i64, Expand);
  setOperationAction(ISD::ADD, MVT::v2i64, Expand);
  setOperationAction(ISD::SREM, MVT::v2i64, Expand);
  setOperationAction(ISD::Constant          , MVT::i64  , Legal);
  setOperationAction(ISD::SDIV, MVT::v2i64, Expand);
  setOperationAction(ISD::TRUNCATE, MVT::v2i64, Expand);
  setOperationAction(ISD::SIGN_EXTEND, MVT::v2i64, Expand);
  setOperationAction(ISD::ZERO_EXTEND, MVT::v2i64, Expand);
  setOperationAction(ISD::ANY_EXTEND, MVT::v2i64, Expand);
  if (STM.hasHWFP64()) {
    // we support loading/storing v2f64 but not operations on the type
    setOperationAction(ISD::FADD, MVT::v2f64, Expand);
    setOperationAction(ISD::FSUB, MVT::v2f64, Expand);
    setOperationAction(ISD::FMUL, MVT::v2f64, Expand);
    setOperationAction(ISD::FP_ROUND_INREG, MVT::v2f64, Expand);
    setOperationAction(ISD::FP_EXTEND, MVT::v2f64, Expand);
    setOperationAction(ISD::ConstantFP        , MVT::f64  , Legal);
    // We want to expand vector conversions into their scalar
    // counterparts.
    setOperationAction(ISD::TRUNCATE, MVT::v2f64, Expand);
    setOperationAction(ISD::SIGN_EXTEND, MVT::v2f64, Expand);
    setOperationAction(ISD::ZERO_EXTEND, MVT::v2f64, Expand);
    setOperationAction(ISD::ANY_EXTEND, MVT::v2f64, Expand);
    setOperationAction(ISD::FABS, MVT::f64, Expand);
    setOperationAction(ISD::FABS, MVT::v2f64, Expand);
  }
  // TODO: Fix the UDIV24 algorithm so it works for these
  // types correctly. This needs vector comparisons
  // for this to work correctly.
  setOperationAction(ISD::UDIV, MVT::v2i8, Expand);
  setOperationAction(ISD::UDIV, MVT::v4i8, Expand);
  setOperationAction(ISD::UDIV, MVT::v2i16, Expand);
  setOperationAction(ISD::UDIV, MVT::v4i16, Expand);
  setOperationAction(ISD::SUBC, MVT::Other, Expand);
  setOperationAction(ISD::ADDE, MVT::Other, Expand);
  setOperationAction(ISD::ADDC, MVT::Other, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);


  // Use the default implementation.
  setOperationAction(ISD::ConstantFP        , MVT::f32    , Legal);
  setOperationAction(ISD::Constant          , MVT::i32    , Legal);

  setSchedulingPreference(Sched::RegPressure);
  setPow2DivIsCheap(false);
  setSelectIsExpensive(true);
  setJumpIsExpensive(true);

  MaxStoresPerMemcpy  = 4096;
  MaxStoresPerMemmove = 4096;
  MaxStoresPerMemset  = 4096;

}

bool
AMDGPUTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
    const CallInst &I, unsigned Intrinsic) const {
  return false;
}

// The backend supports 32 and 64 bit floating point immediates
bool
AMDGPUTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const {
  if (VT.getScalarType().getSimpleVT().SimpleTy == MVT::f32
      || VT.getScalarType().getSimpleVT().SimpleTy == MVT::f64) {
    return true;
  } else {
    return false;
  }
}

bool
AMDGPUTargetLowering::ShouldShrinkFPConstant(EVT VT) const {
  if (VT.getScalarType().getSimpleVT().SimpleTy == MVT::f32
      || VT.getScalarType().getSimpleVT().SimpleTy == MVT::f64) {
    return false;
  } else {
    return true;
  }
}


// isMaskedValueZeroForTargetNode - Return true if 'Op & Mask' is known to
// be zero. Op is expected to be a target specific node. Used by DAG
// combiner.

void
AMDGPUTargetLowering::computeMaskedBitsForTargetNode(
    const SDValue Op,
    APInt &KnownZero,
    APInt &KnownOne,
    const SelectionDAG &DAG,
    unsigned Depth) const {
  APInt KnownZero2;
  APInt KnownOne2;
  KnownZero = KnownOne = APInt(KnownOne.getBitWidth(), 0); // Don't know anything
  switch (Op.getOpcode()) {
    default: break;
    case ISD::SELECT_CC:
             DAG.ComputeMaskedBits(
                 Op.getOperand(1),
                 KnownZero,
                 KnownOne,
                 Depth + 1
                 );
             DAG.ComputeMaskedBits(
                 Op.getOperand(0),
                 KnownZero2,
                 KnownOne2
                 );
             assert((KnownZero & KnownOne) == 0
                 && "Bits known to be one AND zero?");
             assert((KnownZero2 & KnownOne2) == 0
                 && "Bits known to be one AND zero?");
             // Only known if known in both the LHS and RHS
             KnownOne &= KnownOne2;
             KnownZero &= KnownZero2;
             break;
  };
}

//===----------------------------------------------------------------------===//
//                           Other Lowering Hooks
//===----------------------------------------------------------------------===//

SDValue
AMDGPUTargetLowering::LowerSDIV(SDValue Op, SelectionDAG &DAG) const {
  EVT OVT = Op.getValueType();
  SDValue DST;
  if (OVT.getScalarType() == MVT::i64) {
    DST = LowerSDIV64(Op, DAG);
  } else if (OVT.getScalarType() == MVT::i32) {
    DST = LowerSDIV32(Op, DAG);
  } else if (OVT.getScalarType() == MVT::i16
      || OVT.getScalarType() == MVT::i8) {
    DST = LowerSDIV24(Op, DAG);
  } else {
    DST = SDValue(Op.getNode(), 0);
  }
  return DST;
}

SDValue
AMDGPUTargetLowering::LowerSREM(SDValue Op, SelectionDAG &DAG) const {
  EVT OVT = Op.getValueType();
  SDValue DST;
  if (OVT.getScalarType() == MVT::i64) {
    DST = LowerSREM64(Op, DAG);
  } else if (OVT.getScalarType() == MVT::i32) {
    DST = LowerSREM32(Op, DAG);
  } else if (OVT.getScalarType() == MVT::i16) {
    DST = LowerSREM16(Op, DAG);
  } else if (OVT.getScalarType() == MVT::i8) {
    DST = LowerSREM8(Op, DAG);
  } else {
    DST = SDValue(Op.getNode(), 0);
  }
  return DST;
}

EVT
AMDGPUTargetLowering::genIntType(uint32_t size, uint32_t numEle) const {
  int iSize = (size * numEle);
  int vEle = (iSize >> ((size == 64) ? 6 : 5));
  if (!vEle) {
    vEle = 1;
  }
  if (size == 64) {
    if (vEle == 1) {
      return EVT(MVT::i64);
    } else {
      return EVT(MVT::getVectorVT(MVT::i64, vEle));
    }
  } else {
    if (vEle == 1) {
      return EVT(MVT::i32);
    } else {
      return EVT(MVT::getVectorVT(MVT::i32, vEle));
    }
  }
}

SDValue
AMDGPUTargetLowering::LowerBRCOND(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Cond  = Op.getOperand(1);
  SDValue Jump  = Op.getOperand(2);
  SDValue Result;
  Result = DAG.getNode(
      AMDGPUISD::BRANCH_COND,
      SDLoc(Op),
      Op.getValueType(),
      Chain, Jump, Cond);
  return Result;
}

SDValue
AMDGPUTargetLowering::LowerSDIV24(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT OVT = Op.getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  MVT INTTY;
  MVT FLTTY;
  if (!OVT.isVector()) {
    INTTY = MVT::i32;
    FLTTY = MVT::f32;
  } else if (OVT.getVectorNumElements() == 2) {
    INTTY = MVT::v2i32;
    FLTTY = MVT::v2f32;
  } else if (OVT.getVectorNumElements() == 4) {
    INTTY = MVT::v4i32;
    FLTTY = MVT::v4f32;
  }
  unsigned bitsize = OVT.getScalarType().getSizeInBits();
  // char|short jq = ia ^ ib;
  SDValue jq = DAG.getNode(ISD::XOR, DL, OVT, LHS, RHS);

  // jq = jq >> (bitsize - 2)
  jq = DAG.getNode(ISD::SRA, DL, OVT, jq, DAG.getConstant(bitsize - 2, OVT)); 

  // jq = jq | 0x1
  jq = DAG.getNode(ISD::OR, DL, OVT, jq, DAG.getConstant(1, OVT));

  // jq = (int)jq
  jq = DAG.getSExtOrTrunc(jq, DL, INTTY);

  // int ia = (int)LHS;
  SDValue ia = DAG.getSExtOrTrunc(LHS, DL, INTTY);

  // int ib, (int)RHS;
  SDValue ib = DAG.getSExtOrTrunc(RHS, DL, INTTY);

  // float fa = (float)ia;
  SDValue fa = DAG.getNode(ISD::SINT_TO_FP, DL, FLTTY, ia);

  // float fb = (float)ib;
  SDValue fb = DAG.getNode(ISD::SINT_TO_FP, DL, FLTTY, ib);

  // float fq = native_divide(fa, fb);
  SDValue fq = DAG.getNode(AMDGPUISD::DIV_INF, DL, FLTTY, fa, fb);

  // fq = trunc(fq);
  fq = DAG.getNode(ISD::FTRUNC, DL, FLTTY, fq);

  // float fqneg = -fq;
  SDValue fqneg = DAG.getNode(ISD::FNEG, DL, FLTTY, fq);

  // float fr = mad(fqneg, fb, fa);
  SDValue fr = DAG.getNode(ISD::FADD, DL, FLTTY,
      DAG.getNode(ISD::MUL, DL, FLTTY, fqneg, fb), fa);

  // int iq = (int)fq;
  SDValue iq = DAG.getNode(ISD::FP_TO_SINT, DL, INTTY, fq);

  // fr = fabs(fr);
  fr = DAG.getNode(ISD::FABS, DL, FLTTY, fr);

  // fb = fabs(fb);
  fb = DAG.getNode(ISD::FABS, DL, FLTTY, fb);

  // int cv = fr >= fb;
  SDValue cv;
  if (INTTY == MVT::i32) {
    cv = DAG.getSetCC(DL, INTTY, fr, fb, ISD::SETOGE);
  } else {
    cv = DAG.getSetCC(DL, INTTY, fr, fb, ISD::SETOGE);
  }
  // jq = (cv ? jq : 0);
  jq = DAG.getNode(ISD::SELECT, DL, OVT, cv, jq, 
      DAG.getConstant(0, OVT));
  // dst = iq + jq;
  iq = DAG.getSExtOrTrunc(iq, DL, OVT);
  iq = DAG.getNode(ISD::ADD, DL, OVT, iq, jq);
  return iq;
}

SDValue
AMDGPUTargetLowering::LowerSDIV32(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT OVT = Op.getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  // The LowerSDIV32 function generates equivalent to the following IL.
  // mov r0, LHS
  // mov r1, RHS
  // ilt r10, r0, 0
  // ilt r11, r1, 0
  // iadd r0, r0, r10
  // iadd r1, r1, r11
  // ixor r0, r0, r10
  // ixor r1, r1, r11
  // udiv r0, r0, r1
  // ixor r10, r10, r11
  // iadd r0, r0, r10
  // ixor DST, r0, r10

  // mov r0, LHS
  SDValue r0 = LHS;

  // mov r1, RHS
  SDValue r1 = RHS;

  // ilt r10, r0, 0
  SDValue r10 = DAG.getSelectCC(DL,
      r0, DAG.getConstant(0, OVT),
      DAG.getConstant(-1, MVT::i32),
      DAG.getConstant(0, MVT::i32),
      ISD::SETLT);

  // ilt r11, r1, 0
  SDValue r11 = DAG.getSelectCC(DL,
      r1, DAG.getConstant(0, OVT),
      DAG.getConstant(-1, MVT::i32),
      DAG.getConstant(0, MVT::i32),
      ISD::SETLT);

  // iadd r0, r0, r10
  r0 = DAG.getNode(ISD::ADD, DL, OVT, r0, r10);

  // iadd r1, r1, r11
  r1 = DAG.getNode(ISD::ADD, DL, OVT, r1, r11);

  // ixor r0, r0, r10
  r0 = DAG.getNode(ISD::XOR, DL, OVT, r0, r10);

  // ixor r1, r1, r11
  r1 = DAG.getNode(ISD::XOR, DL, OVT, r1, r11);

  // udiv r0, r0, r1
  r0 = DAG.getNode(ISD::UDIV, DL, OVT, r0, r1);

  // ixor r10, r10, r11
  r10 = DAG.getNode(ISD::XOR, DL, OVT, r10, r11);

  // iadd r0, r0, r10
  r0 = DAG.getNode(ISD::ADD, DL, OVT, r0, r10);

  // ixor DST, r0, r10
  SDValue DST = DAG.getNode(ISD::XOR, DL, OVT, r0, r10); 
  return DST;
}

SDValue
AMDGPUTargetLowering::LowerSDIV64(SDValue Op, SelectionDAG &DAG) const {
  return SDValue(Op.getNode(), 0);
}

SDValue
AMDGPUTargetLowering::LowerSREM8(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT OVT = Op.getValueType();
  MVT INTTY = MVT::i32;
  if (OVT == MVT::v2i8) {
    INTTY = MVT::v2i32;
  } else if (OVT == MVT::v4i8) {
    INTTY = MVT::v4i32;
  }
  SDValue LHS = DAG.getSExtOrTrunc(Op.getOperand(0), DL, INTTY);
  SDValue RHS = DAG.getSExtOrTrunc(Op.getOperand(1), DL, INTTY);
  LHS = DAG.getNode(ISD::SREM, DL, INTTY, LHS, RHS);
  LHS = DAG.getSExtOrTrunc(LHS, DL, OVT);
  return LHS;
}

SDValue
AMDGPUTargetLowering::LowerSREM16(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT OVT = Op.getValueType();
  MVT INTTY = MVT::i32;
  if (OVT == MVT::v2i16) {
    INTTY = MVT::v2i32;
  } else if (OVT == MVT::v4i16) {
    INTTY = MVT::v4i32;
  }
  SDValue LHS = DAG.getSExtOrTrunc(Op.getOperand(0), DL, INTTY);
  SDValue RHS = DAG.getSExtOrTrunc(Op.getOperand(1), DL, INTTY);
  LHS = DAG.getNode(ISD::SREM, DL, INTTY, LHS, RHS);
  LHS = DAG.getSExtOrTrunc(LHS, DL, OVT);
  return LHS;
}

SDValue
AMDGPUTargetLowering::LowerSREM32(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT OVT = Op.getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  // The LowerSREM32 function generates equivalent to the following IL.
  // mov r0, LHS
  // mov r1, RHS
  // ilt r10, r0, 0
  // ilt r11, r1, 0
  // iadd r0, r0, r10
  // iadd r1, r1, r11
  // ixor r0, r0, r10
  // ixor r1, r1, r11
  // udiv r20, r0, r1
  // umul r20, r20, r1
  // sub r0, r0, r20
  // iadd r0, r0, r10
  // ixor DST, r0, r10

  // mov r0, LHS
  SDValue r0 = LHS;

  // mov r1, RHS
  SDValue r1 = RHS;

  // ilt r10, r0, 0
  SDValue r10 = DAG.getSetCC(DL, OVT, r0, DAG.getConstant(0, OVT), ISD::SETLT);

  // ilt r11, r1, 0
  SDValue r11 = DAG.getSetCC(DL, OVT, r1, DAG.getConstant(0, OVT), ISD::SETLT);

  // iadd r0, r0, r10
  r0 = DAG.getNode(ISD::ADD, DL, OVT, r0, r10);

  // iadd r1, r1, r11
  r1 = DAG.getNode(ISD::ADD, DL, OVT, r1, r11);

  // ixor r0, r0, r10
  r0 = DAG.getNode(ISD::XOR, DL, OVT, r0, r10);

  // ixor r1, r1, r11
  r1 = DAG.getNode(ISD::XOR, DL, OVT, r1, r11);

  // udiv r20, r0, r1
  SDValue r20 = DAG.getNode(ISD::UREM, DL, OVT, r0, r1);

  // umul r20, r20, r1
  r20 = DAG.getNode(AMDGPUISD::UMUL, DL, OVT, r20, r1);

  // sub r0, r0, r20
  r0 = DAG.getNode(ISD::SUB, DL, OVT, r0, r20);

  // iadd r0, r0, r10
  r0 = DAG.getNode(ISD::ADD, DL, OVT, r0, r10);

  // ixor DST, r0, r10
  SDValue DST = DAG.getNode(ISD::XOR, DL, OVT, r0, r10); 
  return DST;
}

SDValue
AMDGPUTargetLowering::LowerSREM64(SDValue Op, SelectionDAG &DAG) const {
  return SDValue(Op.getNode(), 0);
}
