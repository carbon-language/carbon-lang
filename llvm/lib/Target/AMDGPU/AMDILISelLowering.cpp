//===-- AMDILISelLowering.cpp - AMDIL DAG Lowering Implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// This file implements the interfaces that AMDIL uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "AMDILISelLowering.h"
#include "AMDILDevices.h"
#include "AMDILIntrinsicInfo.h"
#include "AMDILRegisterInfo.h"
#include "AMDILSubtarget.h"
#include "AMDILUtilityFunctions.h"
#include "llvm/CallingConv.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;
#define ISDBITCAST  ISD::BITCAST
#define MVTGLUE     MVT::Glue
//===----------------------------------------------------------------------===//
// Calling Convention Implementation
//===----------------------------------------------------------------------===//
#include "AMDGPUGenCallingConv.inc"

//===----------------------------------------------------------------------===//
// TargetLowering Implementation Help Functions Begin
//===----------------------------------------------------------------------===//
  static SDValue
getConversionNode(SelectionDAG &DAG, SDValue& Src, SDValue& Dst, bool asType)
{
  DebugLoc DL = Src.getDebugLoc();
  EVT svt = Src.getValueType().getScalarType();
  EVT dvt = Dst.getValueType().getScalarType();
  if (svt.isFloatingPoint() && dvt.isFloatingPoint()) {
    if (dvt.bitsGT(svt)) {
      Src = DAG.getNode(ISD::FP_EXTEND, DL, dvt, Src);
    } else if (svt.bitsLT(svt)) {
      Src = DAG.getNode(ISD::FP_ROUND, DL, dvt, Src,
          DAG.getConstant(1, MVT::i32));
    }
  } else if (svt.isInteger() && dvt.isInteger()) {
    if (!svt.bitsEq(dvt)) {
      Src = DAG.getSExtOrTrunc(Src, DL, dvt);
    }
  } else if (svt.isInteger()) {
    unsigned opcode = (asType) ? ISDBITCAST : ISD::SINT_TO_FP;
    if (!svt.bitsEq(dvt)) {
      if (dvt.getSimpleVT().SimpleTy == MVT::f32) {
        Src = DAG.getSExtOrTrunc(Src, DL, MVT::i32);
      } else if (dvt.getSimpleVT().SimpleTy == MVT::f64) {
        Src = DAG.getSExtOrTrunc(Src, DL, MVT::i64);
      } else {
        assert(0 && "We only support 32 and 64bit fp types");
      }
    }
    Src = DAG.getNode(opcode, DL, dvt, Src);
  } else if (dvt.isInteger()) {
    unsigned opcode = (asType) ? ISDBITCAST : ISD::FP_TO_SINT;
    if (svt.getSimpleVT().SimpleTy == MVT::f32) {
      Src = DAG.getNode(opcode, DL, MVT::i32, Src);
    } else if (svt.getSimpleVT().SimpleTy == MVT::f64) {
      Src = DAG.getNode(opcode, DL, MVT::i64, Src);
    } else {
      assert(0 && "We only support 32 and 64bit fp types");
    }
    Src = DAG.getSExtOrTrunc(Src, DL, dvt);
  }
  return Src;
}
// CondCCodeToCC - Convert a DAG condition code to a AMDIL CC
// condition.
  static AMDILCC::CondCodes
CondCCodeToCC(ISD::CondCode CC, const MVT::SimpleValueType& type)
{
  switch (CC) {
    default:
      {
        errs()<<"Condition Code: "<< (unsigned int)CC<<"\n";
        assert(0 && "Unknown condition code!");
      }
    case ISD::SETO:
      switch(type) {
        case MVT::f32:
          return AMDILCC::IL_CC_F_O;
        case MVT::f64:
          return AMDILCC::IL_CC_D_O;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETUO:
      switch(type) {
        case MVT::f32:
          return AMDILCC::IL_CC_F_UO;
        case MVT::f64:
          return AMDILCC::IL_CC_D_UO;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETGT:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_I_GT;
        case MVT::f32:
          return AMDILCC::IL_CC_F_GT;
        case MVT::f64:
          return AMDILCC::IL_CC_D_GT;
        case MVT::i64:
          return AMDILCC::IL_CC_L_GT;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETGE:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_I_GE;
        case MVT::f32:
          return AMDILCC::IL_CC_F_GE;
        case MVT::f64:
          return AMDILCC::IL_CC_D_GE;
        case MVT::i64:
          return AMDILCC::IL_CC_L_GE;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETLT:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_I_LT;
        case MVT::f32:
          return AMDILCC::IL_CC_F_LT;
        case MVT::f64:
          return AMDILCC::IL_CC_D_LT;
        case MVT::i64:
          return AMDILCC::IL_CC_L_LT;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETLE:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_I_LE;
        case MVT::f32:
          return AMDILCC::IL_CC_F_LE;
        case MVT::f64:
          return AMDILCC::IL_CC_D_LE;
        case MVT::i64:
          return AMDILCC::IL_CC_L_LE;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETNE:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_I_NE;
        case MVT::f32:
          return AMDILCC::IL_CC_F_NE;
        case MVT::f64:
          return AMDILCC::IL_CC_D_NE;
        case MVT::i64:
          return AMDILCC::IL_CC_L_NE;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETEQ:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_I_EQ;
        case MVT::f32:
          return AMDILCC::IL_CC_F_EQ;
        case MVT::f64:
          return AMDILCC::IL_CC_D_EQ;
        case MVT::i64:
          return AMDILCC::IL_CC_L_EQ;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETUGT:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_U_GT;
        case MVT::f32:
          return AMDILCC::IL_CC_F_UGT;
        case MVT::f64:
          return AMDILCC::IL_CC_D_UGT;
        case MVT::i64:
          return AMDILCC::IL_CC_UL_GT;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETUGE:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_U_GE;
        case MVT::f32:
          return AMDILCC::IL_CC_F_UGE;
        case MVT::f64:
          return AMDILCC::IL_CC_D_UGE;
        case MVT::i64:
          return AMDILCC::IL_CC_UL_GE;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETULT:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_U_LT;
        case MVT::f32:
          return AMDILCC::IL_CC_F_ULT;
        case MVT::f64:
          return AMDILCC::IL_CC_D_ULT;
        case MVT::i64:
          return AMDILCC::IL_CC_UL_LT;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETULE:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_U_LE;
        case MVT::f32:
          return AMDILCC::IL_CC_F_ULE;
        case MVT::f64:
          return AMDILCC::IL_CC_D_ULE;
        case MVT::i64:
          return AMDILCC::IL_CC_UL_LE;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETUNE:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_U_NE;
        case MVT::f32:
          return AMDILCC::IL_CC_F_UNE;
        case MVT::f64:
          return AMDILCC::IL_CC_D_UNE;
        case MVT::i64:
          return AMDILCC::IL_CC_UL_NE;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETUEQ:
      switch (type) {
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
          return AMDILCC::IL_CC_U_EQ;
        case MVT::f32:
          return AMDILCC::IL_CC_F_UEQ;
        case MVT::f64:
          return AMDILCC::IL_CC_D_UEQ;
        case MVT::i64:
          return AMDILCC::IL_CC_UL_EQ;
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETOGT:
      switch (type) {
        case MVT::f32:
          return AMDILCC::IL_CC_F_OGT;
        case MVT::f64:
          return AMDILCC::IL_CC_D_OGT;
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
        case MVT::i64:
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETOGE:
      switch (type) {
        case MVT::f32:
          return AMDILCC::IL_CC_F_OGE;
        case MVT::f64:
          return AMDILCC::IL_CC_D_OGE;
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
        case MVT::i64:
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETOLT:
      switch (type) {
        case MVT::f32:
          return AMDILCC::IL_CC_F_OLT;
        case MVT::f64:
          return AMDILCC::IL_CC_D_OLT;
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
        case MVT::i64:
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETOLE:
      switch (type) {
        case MVT::f32:
          return AMDILCC::IL_CC_F_OLE;
        case MVT::f64:
          return AMDILCC::IL_CC_D_OLE;
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
        case MVT::i64:
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETONE:
      switch (type) {
        case MVT::f32:
          return AMDILCC::IL_CC_F_ONE;
        case MVT::f64:
          return AMDILCC::IL_CC_D_ONE;
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
        case MVT::i64:
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
    case ISD::SETOEQ:
      switch (type) {
        case MVT::f32:
          return AMDILCC::IL_CC_F_OEQ;
        case MVT::f64:
          return AMDILCC::IL_CC_D_OEQ;
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
        case MVT::i64:
        default:
          assert(0 && "Opcode combination not generated correctly!");
          return AMDILCC::COND_ERROR;
      };
  };
}

SDValue
AMDILTargetLowering::LowerMemArgument(
    SDValue Chain,
    CallingConv::ID CallConv,
    const SmallVectorImpl<ISD::InputArg> &Ins,
    DebugLoc dl, SelectionDAG &DAG,
    const CCValAssign &VA,
    MachineFrameInfo *MFI,
    unsigned i) const
{
  // Create the nodes corresponding to a load from this parameter slot.
  ISD::ArgFlagsTy Flags = Ins[i].Flags;

  bool AlwaysUseMutable = (CallConv==CallingConv::Fast) &&
    getTargetMachine().Options.GuaranteedTailCallOpt;
  bool isImmutable = !AlwaysUseMutable && !Flags.isByVal();

  // FIXME: For now, all byval parameter objects are marked mutable. This can
  // be changed with more analysis.
  // In case of tail call optimization mark all arguments mutable. Since they
  // could be overwritten by lowering of arguments in case of a tail call.
  int FI = MFI->CreateFixedObject(VA.getValVT().getSizeInBits()/8,
      VA.getLocMemOffset(), isImmutable);
  SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());

  if (Flags.isByVal())
    return FIN;
  return DAG.getLoad(VA.getValVT(), dl, Chain, FIN,
      MachinePointerInfo::getFixedStack(FI),
      false, false, false, 0);
}
//===----------------------------------------------------------------------===//
// TargetLowering Implementation Help Functions End
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TargetLowering Class Implementation Begins
//===----------------------------------------------------------------------===//
  AMDILTargetLowering::AMDILTargetLowering(TargetMachine &TM)
: TargetLowering(TM, new TargetLoweringObjectFileELF())
{
  int types[] =
  {
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

  int IntTypes[] =
  {
    (int)MVT::i8,
    (int)MVT::i16,
    (int)MVT::i32,
    (int)MVT::i64
  };

  int FloatTypes[] =
  {
    (int)MVT::f32,
    (int)MVT::f64
  };

  int VectorTypes[] =
  {
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
  size_t numTypes = sizeof(types) / sizeof(*types);
  size_t numFloatTypes = sizeof(FloatTypes) / sizeof(*FloatTypes);
  size_t numIntTypes = sizeof(IntTypes) / sizeof(*IntTypes);
  size_t numVectorTypes = sizeof(VectorTypes) / sizeof(*VectorTypes);

  const AMDILSubtarget &STM = getTargetMachine().getSubtarget<AMDILSubtarget>();
  // These are the current register classes that are
  // supported

  for (unsigned int x  = 0; x < numTypes; ++x) {
    MVT::SimpleValueType VT = (MVT::SimpleValueType)types[x];

    //FIXME: SIGN_EXTEND_INREG is not meaningful for floating point types
    // We cannot sextinreg, expand to shifts
    setOperationAction(ISD::SIGN_EXTEND_INREG, VT, Custom);
    setOperationAction(ISD::SUBE, VT, Expand);
    setOperationAction(ISD::SUBC, VT, Expand);
    setOperationAction(ISD::ADDE, VT, Expand);
    setOperationAction(ISD::ADDC, VT, Expand);
    setOperationAction(ISD::SETCC, VT, Custom);
    setOperationAction(ISD::BRCOND, VT, Custom);
    setOperationAction(ISD::BR_CC, VT, Custom);
    setOperationAction(ISD::BR_JT, VT, Expand);
    setOperationAction(ISD::BRIND, VT, Expand);
    // TODO: Implement custom UREM/SREM routines
    setOperationAction(ISD::SREM, VT, Expand);
    setOperationAction(ISD::GlobalAddress, VT, Custom);
    setOperationAction(ISD::JumpTable, VT, Custom);
    setOperationAction(ISD::ConstantPool, VT, Custom);
    setOperationAction(ISD::SELECT, VT, Custom);
    setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, VT, Expand);
    if (VT != MVT::i64 && VT != MVT::v2i64) {
      setOperationAction(ISD::SDIV, VT, Custom);
    }
  }
  for (unsigned int x = 0; x < numFloatTypes; ++x) {
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

  for (unsigned int x = 0; x < numIntTypes; ++x) {
    MVT::SimpleValueType VT = (MVT::SimpleValueType)IntTypes[x];

    // GPU also does not have divrem function for signed or unsigned
    setOperationAction(ISD::SDIVREM, VT, Expand);

    // GPU does not have [S|U]MUL_LOHI functions as a single instruction
    setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, VT, Expand);

    // GPU doesn't have a rotl, rotr, or byteswap instruction
    setOperationAction(ISD::ROTR, VT, Expand);
    setOperationAction(ISD::BSWAP, VT, Expand);

    // GPU doesn't have any counting operators
    setOperationAction(ISD::CTPOP, VT, Expand);
    setOperationAction(ISD::CTTZ, VT, Expand);
    setOperationAction(ISD::CTLZ, VT, Expand);
  }

  for ( unsigned int ii = 0; ii < numVectorTypes; ++ii )
  {
    MVT::SimpleValueType VT = (MVT::SimpleValueType)VectorTypes[ii];

    setOperationAction(ISD::BUILD_VECTOR, VT, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Expand);
    setOperationAction(ISD::SDIVREM, VT, Expand);
    setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    // setOperationAction(ISD::VSETCC, VT, Expand);
    setOperationAction(ISD::SETCC, VT, Expand);
    setOperationAction(ISD::SELECT_CC, VT, Expand);
    setOperationAction(ISD::SELECT, VT, Expand);

  }
  if (STM.device()->isSupported(AMDILDeviceInfo::LongOps)) {
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
  }
  if (STM.device()->isSupported(AMDILDeviceInfo::DoubleOps)) {
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
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Custom);
  setOperationAction(ISD::SUBC, MVT::Other, Expand);
  setOperationAction(ISD::ADDE, MVT::Other, Expand);
  setOperationAction(ISD::ADDC, MVT::Other, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setOperationAction(ISD::BR_CC, MVT::Other, Custom);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);
  setOperationAction(ISD::SETCC, MVT::Other, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::Other, Expand);

  setOperationAction(ISD::BUILD_VECTOR, MVT::Other, Custom);
  // Use the default implementation.
  setOperationAction(ISD::VAARG             , MVT::Other, Expand);
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE         , MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE      , MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Custom);
  setOperationAction(ISD::ConstantFP        , MVT::f32    , Legal);
  setOperationAction(ISD::Constant          , MVT::i32    , Legal);
  setOperationAction(ISD::TRAP              , MVT::Other  , Legal);

  setStackPointerRegisterToSaveRestore(AMDGPU::SP);
  setSchedulingPreference(Sched::RegPressure);
  setPow2DivIsCheap(false);
  setPrefLoopAlignment(16);
  setSelectIsExpensive(true);
  setJumpIsExpensive(true);

  maxStoresPerMemcpy  = 4096;
  maxStoresPerMemmove = 4096;
  maxStoresPerMemset  = 4096;

#undef numTypes
#undef numIntTypes
#undef numVectorTypes
#undef numFloatTypes
}

const char *
AMDILTargetLowering::getTargetNodeName(unsigned Opcode) const
{
  switch (Opcode) {
    default: return 0;
    case AMDILISD::CMOVLOG:  return "AMDILISD::CMOVLOG";
    case AMDILISD::MAD:  return "AMDILISD::MAD";
    case AMDILISD::CALL:  return "AMDILISD::CALL";
    case AMDILISD::SELECT_CC: return "AMDILISD::SELECT_CC";
    case AMDILISD::UMUL: return "AMDILISD::UMUL";
    case AMDILISD::DIV_INF: return "AMDILISD::DIV_INF";
    case AMDILISD::VBUILD: return "AMDILISD::VBUILD";
    case AMDILISD::CMP: return "AMDILISD::CMP";
    case AMDILISD::IL_CC_I_LT: return "AMDILISD::IL_CC_I_LT";
    case AMDILISD::IL_CC_I_LE: return "AMDILISD::IL_CC_I_LE";
    case AMDILISD::IL_CC_I_GT: return "AMDILISD::IL_CC_I_GT";
    case AMDILISD::IL_CC_I_GE: return "AMDILISD::IL_CC_I_GE";
    case AMDILISD::IL_CC_I_EQ: return "AMDILISD::IL_CC_I_EQ";
    case AMDILISD::IL_CC_I_NE: return "AMDILISD::IL_CC_I_NE";
    case AMDILISD::RET_FLAG: return "AMDILISD::RET_FLAG";
    case AMDILISD::BRANCH_COND: return "AMDILISD::BRANCH_COND";

  };
}
bool
AMDILTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
    const CallInst &I, unsigned Intrinsic) const
{
  return false;
}

// The backend supports 32 and 64 bit floating point immediates
bool
AMDILTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const
{
  if (VT.getScalarType().getSimpleVT().SimpleTy == MVT::f32
      || VT.getScalarType().getSimpleVT().SimpleTy == MVT::f64) {
    return true;
  } else {
    return false;
  }
}

bool
AMDILTargetLowering::ShouldShrinkFPConstant(EVT VT) const
{
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
AMDILTargetLowering::computeMaskedBitsForTargetNode(
    const SDValue Op,
    APInt &KnownZero,
    APInt &KnownOne,
    const SelectionDAG &DAG,
    unsigned Depth) const
{
  APInt KnownZero2;
  APInt KnownOne2;
  KnownZero = KnownOne = APInt(KnownOne.getBitWidth(), 0); // Don't know anything
  switch (Op.getOpcode()) {
    default: break;
    case AMDILISD::SELECT_CC:
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

// This is the function that determines which calling convention should
// be used. Currently there is only one calling convention
CCAssignFn*
AMDILTargetLowering::CCAssignFnForNode(unsigned int Op) const
{
  //uint64_t CC = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  return CC_AMDIL32;
}

// LowerCallResult - Lower the result values of an ISD::CALL into the
// appropriate copies out of appropriate physical registers.  This assumes that
// Chain/InFlag are the input chain/flag to use, and that TheCall is the call
// being lowered.  The returns a SDNode with the same number of values as the
// ISD::CALL.
SDValue
AMDILTargetLowering::LowerCallResult(
    SDValue Chain,
    SDValue InFlag,
    CallingConv::ID CallConv,
    bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins,
    DebugLoc dl,
    SelectionDAG &DAG,
    SmallVectorImpl<SDValue> &InVals) const
{
  // Assign locations to each value returned by this call
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, RetCC_AMDIL32);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    EVT CopyVT = RVLocs[i].getValVT();
    if (RVLocs[i].isRegLoc()) {
      Chain = DAG.getCopyFromReg(
          Chain,
          dl,
          RVLocs[i].getLocReg(),
          CopyVT,
          InFlag
          ).getValue(1);
      SDValue Val = Chain.getValue(0);
      InFlag = Chain.getValue(2);
      InVals.push_back(Val);
    }
  }

  return Chain;

}

//===----------------------------------------------------------------------===//
//                           Other Lowering Hooks
//===----------------------------------------------------------------------===//

// Recursively assign SDNodeOrdering to any unordered nodes
// This is necessary to maintain source ordering of instructions
// under -O0 to avoid odd-looking "skipping around" issues.
  static const SDValue
Ordered( SelectionDAG &DAG, unsigned order, const SDValue New )
{
  if (order != 0 && DAG.GetOrdering( New.getNode() ) == 0) {
    DAG.AssignOrdering( New.getNode(), order );
    for (unsigned i = 0, e = New.getNumOperands(); i < e; ++i)
      Ordered( DAG, order, New.getOperand(i) );
  }
  return New;
}

#define LOWER(A) \
  case ISD:: A: \
return Ordered( DAG, DAG.GetOrdering( Op.getNode() ), Lower##A(Op, DAG) )

SDValue
AMDILTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const
{
  switch (Op.getOpcode()) {
    default:
      Op.getNode()->dump();
      assert(0 && "Custom lowering code for this"
          "instruction is not implemented yet!");
      break;
      LOWER(GlobalAddress);
      LOWER(JumpTable);
      LOWER(ConstantPool);
      LOWER(ExternalSymbol);
      LOWER(SDIV);
      LOWER(SREM);
      LOWER(BUILD_VECTOR);
      LOWER(SELECT);
      LOWER(SETCC);
      LOWER(SIGN_EXTEND_INREG);
      LOWER(DYNAMIC_STACKALLOC);
      LOWER(BRCOND);
      LOWER(BR_CC);
  }
  return Op;
}

#undef LOWER

SDValue
AMDILTargetLowering::LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const
{
  SDValue DST = Op;
  const GlobalAddressSDNode *GADN = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *G = GADN->getGlobal();
  DebugLoc DL = Op.getDebugLoc();
  const GlobalVariable *GV = dyn_cast<GlobalVariable>(G);
  if (!GV) {
    DST = DAG.getTargetGlobalAddress(GV, DL, MVT::i32);
  } else {
    if (GV->hasInitializer()) {
      const Constant *C = dyn_cast<Constant>(GV->getInitializer());
      if (const ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
        DST = DAG.getConstant(CI->getValue(), Op.getValueType());
      } else if (const ConstantFP *CF = dyn_cast<ConstantFP>(C)) {
        DST = DAG.getConstantFP(CF->getValueAPF(),
            Op.getValueType());
      } else if (dyn_cast<ConstantAggregateZero>(C)) {
        EVT VT = Op.getValueType();
        if (VT.isInteger()) {
          DST = DAG.getConstant(0, VT);
        } else {
          DST = DAG.getConstantFP(0, VT);
        }
      } else {
        assert(!"lowering this type of Global Address "
            "not implemented yet!");
        C->dump();
        DST = DAG.getTargetGlobalAddress(GV, DL, MVT::i32);
      }
    } else {
      DST = DAG.getTargetGlobalAddress(GV, DL, MVT::i32);
    }
  }
  return DST;
}

SDValue
AMDILTargetLowering::LowerJumpTable(SDValue Op, SelectionDAG &DAG) const
{
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  SDValue Result = DAG.getTargetJumpTable(JT->getIndex(), MVT::i32);
  return Result;
}
SDValue
AMDILTargetLowering::LowerConstantPool(SDValue Op, SelectionDAG &DAG) const
{
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  EVT PtrVT = Op.getValueType();
  SDValue Result;
  if (CP->isMachineConstantPoolEntry()) {
    Result = DAG.getTargetConstantPool(CP->getMachineCPVal(), PtrVT,
        CP->getAlignment(), CP->getOffset(), CP->getTargetFlags());
  } else {
    Result = DAG.getTargetConstantPool(CP->getConstVal(), PtrVT,
        CP->getAlignment(), CP->getOffset(), CP->getTargetFlags());
  }
  return Result;
}

SDValue
AMDILTargetLowering::LowerExternalSymbol(SDValue Op, SelectionDAG &DAG) const
{
  const char *Sym = cast<ExternalSymbolSDNode>(Op)->getSymbol();
  SDValue Result = DAG.getTargetExternalSymbol(Sym, MVT::i32);
  return Result;
}

/// LowerFORMAL_ARGUMENTS - transform physical registers into
/// virtual registers and generate load operations for
/// arguments places on the stack.
/// TODO: isVarArg, hasStructRet, isMemReg
  SDValue
AMDILTargetLowering::LowerFormalArguments(SDValue Chain,
    CallingConv::ID CallConv,
    bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins,
    DebugLoc dl,
    SelectionDAG &DAG,
    SmallVectorImpl<SDValue> &InVals)
const
{

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  //const Function *Fn = MF.getFunction();
  //MachineRegisterInfo &RegInfo = MF.getRegInfo();

  SmallVector<CCValAssign, 16> ArgLocs;
  CallingConv::ID CC = MF.getFunction()->getCallingConv();
  //bool hasStructRet = MF.getFunction()->hasStructRetAttr();

  CCState CCInfo(CC, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());

  // When more calling conventions are added, they need to be chosen here
  CCInfo.AnalyzeFormalArguments(Ins, CC_AMDIL32);
  SDValue StackPtr;

  //unsigned int FirstStackArgLoc = 0;

  for (unsigned int i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();
      const TargetRegisterClass *RC = getRegClassFor(
          RegVT.getSimpleVT().SimpleTy);

      unsigned int Reg = MF.addLiveIn(VA.getLocReg(), RC);
      SDValue ArgValue = DAG.getCopyFromReg(
          Chain,
          dl,
          Reg,
          RegVT);
      // If this is an 8 or 16-bit value, it is really passed
      // promoted to 32 bits.  Insert an assert[sz]ext to capture
      // this, then truncate to the right size.

      if (VA.getLocInfo() == CCValAssign::SExt) {
        ArgValue = DAG.getNode(
            ISD::AssertSext,
            dl,
            RegVT,
            ArgValue,
            DAG.getValueType(VA.getValVT()));
      } else if (VA.getLocInfo() == CCValAssign::ZExt) {
        ArgValue = DAG.getNode(
            ISD::AssertZext,
            dl,
            RegVT,
            ArgValue,
            DAG.getValueType(VA.getValVT()));
      }
      if (VA.getLocInfo() != CCValAssign::Full) {
        ArgValue = DAG.getNode(
            ISD::TRUNCATE,
            dl,
            VA.getValVT(),
            ArgValue);
      }
      // Add the value to the list of arguments
      // to be passed in registers
      InVals.push_back(ArgValue);
      if (isVarArg) {
        assert(0 && "Variable arguments are not yet supported");
        // See MipsISelLowering.cpp for ideas on how to implement
      }
    } else if(VA.isMemLoc()) {
      InVals.push_back(LowerMemArgument(Chain, CallConv, Ins,
            dl, DAG, VA, MFI, i));
    } else {
      assert(0 && "found a Value Assign that is "
          "neither a register or a memory location");
    }
  }
  /*if (hasStructRet) {
    assert(0 && "Has struct return is not yet implemented");
  // See MipsISelLowering.cpp for ideas on how to implement
  }*/

  if (isVarArg) {
    assert(0 && "Variable arguments are not yet supported");
    // See X86/PPC/CellSPU ISelLowering.cpp for ideas on how to implement
  }
  // This needs to be changed to non-zero if the return function needs
  // to pop bytes
  return Chain;
}
/// CreateCopyOfByValArgument - Make a copy of an aggregate at address specified
/// by "Src" to address "Dst" with size and alignment information specified by
/// the specific parameter attribute. The copy will be passed as a byval
/// function parameter.
static SDValue
CreateCopyOfByValArgument(SDValue Src, SDValue Dst, SDValue Chain,
    ISD::ArgFlagsTy Flags, SelectionDAG &DAG) {
  assert(0 && "MemCopy does not exist yet");
  SDValue SizeNode     = DAG.getConstant(Flags.getByValSize(), MVT::i32);

  return DAG.getMemcpy(Chain,
      Src.getDebugLoc(),
      Dst, Src, SizeNode, Flags.getByValAlign(),
      /*IsVol=*/false, /*AlwaysInline=*/true, 
      MachinePointerInfo(), MachinePointerInfo());
}

SDValue
AMDILTargetLowering::LowerMemOpCallTo(SDValue Chain,
    SDValue StackPtr, SDValue Arg,
    DebugLoc dl, SelectionDAG &DAG,
    const CCValAssign &VA,
    ISD::ArgFlagsTy Flags) const
{
  unsigned int LocMemOffset = VA.getLocMemOffset();
  SDValue PtrOff = DAG.getIntPtrConstant(LocMemOffset);
  PtrOff = DAG.getNode(ISD::ADD,
      dl,
      getPointerTy(), StackPtr, PtrOff);
  if (Flags.isByVal()) {
    PtrOff = CreateCopyOfByValArgument(Arg, PtrOff, Chain, Flags, DAG);
  } else {
    PtrOff = DAG.getStore(Chain, dl, Arg, PtrOff,
        MachinePointerInfo::getStack(LocMemOffset),
        false, false, 0);
  }
  return PtrOff;
}
/// LowerCAL - functions arguments are copied from virtual
/// regs to (physical regs)/(stack frame), CALLSEQ_START and
/// CALLSEQ_END are emitted.
/// TODO: isVarArg, isTailCall, hasStructRet
SDValue
AMDILTargetLowering::LowerCall(CallLoweringInfo &CLI,
    SmallVectorImpl<SDValue> &InVals) const

#if 0
    SDValue Chain, SDValue Callee,
    CallingConv::ID CallConv, bool isVarArg, bool doesNotRet,
    bool& isTailCall,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals,
    const SmallVectorImpl<ISD::InputArg> &Ins,
    DebugLoc dl, SelectionDAG &DAG,
#endif
{
  CLI.IsTailCall = false;
  MachineFunction& MF = CLI.DAG.getMachineFunction();
  // FIXME: DO we need to handle fast calling conventions and tail call
  // optimizations?? X86/PPC ISelLowering
  /*bool hasStructRet = (TheCall->getNumArgs())
    ? TheCall->getArgFlags(0).device()->isSRet()
    : false;*/

  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Analyze operands of the call, assigning locations to each operand
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CLI.CallConv, CLI.IsVarArg, CLI.DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *CLI.DAG.getContext());
  // Analyize the calling operands, but need to change
  // if we have more than one calling convetion
  CCInfo.AnalyzeCallOperands(CLI.Outs, CCAssignFnForNode(CLI.CallConv));

  unsigned int NumBytes = CCInfo.getNextStackOffset();
  if (CLI.IsTailCall) {
    assert(CLI.IsTailCall && "Tail Call not handled yet!");
    // See X86/PPC ISelLowering
  }

  CLI.Chain = CLI.DAG.getCALLSEQ_START(CLI.Chain,
                                   CLI.DAG.getIntPtrConstant(NumBytes, true));

  SmallVector<std::pair<unsigned int, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;
  //unsigned int FirstStacArgLoc = 0;
  //int LastArgStackLoc = 0;

  // Walk the register/memloc assignments, insert copies/loads
  for (unsigned int i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    //bool isByVal = Flags.isByVal(); // handle byval/bypointer registers
    // Arguments start after the 5 first operands of ISD::CALL
    SDValue Arg = CLI.OutVals[i];
    //Promote the value if needed
    switch(VA.getLocInfo()) {
      default: assert(0 && "Unknown loc info!");
      case CCValAssign::Full:
               break;
      case CCValAssign::SExt:
               Arg = CLI.DAG.getNode(ISD::SIGN_EXTEND,
                   CLI.DL,
                   VA.getLocVT(), Arg);
               break;
      case CCValAssign::ZExt:
               Arg = CLI.DAG.getNode(ISD::ZERO_EXTEND,
                   CLI.DL,
                   VA.getLocVT(), Arg);
               break;
      case CCValAssign::AExt:
               Arg = CLI.DAG.getNode(ISD::ANY_EXTEND,
                   CLI.DL,
                   VA.getLocVT(), Arg);
               break;
    }

    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else if (VA.isMemLoc()) {
      // Create the frame index object for this incoming parameter
      int FI = MFI->CreateFixedObject(VA.getValVT().getSizeInBits()/8,
          VA.getLocMemOffset(), true);
      SDValue PtrOff = CLI.DAG.getFrameIndex(FI,getPointerTy());

      // emit ISD::STORE whichs stores the
      // parameter value to a stack Location
      MemOpChains.push_back(CLI.DAG.getStore(CLI.Chain, CLI.DL, Arg, PtrOff,
            MachinePointerInfo::getFixedStack(FI),
            false, false, 0));
    } else {
      assert(0 && "Not a Reg/Mem Loc, major error!");
    }
  }
  if (!MemOpChains.empty()) {
    CLI.Chain = CLI.DAG.getNode(ISD::TokenFactor,
        CLI.DL,
        MVT::Other,
        &MemOpChains[0],
        MemOpChains.size());
  }
  SDValue InFlag;
  if (!CLI.IsTailCall) {
    for (unsigned int i = 0, e = RegsToPass.size(); i != e; ++i) {
      CLI.Chain = CLI.DAG.getCopyToReg(CLI.Chain,
          CLI.DL,
          RegsToPass[i].first,
          RegsToPass[i].second,
          InFlag);
      InFlag = CLI.Chain.getValue(1);
    }
  }

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common,
  // every direct call is) turn it into a TargetGlobalAddress/
  // TargetExternalSymbol
  // node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(CLI.Callee))  {
    CLI.Callee = CLI.DAG.getTargetGlobalAddress(G->getGlobal(), CLI.DL, getPointerTy());
  }
  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(CLI.Callee)) {
    CLI.Callee = CLI.DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy());
  }
  else if (CLI.IsTailCall) {
    assert(0 && "Tail calls are not handled yet");
    // see X86 ISelLowering for ideas on implementation: 1708
  }

  SDVTList NodeTys = CLI.DAG.getVTList(MVT::Other, MVTGLUE);
  SmallVector<SDValue, 8> Ops;

  if (CLI.IsTailCall) {
    assert(0 && "Tail calls are not handled yet");
    // see X86 ISelLowering for ideas on implementation: 1721
  }
  // If this is a direct call, pass the chain and the callee
  if (CLI.Callee.getNode()) {
    Ops.push_back(CLI.Chain);
    Ops.push_back(CLI.Callee);
  }

  if (CLI.IsTailCall) {
    assert(0 && "Tail calls are not handled yet");
    // see X86 ISelLowering for ideas on implementation: 1739
  }

  // Add argument registers to the end of the list so that they are known
  // live into the call
  for (unsigned int i = 0, e = RegsToPass.size(); i != e; ++i) {
    Ops.push_back(CLI.DAG.getRegister(
          RegsToPass[i].first,
          RegsToPass[i].second.getValueType()));
  }
  if (InFlag.getNode()) {
    Ops.push_back(InFlag);
  }

  // Emit Tail Call
  if (CLI.IsTailCall) {
    assert(0 && "Tail calls are not handled yet");
    // see X86 ISelLowering for ideas on implementation: 1762
  }

  CLI.Chain = CLI.DAG.getNode(AMDILISD::CALL,
      CLI.DL,
      NodeTys, &Ops[0], Ops.size());
  InFlag = CLI.Chain.getValue(1);

  // Create the CALLSEQ_END node
  CLI.Chain = CLI.DAG.getCALLSEQ_END(
      CLI.Chain,
      CLI.DAG.getIntPtrConstant(NumBytes, true),
      CLI.DAG.getIntPtrConstant(0, true),
      InFlag);
  InFlag = CLI.Chain.getValue(1);
  // Handle result values, copying them out of physregs into vregs that
  // we return
  return LowerCallResult(CLI.Chain, InFlag, CLI.CallConv, CLI.IsVarArg, CLI.Ins, CLI.DL, CLI.DAG,
      InVals);
}

SDValue
AMDILTargetLowering::LowerSDIV(SDValue Op, SelectionDAG &DAG) const
{
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
AMDILTargetLowering::LowerSREM(SDValue Op, SelectionDAG &DAG) const
{
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

SDValue
AMDILTargetLowering::LowerBUILD_VECTOR( SDValue Op, SelectionDAG &DAG ) const
{
  EVT VT = Op.getValueType();
  SDValue Nodes1;
  SDValue second;
  SDValue third;
  SDValue fourth;
  DebugLoc DL = Op.getDebugLoc();
  Nodes1 = DAG.getNode(AMDILISD::VBUILD,
      DL,
      VT, Op.getOperand(0));
#if 0
  bool allEqual = true;
  for (unsigned x = 1, y = Op.getNumOperands(); x < y; ++x) {
    if (Op.getOperand(0) != Op.getOperand(x)) {
      allEqual = false;
      break;
    }
  }
  if (allEqual) {
    return Nodes1;
  }
#endif
  switch(Op.getNumOperands()) {
    default:
    case 1:
      break;
    case 4:
      fourth = Op.getOperand(3);
      if (fourth.getOpcode() != ISD::UNDEF) {
        Nodes1 = DAG.getNode(
            ISD::INSERT_VECTOR_ELT,
            DL,
            Op.getValueType(),
            Nodes1,
            fourth,
            DAG.getConstant(7, MVT::i32));
      }
    case 3:
      third = Op.getOperand(2);
      if (third.getOpcode() != ISD::UNDEF) {
        Nodes1 = DAG.getNode(
            ISD::INSERT_VECTOR_ELT,
            DL,
            Op.getValueType(),
            Nodes1,
            third,
            DAG.getConstant(6, MVT::i32));
      }
    case 2:
      second = Op.getOperand(1);
      if (second.getOpcode() != ISD::UNDEF) {
        Nodes1 = DAG.getNode(
            ISD::INSERT_VECTOR_ELT,
            DL,
            Op.getValueType(),
            Nodes1,
            second,
            DAG.getConstant(5, MVT::i32));
      }
      break;
  };
  return Nodes1;
}

SDValue
AMDILTargetLowering::LowerSELECT(SDValue Op, SelectionDAG &DAG) const
{
  SDValue Cond = Op.getOperand(0);
  SDValue LHS = Op.getOperand(1);
  SDValue RHS = Op.getOperand(2);
  DebugLoc DL = Op.getDebugLoc();
  Cond = getConversionNode(DAG, Cond, Op, true);
  Cond = DAG.getNode(AMDILISD::CMOVLOG,
      DL,
      Op.getValueType(), Cond, LHS, RHS);
  return Cond;
}
SDValue
AMDILTargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const
{
  SDValue Cond;
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue CC  = Op.getOperand(2);
  DebugLoc DL = Op.getDebugLoc();
  ISD::CondCode SetCCOpcode = cast<CondCodeSDNode>(CC)->get();
  unsigned int AMDILCC = CondCCodeToCC(
      SetCCOpcode,
      LHS.getValueType().getSimpleVT().SimpleTy);
  assert((AMDILCC != AMDILCC::COND_ERROR) && "Invalid SetCC!");
  Cond = DAG.getNode(
      ISD::SELECT_CC,
      Op.getDebugLoc(),
      LHS.getValueType(),
      LHS, RHS,
      DAG.getConstant(-1, MVT::i32),
      DAG.getConstant(0, MVT::i32),
      CC);
  Cond = getConversionNode(DAG, Cond, Op, true);
  Cond = DAG.getNode(
      ISD::AND,
      DL,
      Cond.getValueType(),
      DAG.getConstant(1, Cond.getValueType()),
      Cond);
  return Cond;
}

SDValue
AMDILTargetLowering::LowerSIGN_EXTEND_INREG(SDValue Op, SelectionDAG &DAG) const
{
  SDValue Data = Op.getOperand(0);
  VTSDNode *BaseType = cast<VTSDNode>(Op.getOperand(1));
  DebugLoc DL = Op.getDebugLoc();
  EVT DVT = Data.getValueType();
  EVT BVT = BaseType->getVT();
  unsigned baseBits = BVT.getScalarType().getSizeInBits();
  unsigned srcBits = DVT.isSimple() ? DVT.getScalarType().getSizeInBits() : 1;
  unsigned shiftBits = srcBits - baseBits;
  if (srcBits < 32) {
    // If the op is less than 32 bits, then it needs to extend to 32bits
    // so it can properly keep the upper bits valid.
    EVT IVT = genIntType(32, DVT.isVector() ? DVT.getVectorNumElements() : 1);
    Data = DAG.getNode(ISD::ZERO_EXTEND, DL, IVT, Data);
    shiftBits = 32 - baseBits;
    DVT = IVT;
  }
  SDValue Shift = DAG.getConstant(shiftBits, DVT);
  // Shift left by 'Shift' bits.
  Data = DAG.getNode(ISD::SHL, DL, DVT, Data, Shift);
  // Signed shift Right by 'Shift' bits.
  Data = DAG.getNode(ISD::SRA, DL, DVT, Data, Shift);
  if (srcBits < 32) {
    // Once the sign extension is done, the op needs to be converted to
    // its original type.
    Data = DAG.getSExtOrTrunc(Data, DL, Op.getOperand(0).getValueType());
  }
  return Data;
}
EVT
AMDILTargetLowering::genIntType(uint32_t size, uint32_t numEle) const
{
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
AMDILTargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
    SelectionDAG &DAG) const
{
  SDValue Chain = Op.getOperand(0);
  SDValue Size = Op.getOperand(1);
  unsigned int SPReg = AMDGPU::SP;
  DebugLoc DL = Op.getDebugLoc();
  SDValue SP = DAG.getCopyFromReg(Chain,
      DL,
      SPReg, MVT::i32);
  SDValue NewSP = DAG.getNode(ISD::ADD,
      DL,
      MVT::i32, SP, Size);
  Chain = DAG.getCopyToReg(SP.getValue(1),
      DL,
      SPReg, NewSP);
  SDValue Ops[2] = {NewSP, Chain};
  Chain = DAG.getMergeValues(Ops, 2 ,DL);
  return Chain;
}
SDValue
AMDILTargetLowering::LowerBRCOND(SDValue Op, SelectionDAG &DAG) const
{
  SDValue Chain = Op.getOperand(0);
  SDValue Cond  = Op.getOperand(1);
  SDValue Jump  = Op.getOperand(2);
  SDValue Result;
  Result = DAG.getNode(
      AMDILISD::BRANCH_COND,
      Op.getDebugLoc(),
      Op.getValueType(),
      Chain, Jump, Cond);
  return Result;
}

SDValue
AMDILTargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const
{
  SDValue Chain = Op.getOperand(0);
  SDValue CC = Op.getOperand(1);
  SDValue LHS   = Op.getOperand(2);
  SDValue RHS   = Op.getOperand(3);
  SDValue JumpT  = Op.getOperand(4);
  SDValue CmpValue;
  SDValue Result;
  CmpValue = DAG.getNode(
      ISD::SELECT_CC,
      Op.getDebugLoc(),
      LHS.getValueType(),
      LHS, RHS,
      DAG.getConstant(-1, MVT::i32),
      DAG.getConstant(0, MVT::i32),
      CC);
  Result = DAG.getNode(
      AMDILISD::BRANCH_COND,
      CmpValue.getDebugLoc(),
      MVT::Other, Chain,
      JumpT, CmpValue);
  return Result;
}

// LowerRET - Lower an ISD::RET node.
SDValue
AMDILTargetLowering::LowerReturn(SDValue Chain,
    CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals,
    DebugLoc dl, SelectionDAG &DAG)
const
{
  //MachineFunction& MF = DAG.getMachineFunction();
  // CCValAssign - represent the assignment of the return value
  // to a location
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slot
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());

  // Analyze return values of ISD::RET
  CCInfo.AnalyzeReturn(Outs, RetCC_AMDIL32);
  // If this is the first return lowered for this function, add
  // the regs to the liveout set for the function
  MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
  for (unsigned int i = 0, e = RVLocs.size(); i != e; ++i) {
    if (RVLocs[i].isRegLoc() && !MRI.isLiveOut(RVLocs[i].getLocReg())) {
      MRI.addLiveOut(RVLocs[i].getLocReg());
    }
  }
  // FIXME: implement this when tail call is implemented
  // Chain = GetPossiblePreceedingTailCall(Chain, AMDILISD::TAILCALL);
  // both x86 and ppc implement this in ISelLowering

  // Regular return here
  SDValue Flag;
  SmallVector<SDValue, 6> RetOps;
  RetOps.push_back(Chain);
  RetOps.push_back(DAG.getConstant(0/*getBytesToPopOnReturn()*/, MVT::i32));
  for (unsigned int i = 0, e = RVLocs.size(); i != e; ++i) {
    CCValAssign &VA = RVLocs[i];
    SDValue ValToCopy = OutVals[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    // ISD::Ret => ret chain, (regnum1, val1), ...
    // So i * 2 + 1 index only the regnums
    Chain = DAG.getCopyToReg(Chain,
        dl,
        VA.getLocReg(),
        ValToCopy,
        Flag);
    // guarantee that all emitted copies are stuck together
    // avoiding something bad
    Flag = Chain.getValue(1);
  }
  /*if (MF.getFunction()->hasStructRetAttr()) {
    assert(0 && "Struct returns are not yet implemented!");
  // Both MIPS and X86 have this
  }*/
  RetOps[0] = Chain;
  if (Flag.getNode())
    RetOps.push_back(Flag);

  Flag = DAG.getNode(AMDILISD::RET_FLAG,
      dl,
      MVT::Other, &RetOps[0], RetOps.size());
  return Flag;
}

unsigned int
AMDILTargetLowering::getFunctionAlignment(const Function *) const
{
  return 0;
}

SDValue
AMDILTargetLowering::LowerSDIV24(SDValue Op, SelectionDAG &DAG) const
{
  DebugLoc DL = Op.getDebugLoc();
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
  SDValue fq = DAG.getNode(AMDILISD::DIV_INF, DL, FLTTY, fa, fb);

  // fq = trunc(fq);
  fq = DAG.getNode(ISD::FTRUNC, DL, FLTTY, fq);

  // float fqneg = -fq;
  SDValue fqneg = DAG.getNode(ISD::FNEG, DL, FLTTY, fq);

  // float fr = mad(fqneg, fb, fa);
  SDValue fr = DAG.getNode(AMDILISD::MAD, DL, FLTTY, fqneg, fb, fa);

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
  jq = DAG.getNode(AMDILISD::CMOVLOG, DL, OVT, cv, jq, 
      DAG.getConstant(0, OVT));
  // dst = iq + jq;
  iq = DAG.getSExtOrTrunc(iq, DL, OVT);
  iq = DAG.getNode(ISD::ADD, DL, OVT, iq, jq);
  return iq;
}

SDValue
AMDILTargetLowering::LowerSDIV32(SDValue Op, SelectionDAG &DAG) const
{
  DebugLoc DL = Op.getDebugLoc();
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
AMDILTargetLowering::LowerSDIV64(SDValue Op, SelectionDAG &DAG) const
{
  return SDValue(Op.getNode(), 0);
}

SDValue
AMDILTargetLowering::LowerSREM8(SDValue Op, SelectionDAG &DAG) const
{
  DebugLoc DL = Op.getDebugLoc();
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
AMDILTargetLowering::LowerSREM16(SDValue Op, SelectionDAG &DAG) const
{
  DebugLoc DL = Op.getDebugLoc();
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
AMDILTargetLowering::LowerSREM32(SDValue Op, SelectionDAG &DAG) const
{
  DebugLoc DL = Op.getDebugLoc();
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
  SDValue r10 = DAG.getNode(AMDILISD::CMP, DL, OVT,
      DAG.getConstant(CondCCodeToCC(ISD::SETLT, MVT::i32), MVT::i32),
      r0, DAG.getConstant(0, OVT));

  // ilt r11, r1, 0
  SDValue r11 = DAG.getNode(AMDILISD::CMP, DL, OVT, 
      DAG.getConstant(CondCCodeToCC(ISD::SETLT, MVT::i32), MVT::i32),
      r1, DAG.getConstant(0, OVT));

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
  r20 = DAG.getNode(AMDILISD::UMUL, DL, OVT, r20, r1);

  // sub r0, r0, r20
  r0 = DAG.getNode(ISD::SUB, DL, OVT, r0, r20);

  // iadd r0, r0, r10
  r0 = DAG.getNode(ISD::ADD, DL, OVT, r0, r10);

  // ixor DST, r0, r10
  SDValue DST = DAG.getNode(ISD::XOR, DL, OVT, r0, r10); 
  return DST;
}

SDValue
AMDILTargetLowering::LowerSREM64(SDValue Op, SelectionDAG &DAG) const
{
  return SDValue(Op.getNode(), 0);
}
