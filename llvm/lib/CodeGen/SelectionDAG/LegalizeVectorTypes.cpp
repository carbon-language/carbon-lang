//===------- LegalizeVectorTypes.cpp - Legalization of vector types -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file performs vector type splitting and scalarization for LegalizeTypes.
// Scalarization is the act of changing a computation in an illegal one-element
// vector type to be a computation in its scalar element type.  For example,
// implementing <1 x f32> arithmetic in a scalar f32 register.  This is needed
// as a base case when scalarizing vector arithmetic like <4 x f32>, which
// eventually decomposes to scalars if the target doesn't support v4f32 or v2f32
// types.
// Splitting is the act of changing a computation in an invalid vector type to
// be a computation in multiple vectors of a smaller type.  For example,
// implementing <128 x f32> operations in terms of two <64 x f32> operations.
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  Result Vector Scalarization: <1 x ty> -> ty.
//===----------------------------------------------------------------------===//

void DAGTypeLegalizer::ScalarizeVectorResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Scalarize node result " << ResNo << ": "; N->dump(&DAG);
        cerr << "\n");
  SDValue R = SDValue();

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "ScalarizeVectorResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to scalarize the result of this operator!");
    abort();

  case ISD::BIT_CONVERT:    R = ScalarizeVecRes_BIT_CONVERT(N); break;
  case ISD::BUILD_VECTOR:   R = N->getOperand(0); break;
  case ISD::FPOWI:          R = ScalarizeVecRes_FPOWI(N); break;
  case ISD::INSERT_VECTOR_ELT: R = ScalarizeVecRes_INSERT_VECTOR_ELT(N); break;
  case ISD::LOAD:           R = ScalarizeVecRes_LOAD(cast<LoadSDNode>(N));break;
  case ISD::SELECT:         R = ScalarizeVecRes_SELECT(N); break;
  case ISD::UNDEF:          R = ScalarizeVecRes_UNDEF(N); break;
  case ISD::VECTOR_SHUFFLE: R = ScalarizeVecRes_VECTOR_SHUFFLE(N); break;
  case ISD::VSETCC:         R = ScalarizeVecRes_VSETCC(N); break;

  case ISD::CTLZ:
  case ISD::CTPOP:
  case ISD::CTTZ:
  case ISD::FABS:
  case ISD::FCOS:
  case ISD::FNEG:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::FSIN:
  case ISD::FSQRT:
  case ISD::FTRUNC:
  case ISD::FFLOOR:
  case ISD::FCEIL:
  case ISD::FRINT:
  case ISD::FNEARBYINT:
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: R = ScalarizeVecRes_UnaryOp(N); break;

  case ISD::ADD:
  case ISD::AND:
  case ISD::FADD:
  case ISD::FDIV:
  case ISD::FMUL:
  case ISD::FPOW:
  case ISD::FREM:
  case ISD::FSUB:
  case ISD::MUL:
  case ISD::OR:
  case ISD::SDIV:
  case ISD::SREM:
  case ISD::SUB:
  case ISD::UDIV:
  case ISD::UREM:
  case ISD::XOR:  R = ScalarizeVecRes_BinOp(N); break;
  }

  // If R is null, the sub-method took care of registering the result.
  if (R.getNode())
    SetScalarizedVector(SDValue(N, ResNo), R);
}

SDValue DAGTypeLegalizer::ScalarizeVecRes_BinOp(SDNode *N) {
  SDValue LHS = GetScalarizedVector(N->getOperand(0));
  SDValue RHS = GetScalarizedVector(N->getOperand(1));
  return DAG.getNode(N->getOpcode(), LHS.getValueType(), LHS, RHS);
}

SDValue DAGTypeLegalizer::ScalarizeVecRes_BIT_CONVERT(SDNode *N) {
  MVT NewVT = N->getValueType(0).getVectorElementType();
  return DAG.getNode(ISD::BIT_CONVERT, NewVT, N->getOperand(0));
}

SDValue DAGTypeLegalizer::ScalarizeVecRes_FPOWI(SDNode *N) {
  SDValue Op = GetScalarizedVector(N->getOperand(0));
  return DAG.getNode(ISD::FPOWI, Op.getValueType(), Op, N->getOperand(1));
}

SDValue DAGTypeLegalizer::ScalarizeVecRes_INSERT_VECTOR_ELT(SDNode *N) {
  // The value to insert may have a wider type than the vector element type,
  // so be sure to truncate it to the element type if necessary.
  SDValue Op = N->getOperand(1);
  MVT EltVT = N->getValueType(0).getVectorElementType();
  if (Op.getValueType() != EltVT)
    // FIXME: Can this happen for floating point types?
    Op = DAG.getNode(ISD::TRUNCATE, EltVT, Op);
  return Op;
}

SDValue DAGTypeLegalizer::ScalarizeVecRes_LOAD(LoadSDNode *N) {
  assert(N->isUnindexed() && "Indexed vector load?");

  SDValue Result = DAG.getLoad(ISD::UNINDEXED, N->getExtensionType(),
                               N->getValueType(0).getVectorElementType(),
                               N->getChain(), N->getBasePtr(),
                               DAG.getNode(ISD::UNDEF,
                                           N->getBasePtr().getValueType()),
                               N->getSrcValue(), N->getSrcValueOffset(),
                               N->getMemoryVT().getVectorElementType(),
                               N->isVolatile(), N->getAlignment());

  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDValue(N, 1), Result.getValue(1));
  return Result;
}

SDValue DAGTypeLegalizer::ScalarizeVecRes_UnaryOp(SDNode *N) {
  // Get the dest type - it doesn't always match the input type, e.g. int_to_fp.
  MVT DestVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Op = GetScalarizedVector(N->getOperand(0));
  return DAG.getNode(N->getOpcode(), DestVT, Op);
}

SDValue DAGTypeLegalizer::ScalarizeVecRes_UNDEF(SDNode *N) {
  return DAG.getNode(ISD::UNDEF, N->getValueType(0).getVectorElementType());
}

SDValue DAGTypeLegalizer::ScalarizeVecRes_SELECT(SDNode *N) {
  SDValue LHS = GetScalarizedVector(N->getOperand(1));
  return DAG.getNode(ISD::SELECT, LHS.getValueType(), N->getOperand(0), LHS,
                     GetScalarizedVector(N->getOperand(2)));
}

SDValue DAGTypeLegalizer::ScalarizeVecRes_VECTOR_SHUFFLE(SDNode *N) {
  // Figure out if the scalar is the LHS or RHS and return it.
  SDValue EltNum = N->getOperand(2).getOperand(0);
  unsigned Op = cast<ConstantSDNode>(EltNum)->getValue() != 0;
  return GetScalarizedVector(N->getOperand(Op));
}

SDValue DAGTypeLegalizer::ScalarizeVecRes_VSETCC(SDNode *N) {
  MVT NewVT = N->getValueType(0).getVectorElementType();
  SDValue LHS = GetScalarizedVector(N->getOperand(0));
  SDValue RHS = GetScalarizedVector(N->getOperand(1));
  LHS = DAG.getNode(ISD::SETCC, TLI.getSetCCResultType(LHS), LHS, RHS,
                    N->getOperand(2));
  return
    DAG.getNode(ISD::SELECT, NewVT, LHS,
                DAG.getConstant(APInt::getAllOnesValue(NewVT.getSizeInBits()),
                                NewVT),
                DAG.getConstant(0ULL, NewVT));
}


//===----------------------------------------------------------------------===//
//  Operand Vector Scalarization <1 x ty> -> ty.
//===----------------------------------------------------------------------===//

bool DAGTypeLegalizer::ScalarizeVectorOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Scalarize node operand " << OpNo << ": "; N->dump(&DAG);
        cerr << "\n");
  SDValue Res = SDValue();

  if (Res.getNode() == 0) {
    switch (N->getOpcode()) {
    default:
#ifndef NDEBUG
      cerr << "ScalarizeVectorOperand Op #" << OpNo << ": ";
      N->dump(&DAG); cerr << "\n";
#endif
      assert(0 && "Do not know how to scalarize this operator's operand!");
      abort();

    case ISD::BIT_CONVERT:
      Res = ScalarizeVecOp_BIT_CONVERT(N); break;

    case ISD::EXTRACT_VECTOR_ELT:
      Res = ScalarizeVecOp_EXTRACT_VECTOR_ELT(N); break;

    case ISD::STORE:
      Res = ScalarizeVecOp_STORE(cast<StoreSDNode>(N), OpNo); break;
    }
  }

  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.getNode()) return false;

  // If the result is N, the sub-method updated N in place.  Check to see if any
  // operands are new, and if so, mark them.
  if (Res.getNode() == N) {
    // Mark N as new and remark N and its operands.  This allows us to correctly
    // revisit N if it needs another step of promotion and allows us to visit
    // any new operands to N.
    ReanalyzeNode(N);
    return true;
  }

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");

  ReplaceValueWith(SDValue(N, 0), Res);
  return false;
}

/// ScalarizeVecOp_BIT_CONVERT - If the value to convert is a vector that needs
/// to be scalarized, it must be <1 x ty>.  Convert the element instead.
SDValue DAGTypeLegalizer::ScalarizeVecOp_BIT_CONVERT(SDNode *N) {
  SDValue Elt = GetScalarizedVector(N->getOperand(0));
  return DAG.getNode(ISD::BIT_CONVERT, N->getValueType(0), Elt);
}

/// ScalarizeVecOp_EXTRACT_VECTOR_ELT - If the input is a vector that needs to
/// be scalarized, it must be <1 x ty>, so just return the element, ignoring the
/// index.
SDValue DAGTypeLegalizer::ScalarizeVecOp_EXTRACT_VECTOR_ELT(SDNode *N) {
  return GetScalarizedVector(N->getOperand(0));
}

/// ScalarizeVecOp_STORE - If the value to store is a vector that needs to be
/// scalarized, it must be <1 x ty>.  Just store the element.
SDValue DAGTypeLegalizer::ScalarizeVecOp_STORE(StoreSDNode *N, unsigned OpNo){
  assert(N->isUnindexed() && "Indexed store of one-element vector?");
  assert(OpNo == 1 && "Do not know how to scalarize this operand!");

  if (N->isTruncatingStore())
    return DAG.getTruncStore(N->getChain(),
                             GetScalarizedVector(N->getOperand(1)),
                             N->getBasePtr(),
                             N->getSrcValue(), N->getSrcValueOffset(),
                             N->getMemoryVT().getVectorElementType(),
                             N->isVolatile(), N->getAlignment());

  return DAG.getStore(N->getChain(), GetScalarizedVector(N->getOperand(1)),
                      N->getBasePtr(), N->getSrcValue(), N->getSrcValueOffset(),
                      N->isVolatile(), N->getAlignment());
}


//===----------------------------------------------------------------------===//
//  Result Vector Splitting
//===----------------------------------------------------------------------===//

/// SplitVectorResult - This method is called when the specified result of the
/// specified node is found to need vector splitting.  At this point, the node
/// may also have invalid operands or may have other results that need
/// legalization, we just know that (at least) one result needs vector
/// splitting.
void DAGTypeLegalizer::SplitVectorResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Split node result: "; N->dump(&DAG); cerr << "\n");
  SDValue Lo, Hi;

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "SplitVectorResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to split the result of this operator!");
    abort();

  case ISD::MERGE_VALUES: SplitRes_MERGE_VALUES(N, Lo, Hi); break;
  case ISD::SELECT:       SplitRes_SELECT(N, Lo, Hi); break;
  case ISD::SELECT_CC:    SplitRes_SELECT_CC(N, Lo, Hi); break;
  case ISD::UNDEF:        SplitRes_UNDEF(N, Lo, Hi); break;

  case ISD::BIT_CONVERT:    SplitVecRes_BIT_CONVERT(N, Lo, Hi); break;
  case ISD::BUILD_VECTOR:   SplitVecRes_BUILD_VECTOR(N, Lo, Hi); break;
  case ISD::CONCAT_VECTORS: SplitVecRes_CONCAT_VECTORS(N, Lo, Hi); break;
  case ISD::FPOWI:          SplitVecRes_FPOWI(N, Lo, Hi); break;
  case ISD::INSERT_VECTOR_ELT: SplitVecRes_INSERT_VECTOR_ELT(N, Lo, Hi); break;
  case ISD::LOAD:           SplitVecRes_LOAD(cast<LoadSDNode>(N), Lo, Hi);break;
  case ISD::VECTOR_SHUFFLE: SplitVecRes_VECTOR_SHUFFLE(N, Lo, Hi); break;
  case ISD::VSETCC:         SplitVecRes_VSETCC(N, Lo, Hi); break;

  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP:
  case ISD::FNEG:
  case ISD::FABS:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::FTRUNC:
  case ISD::FFLOOR:
  case ISD::FCEIL:
  case ISD::FRINT:
  case ISD::FNEARBYINT:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: SplitVecRes_UnaryOp(N, Lo, Hi); break;

  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::FDIV:
  case ISD::FPOW:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::UREM:
  case ISD::SREM:
  case ISD::FREM: SplitVecRes_BinOp(N, Lo, Hi); break;
  }

  // If Lo/Hi is null, the sub-method took care of registering results etc.
  if (Lo.getNode())
    SetSplitVector(SDValue(N, ResNo), Lo, Hi);
}

void DAGTypeLegalizer::SplitVecRes_BinOp(SDNode *N, SDValue &Lo,
                                         SDValue &Hi) {
  SDValue LHSLo, LHSHi;
  GetSplitVector(N->getOperand(0), LHSLo, LHSHi);
  SDValue RHSLo, RHSHi;
  GetSplitVector(N->getOperand(1), RHSLo, RHSHi);

  Lo = DAG.getNode(N->getOpcode(), LHSLo.getValueType(), LHSLo, RHSLo);
  Hi = DAG.getNode(N->getOpcode(), LHSHi.getValueType(), LHSHi, RHSHi);
}

void DAGTypeLegalizer::SplitVecRes_BIT_CONVERT(SDNode *N, SDValue &Lo,
                                               SDValue &Hi) {
  // We know the result is a vector.  The input may be either a vector or a
  // scalar value.
  MVT LoVT, HiVT;
  GetSplitDestVTs(N->getValueType(0), LoVT, HiVT);

  SDValue InOp = N->getOperand(0);
  MVT InVT = InOp.getValueType();

  // Handle some special cases efficiently.
  switch (getTypeAction(InVT)) {
  default:
    assert(false && "Unknown type action!");
  case Legal:
  case PromoteInteger:
  case SoftenFloat:
  case ScalarizeVector:
    break;
  case ExpandInteger:
  case ExpandFloat:
    // A scalar to vector conversion, where the scalar needs expansion.
    // If the vector is being split in two then we can just convert the
    // expanded pieces.
    if (LoVT == HiVT) {
      GetExpandedOp(InOp, Lo, Hi);
      if (TLI.isBigEndian())
        std::swap(Lo, Hi);
      Lo = DAG.getNode(ISD::BIT_CONVERT, LoVT, Lo);
      Hi = DAG.getNode(ISD::BIT_CONVERT, HiVT, Hi);
      return;
    }
    break;
  case SplitVector:
    // If the input is a vector that needs to be split, convert each split
    // piece of the input now.
    GetSplitVector(InOp, Lo, Hi);
    Lo = DAG.getNode(ISD::BIT_CONVERT, LoVT, Lo);
    Hi = DAG.getNode(ISD::BIT_CONVERT, HiVT, Hi);
    return;
  }

  // In the general case, convert the input to an integer and split it by hand.
  MVT LoIntVT = MVT::getIntegerVT(LoVT.getSizeInBits());
  MVT HiIntVT = MVT::getIntegerVT(HiVT.getSizeInBits());
  if (TLI.isBigEndian())
    std::swap(LoIntVT, HiIntVT);

  SplitInteger(BitConvertToInteger(InOp), LoIntVT, HiIntVT, Lo, Hi);

  if (TLI.isBigEndian())
    std::swap(Lo, Hi);
  Lo = DAG.getNode(ISD::BIT_CONVERT, LoVT, Lo);
  Hi = DAG.getNode(ISD::BIT_CONVERT, HiVT, Hi);
}

void DAGTypeLegalizer::SplitVecRes_BUILD_VECTOR(SDNode *N, SDValue &Lo,
                                                SDValue &Hi) {
  MVT LoVT, HiVT;
  GetSplitDestVTs(N->getValueType(0), LoVT, HiVT);
  unsigned LoNumElts = LoVT.getVectorNumElements();
  SmallVector<SDValue, 8> LoOps(N->op_begin(), N->op_begin()+LoNumElts);
  Lo = DAG.getNode(ISD::BUILD_VECTOR, LoVT, &LoOps[0], LoOps.size());

  SmallVector<SDValue, 8> HiOps(N->op_begin()+LoNumElts, N->op_end());
  Hi = DAG.getNode(ISD::BUILD_VECTOR, HiVT, &HiOps[0], HiOps.size());
}

void DAGTypeLegalizer::SplitVecRes_CONCAT_VECTORS(SDNode *N, SDValue &Lo,
                                                  SDValue &Hi) {
  assert(!(N->getNumOperands() & 1) && "Unsupported CONCAT_VECTORS");
  unsigned NumSubvectors = N->getNumOperands() / 2;
  if (NumSubvectors == 1) {
    Lo = N->getOperand(0);
    Hi = N->getOperand(1);
    return;
  }

  MVT LoVT, HiVT;
  GetSplitDestVTs(N->getValueType(0), LoVT, HiVT);

  SmallVector<SDValue, 8> LoOps(N->op_begin(), N->op_begin()+NumSubvectors);
  Lo = DAG.getNode(ISD::CONCAT_VECTORS, LoVT, &LoOps[0], LoOps.size());

  SmallVector<SDValue, 8> HiOps(N->op_begin()+NumSubvectors, N->op_end());
  Hi = DAG.getNode(ISD::CONCAT_VECTORS, HiVT, &HiOps[0], HiOps.size());
}

void DAGTypeLegalizer::SplitVecRes_FPOWI(SDNode *N, SDValue &Lo,
                                         SDValue &Hi) {
  GetSplitVector(N->getOperand(0), Lo, Hi);
  Lo = DAG.getNode(ISD::FPOWI, Lo.getValueType(), Lo, N->getOperand(1));
  Hi = DAG.getNode(ISD::FPOWI, Hi.getValueType(), Hi, N->getOperand(1));
}

void DAGTypeLegalizer::SplitVecRes_INSERT_VECTOR_ELT(SDNode *N, SDValue &Lo,
                                                     SDValue &Hi) {
  SDValue Vec = N->getOperand(0);
  SDValue Elt = N->getOperand(1);
  SDValue Idx = N->getOperand(2);
  GetSplitVector(Vec, Lo, Hi);

  if (ConstantSDNode *CIdx = dyn_cast<ConstantSDNode>(Idx)) {
    unsigned IdxVal = CIdx->getValue();
    unsigned LoNumElts = Lo.getValueType().getVectorNumElements();
    if (IdxVal < LoNumElts)
      Lo = DAG.getNode(ISD::INSERT_VECTOR_ELT, Lo.getValueType(), Lo, Elt, Idx);
    else
      Hi = DAG.getNode(ISD::INSERT_VECTOR_ELT, Hi.getValueType(), Hi, Elt,
                       DAG.getIntPtrConstant(IdxVal - LoNumElts));
    return;
  }

  // Spill the vector to the stack.
  MVT VecVT = Vec.getValueType();
  MVT EltVT = VecVT.getVectorElementType();
  SDValue StackPtr = DAG.CreateStackTemporary(VecVT);
  SDValue Store = DAG.getStore(DAG.getEntryNode(), Vec, StackPtr, NULL, 0);

  // Store the new element.  This may be larger than the vector element type,
  // so use a truncating store.
  SDValue EltPtr = GetVectorElementPointer(StackPtr, EltVT, Idx);
  Store = DAG.getTruncStore(Store, Elt, EltPtr, NULL, 0, EltVT);

  // Reload the vector from the stack.
  SDValue Load = DAG.getLoad(VecVT, Store, StackPtr, NULL, 0);

  // Split it.
  SplitVecRes_LOAD(cast<LoadSDNode>(Load.getNode()), Lo, Hi);
}

void DAGTypeLegalizer::SplitVecRes_LOAD(LoadSDNode *LD, SDValue &Lo,
                                        SDValue &Hi) {
  assert(ISD::isUNINDEXEDLoad(LD) && "Indexed load during type legalization!");
  MVT LoVT, HiVT;
  GetSplitDestVTs(LD->getValueType(0), LoVT, HiVT);

  ISD::LoadExtType ExtType = LD->getExtensionType();
  SDValue Ch = LD->getChain();
  SDValue Ptr = LD->getBasePtr();
  SDValue Offset = DAG.getNode(ISD::UNDEF, Ptr.getValueType());
  const Value *SV = LD->getSrcValue();
  int SVOffset = LD->getSrcValueOffset();
  MVT MemoryVT = LD->getMemoryVT();
  unsigned Alignment = LD->getAlignment();
  bool isVolatile = LD->isVolatile();

  MVT LoMemVT, HiMemVT;
  GetSplitDestVTs(MemoryVT, LoMemVT, HiMemVT);

  Lo = DAG.getLoad(ISD::UNINDEXED, ExtType, LoVT, Ch, Ptr, Offset,
                   SV, SVOffset, LoMemVT, isVolatile, Alignment);

  unsigned IncrementSize = LoMemVT.getSizeInBits()/8;
  Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                    DAG.getIntPtrConstant(IncrementSize));
  SVOffset += IncrementSize;
  Alignment = MinAlign(Alignment, IncrementSize);
  Hi = DAG.getLoad(ISD::UNINDEXED, ExtType, HiVT, Ch, Ptr, Offset,
                   SV, SVOffset, HiMemVT, isVolatile, Alignment);

  // Build a factor node to remember that this load is independent of the
  // other one.
  Ch = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                   Hi.getValue(1));

  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDValue(LD, 1), Ch);
}

void DAGTypeLegalizer::SplitVecRes_UnaryOp(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  // Get the dest types - they may not match the input types, e.g. int_to_fp.
  MVT LoVT, HiVT;
  GetSplitDestVTs(N->getValueType(0), LoVT, HiVT);

  GetSplitVector(N->getOperand(0), Lo, Hi);
  Lo = DAG.getNode(N->getOpcode(), LoVT, Lo);
  Hi = DAG.getNode(N->getOpcode(), HiVT, Hi);
}

void DAGTypeLegalizer::SplitVecRes_VECTOR_SHUFFLE(SDNode *N, SDValue &Lo,
                                                  SDValue &Hi) {
  // Build the low part.
  SDValue Mask = N->getOperand(2);
  SmallVector<SDValue, 16> Ops;
  MVT LoVT, HiVT;
  GetSplitDestVTs(N->getValueType(0), LoVT, HiVT);
  MVT EltVT = LoVT.getVectorElementType();
  unsigned LoNumElts = LoVT.getVectorNumElements();
  unsigned NumElements = Mask.getNumOperands();

  // Insert all of the elements from the input that are needed.  We use
  // buildvector of extractelement here because the input vectors will have
  // to be legalized, so this makes the code simpler.
  for (unsigned i = 0; i != LoNumElts; ++i) {
    unsigned Idx = cast<ConstantSDNode>(Mask.getOperand(i))->getValue();
    SDValue InVec = N->getOperand(0);
    if (Idx >= NumElements) {
      InVec = N->getOperand(1);
      Idx -= NumElements;
    }
    Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, EltVT, InVec,
                              DAG.getIntPtrConstant(Idx)));
  }
  Lo = DAG.getNode(ISD::BUILD_VECTOR, LoVT, &Ops[0], Ops.size());
  Ops.clear();

  for (unsigned i = LoNumElts; i != NumElements; ++i) {
    unsigned Idx = cast<ConstantSDNode>(Mask.getOperand(i))->getValue();
    SDValue InVec = N->getOperand(0);
    if (Idx >= NumElements) {
      InVec = N->getOperand(1);
      Idx -= NumElements;
    }
    Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, EltVT, InVec,
                              DAG.getIntPtrConstant(Idx)));
  }
  Hi = DAG.getNode(ISD::BUILD_VECTOR, HiVT, &Ops[0], Ops.size());
}

void DAGTypeLegalizer::SplitVecRes_VSETCC(SDNode *N, SDValue &Lo,
                                          SDValue &Hi) {
  MVT LoVT, HiVT;
  GetSplitDestVTs(N->getValueType(0), LoVT, HiVT);

  SDValue LL, LH, RL, RH;
  GetSplitVector(N->getOperand(0), LL, LH);
  GetSplitVector(N->getOperand(1), RL, RH);

  Lo = DAG.getNode(ISD::VSETCC, LoVT, LL, RL, N->getOperand(2));
  Hi = DAG.getNode(ISD::VSETCC, HiVT, LH, RH, N->getOperand(2));
}


//===----------------------------------------------------------------------===//
//  Operand Vector Splitting
//===----------------------------------------------------------------------===//

/// SplitVectorOperand - This method is called when the specified operand of the
/// specified node is found to need vector splitting.  At this point, all of the
/// result types of the node are known to be legal, but other operands of the
/// node may need legalization as well as the specified one.
bool DAGTypeLegalizer::SplitVectorOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Split node operand: "; N->dump(&DAG); cerr << "\n");
  SDValue Res = SDValue();

  if (Res.getNode() == 0) {
    switch (N->getOpcode()) {
    default:
#ifndef NDEBUG
      cerr << "SplitVectorOperand Op #" << OpNo << ": ";
      N->dump(&DAG); cerr << "\n";
#endif
      assert(0 && "Do not know how to split this operator's operand!");
      abort();

    case ISD::BIT_CONVERT:       Res = SplitVecOp_BIT_CONVERT(N); break;
    case ISD::EXTRACT_SUBVECTOR: Res = SplitVecOp_EXTRACT_SUBVECTOR(N); break;
    case ISD::EXTRACT_VECTOR_ELT:Res = SplitVecOp_EXTRACT_VECTOR_ELT(N); break;
    case ISD::STORE:             Res = SplitVecOp_STORE(cast<StoreSDNode>(N),
                                                        OpNo); break;
    case ISD::VECTOR_SHUFFLE:    Res = SplitVecOp_VECTOR_SHUFFLE(N, OpNo);break;
    }
  }

  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.getNode()) return false;

  // If the result is N, the sub-method updated N in place.  Check to see if any
  // operands are new, and if so, mark them.
  if (Res.getNode() == N) {
    // Mark N as new and remark N and its operands.  This allows us to correctly
    // revisit N if it needs another step of promotion and allows us to visit
    // any new operands to N.
    ReanalyzeNode(N);
    return true;
  }

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");

  ReplaceValueWith(SDValue(N, 0), Res);
  return false;
}

SDValue DAGTypeLegalizer::SplitVecOp_BIT_CONVERT(SDNode *N) {
  // For example, i64 = BIT_CONVERT v4i16 on alpha.  Typically the vector will
  // end up being split all the way down to individual components.  Convert the
  // split pieces into integers and reassemble.
  SDValue Lo, Hi;
  GetSplitVector(N->getOperand(0), Lo, Hi);
  Lo = BitConvertToInteger(Lo);
  Hi = BitConvertToInteger(Hi);

  if (TLI.isBigEndian())
    std::swap(Lo, Hi);

  return DAG.getNode(ISD::BIT_CONVERT, N->getValueType(0),
                     JoinIntegers(Lo, Hi));
}

SDValue DAGTypeLegalizer::SplitVecOp_EXTRACT_SUBVECTOR(SDNode *N) {
  // We know that the extracted result type is legal.  For now, assume the index
  // is a constant.
  MVT SubVT = N->getValueType(0);
  SDValue Idx = N->getOperand(1);
  SDValue Lo, Hi;
  GetSplitVector(N->getOperand(0), Lo, Hi);

  uint64_t LoElts = Lo.getValueType().getVectorNumElements();
  uint64_t IdxVal = cast<ConstantSDNode>(Idx)->getValue();

  if (IdxVal < LoElts) {
    assert(IdxVal + SubVT.getVectorNumElements() <= LoElts &&
           "Extracted subvector crosses vector split!");
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, SubVT, Lo, Idx);
  } else {
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, SubVT, Hi,
                       DAG.getConstant(IdxVal - LoElts, Idx.getValueType()));
  }
}

SDValue DAGTypeLegalizer::SplitVecOp_EXTRACT_VECTOR_ELT(SDNode *N) {
  SDValue Vec = N->getOperand(0);
  SDValue Idx = N->getOperand(1);
  MVT VecVT = Vec.getValueType();

  if (isa<ConstantSDNode>(Idx)) {
    uint64_t IdxVal = cast<ConstantSDNode>(Idx)->getValue();
    assert(IdxVal < VecVT.getVectorNumElements() && "Invalid vector index!");

    SDValue Lo, Hi;
    GetSplitVector(Vec, Lo, Hi);

    uint64_t LoElts = Lo.getValueType().getVectorNumElements();

    if (IdxVal < LoElts)
      return DAG.UpdateNodeOperands(SDValue(N, 0), Lo, Idx);
    else
      return DAG.UpdateNodeOperands(SDValue(N, 0), Hi,
                                    DAG.getConstant(IdxVal - LoElts,
                                                    Idx.getValueType()));
  }

  // Store the vector to the stack.
  MVT EltVT = VecVT.getVectorElementType();
  SDValue StackPtr = DAG.CreateStackTemporary(VecVT);
  SDValue Store = DAG.getStore(DAG.getEntryNode(), Vec, StackPtr, NULL, 0);

  // Load back the required element.
  StackPtr = GetVectorElementPointer(StackPtr, EltVT, Idx);
  return DAG.getLoad(EltVT, Store, StackPtr, NULL, 0);
}

SDValue DAGTypeLegalizer::SplitVecOp_STORE(StoreSDNode *N, unsigned OpNo) {
  assert(N->isUnindexed() && "Indexed store of vector?");
  assert(OpNo == 1 && "Can only split the stored value");

  bool isTruncating = N->isTruncatingStore();
  SDValue Ch  = N->getChain();
  SDValue Ptr = N->getBasePtr();
  int SVOffset = N->getSrcValueOffset();
  MVT MemoryVT = N->getMemoryVT();
  unsigned Alignment = N->getAlignment();
  bool isVol = N->isVolatile();
  SDValue Lo, Hi;
  GetSplitVector(N->getOperand(1), Lo, Hi);

  MVT LoMemVT, HiMemVT;
  GetSplitDestVTs(MemoryVT, LoMemVT, HiMemVT);

  unsigned IncrementSize = LoMemVT.getSizeInBits()/8;

  if (isTruncating)
    Lo = DAG.getTruncStore(Ch, Lo, Ptr, N->getSrcValue(), SVOffset,
                           LoMemVT, isVol, Alignment);
  else
    Lo = DAG.getStore(Ch, Lo, Ptr, N->getSrcValue(), SVOffset,
                      isVol, Alignment);

  // Increment the pointer to the other half.
  Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                    DAG.getIntPtrConstant(IncrementSize));

  if (isTruncating)
    Hi = DAG.getTruncStore(Ch, Hi, Ptr,
                           N->getSrcValue(), SVOffset+IncrementSize,
                           HiMemVT,
                           isVol, MinAlign(Alignment, IncrementSize));
  else
    Hi = DAG.getStore(Ch, Hi, Ptr, N->getSrcValue(), SVOffset+IncrementSize,
                      isVol, MinAlign(Alignment, IncrementSize));

  return DAG.getNode(ISD::TokenFactor, MVT::Other, Lo, Hi);
}

SDValue DAGTypeLegalizer::SplitVecOp_VECTOR_SHUFFLE(SDNode *N, unsigned OpNo){
  assert(OpNo == 2 && "Shuffle source type differs from result type?");
  SDValue Mask = N->getOperand(2);
  unsigned MaskLength = Mask.getValueType().getVectorNumElements();
  unsigned LargestMaskEntryPlusOne = 2 * MaskLength;
  unsigned MinimumBitWidth = Log2_32_Ceil(LargestMaskEntryPlusOne);

  // Look for a legal vector type to place the mask values in.
  // Note that there may not be *any* legal vector-of-integer
  // type for which the element type is legal!
  for (MVT::SimpleValueType EltVT = MVT::FIRST_INTEGER_VALUETYPE;
       EltVT <= MVT::LAST_INTEGER_VALUETYPE;
       // Integer values types are consecutively numbered.  Exploit this.
       EltVT = MVT::SimpleValueType(EltVT + 1)) {

    // Is the element type big enough to hold the values?
    if (MVT(EltVT).getSizeInBits() < MinimumBitWidth)
      // Nope.
      continue;

    // Is the vector type legal?
    MVT VecVT = MVT::getVectorVT(EltVT, MaskLength);
    if (!isTypeLegal(VecVT))
      // Nope.
      continue;

    // If the element type is not legal, find a larger legal type to use for
    // the BUILD_VECTOR operands.  This is an ugly hack, but seems to work!
    // FIXME: The real solution is to change VECTOR_SHUFFLE into a variadic
    // node where the shuffle mask is a list of integer operands, #2 .. #2+n.
    for (MVT::SimpleValueType OpVT = EltVT; OpVT <= MVT::LAST_INTEGER_VALUETYPE;
         // Integer values types are consecutively numbered.  Exploit this.
         OpVT = MVT::SimpleValueType(OpVT + 1)) {
      if (!isTypeLegal(OpVT))
        continue;

      // Success!  Rebuild the vector using the legal types.
      SmallVector<SDValue, 16> Ops(MaskLength);
      for (unsigned i = 0; i < MaskLength; ++i) {
        uint64_t Idx =
          cast<ConstantSDNode>(Mask.getOperand(i))->getValue();
        Ops[i] = DAG.getConstant(Idx, OpVT);
      }
      return DAG.UpdateNodeOperands(SDValue(N,0),
                                    N->getOperand(0), N->getOperand(1),
                                    DAG.getNode(ISD::BUILD_VECTOR,
                                                VecVT, &Ops[0], Ops.size()));
    }

    // Continuing is pointless - failure is certain.
    break;
  }
  assert(false && "Failed to find an appropriate mask type!");
  return SDValue(N, 0);
}
