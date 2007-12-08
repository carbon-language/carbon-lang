//===-- LegalizeTypesExpand.cpp - Expansion for LegalizeTypes -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements expansion support for LegalizeTypes.  Expansion is the
// act of changing a computation in an invalid type to be a computation in
// multiple registers of a smaller type.  For example, implementing i64
// arithmetic in two i32 registers (as is often needed on 32-bit targets, for
// example).
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
#include "llvm/Constants.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  Result Expansion
//===----------------------------------------------------------------------===//

/// ExpandResult - This method is called when the specified result of the
/// specified node is found to need expansion.  At this point, the node may also
/// have invalid operands or may have other results that need promotion, we just
/// know that (at least) one result needs expansion.
void DAGTypeLegalizer::ExpandResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Expand node result: "; N->dump(&DAG); cerr << "\n");
  SDOperand Lo, Hi;
  Lo = Hi = SDOperand();

  // See if the target wants to custom expand this node.
  if (TLI.getOperationAction(N->getOpcode(), N->getValueType(0)) == 
          TargetLowering::Custom) {
    // If the target wants to, allow it to lower this itself.
    if (SDNode *P = TLI.ExpandOperationResult(N, DAG)) {
      // Everything that once used N now uses P.  We are guaranteed that the
      // result value types of N and the result value types of P match.
      ReplaceNodeWith(N, P);
      return;
    }
  }

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "ExpandResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to expand the result of this operator!");
    abort();
      
  case ISD::UNDEF:       ExpandResult_UNDEF(N, Lo, Hi); break;
  case ISD::Constant:    ExpandResult_Constant(N, Lo, Hi); break;
  case ISD::BUILD_PAIR:  ExpandResult_BUILD_PAIR(N, Lo, Hi); break;
  case ISD::MERGE_VALUES: ExpandResult_MERGE_VALUES(N, Lo, Hi); break;
  case ISD::ANY_EXTEND:  ExpandResult_ANY_EXTEND(N, Lo, Hi); break;
  case ISD::ZERO_EXTEND: ExpandResult_ZERO_EXTEND(N, Lo, Hi); break;
  case ISD::SIGN_EXTEND: ExpandResult_SIGN_EXTEND(N, Lo, Hi); break;
  case ISD::BIT_CONVERT: ExpandResult_BIT_CONVERT(N, Lo, Hi); break;
  case ISD::SIGN_EXTEND_INREG: ExpandResult_SIGN_EXTEND_INREG(N, Lo, Hi); break;
  case ISD::LOAD:        ExpandResult_LOAD(cast<LoadSDNode>(N), Lo, Hi); break;
    
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:         ExpandResult_Logical(N, Lo, Hi); break;
  case ISD::BSWAP:       ExpandResult_BSWAP(N, Lo, Hi); break;
  case ISD::ADD:
  case ISD::SUB:         ExpandResult_ADDSUB(N, Lo, Hi); break;
  case ISD::ADDC:
  case ISD::SUBC:        ExpandResult_ADDSUBC(N, Lo, Hi); break;
  case ISD::ADDE:
  case ISD::SUBE:        ExpandResult_ADDSUBE(N, Lo, Hi); break;
  case ISD::SELECT:      ExpandResult_SELECT(N, Lo, Hi); break;
  case ISD::SELECT_CC:   ExpandResult_SELECT_CC(N, Lo, Hi); break;
  case ISD::MUL:         ExpandResult_MUL(N, Lo, Hi); break;
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:         ExpandResult_Shift(N, Lo, Hi); break;
  }
  
  // If Lo/Hi is null, the sub-method took care of registering results etc.
  if (Lo.Val)
    SetExpandedOp(SDOperand(N, ResNo), Lo, Hi);
}

void DAGTypeLegalizer::ExpandResult_UNDEF(SDNode *N,
                                          SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  Lo = Hi = DAG.getNode(ISD::UNDEF, NVT);
}

void DAGTypeLegalizer::ExpandResult_Constant(SDNode *N,
                                             SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  uint64_t Cst = cast<ConstantSDNode>(N)->getValue();
  Lo = DAG.getConstant(Cst, NVT);
  Hi = DAG.getConstant(Cst >> MVT::getSizeInBits(NVT), NVT);
}

void DAGTypeLegalizer::ExpandResult_BUILD_PAIR(SDNode *N,
                                               SDOperand &Lo, SDOperand &Hi) {
  // Return the operands.
  Lo = N->getOperand(0);
  Hi = N->getOperand(1);
}

void DAGTypeLegalizer::ExpandResult_MERGE_VALUES(SDNode *N,
                                                 SDOperand &Lo, SDOperand &Hi) {
  // A MERGE_VALUES node can produce any number of values.  We know that the
  // first illegal one needs to be expanded into Lo/Hi.
  unsigned i;
  
  // The string of legal results gets turns into the input operands, which have
  // the same type.
  for (i = 0; isTypeLegal(N->getValueType(i)); ++i)
    ReplaceValueWith(SDOperand(N, i), SDOperand(N->getOperand(i)));

  // The first illegal result must be the one that needs to be expanded.
  GetExpandedOp(N->getOperand(i), Lo, Hi);

  // Legalize the rest of the results into the input operands whether they are
  // legal or not.
  unsigned e = N->getNumValues();
  for (++i; i != e; ++i)
    ReplaceValueWith(SDOperand(N, i), SDOperand(N->getOperand(i)));
}

void DAGTypeLegalizer::ExpandResult_ANY_EXTEND(SDNode *N,
                                               SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDOperand Op = N->getOperand(0);
  if (MVT::getSizeInBits(Op.getValueType()) <= MVT::getSizeInBits(NVT)) {
    // The low part is any extension of the input (which degenerates to a copy).
    Lo = DAG.getNode(ISD::ANY_EXTEND, NVT, Op);
    Hi = DAG.getNode(ISD::UNDEF, NVT);   // The high part is undefined.
  } else {
    // For example, extension of an i48 to an i64.  The operand type necessarily
    // promotes to the result type, so will end up being expanded too.
    assert(getTypeAction(Op.getValueType()) == Promote &&
           "Don't know how to expand this result!");
    SDOperand Res = GetPromotedOp(Op);
    assert(Res.getValueType() == N->getValueType(0) &&
           "Operand over promoted?");
    // Split the promoted operand.  This will simplify when it is expanded.
    SplitOp(Res, Lo, Hi);
  }
}

void DAGTypeLegalizer::ExpandResult_ZERO_EXTEND(SDNode *N,
                                                SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDOperand Op = N->getOperand(0);
  if (MVT::getSizeInBits(Op.getValueType()) <= MVT::getSizeInBits(NVT)) {
    // The low part is zero extension of the input (which degenerates to a copy).
    Lo = DAG.getNode(ISD::ZERO_EXTEND, NVT, N->getOperand(0));
    Hi = DAG.getConstant(0, NVT);   // The high part is just a zero.
  } else {
    // For example, extension of an i48 to an i64.  The operand type necessarily
    // promotes to the result type, so will end up being expanded too.
    assert(getTypeAction(Op.getValueType()) == Promote &&
           "Don't know how to expand this result!");
    SDOperand Res = GetPromotedOp(Op);
    assert(Res.getValueType() == N->getValueType(0) &&
           "Operand over promoted?");
    // Split the promoted operand.  This will simplify when it is expanded.
    SplitOp(Res, Lo, Hi);
    unsigned ExcessBits =
      MVT::getSizeInBits(Op.getValueType()) - MVT::getSizeInBits(NVT);
    Hi = DAG.getZeroExtendInReg(Hi, MVT::getIntegerType(ExcessBits));
  }
}

void DAGTypeLegalizer::ExpandResult_SIGN_EXTEND(SDNode *N,
                                                SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDOperand Op = N->getOperand(0);
  if (MVT::getSizeInBits(Op.getValueType()) <= MVT::getSizeInBits(NVT)) {
    // The low part is sign extension of the input (which degenerates to a copy).
    Lo = DAG.getNode(ISD::SIGN_EXTEND, NVT, N->getOperand(0));
    // The high part is obtained by SRA'ing all but one of the bits of low part.
    unsigned LoSize = MVT::getSizeInBits(NVT);
    Hi = DAG.getNode(ISD::SRA, NVT, Lo,
                     DAG.getConstant(LoSize-1, TLI.getShiftAmountTy()));
  } else {
    // For example, extension of an i48 to an i64.  The operand type necessarily
    // promotes to the result type, so will end up being expanded too.
    assert(getTypeAction(Op.getValueType()) == Promote &&
           "Don't know how to expand this result!");
    SDOperand Res = GetPromotedOp(Op);
    assert(Res.getValueType() == N->getValueType(0) &&
           "Operand over promoted?");
    // Split the promoted operand.  This will simplify when it is expanded.
    SplitOp(Res, Lo, Hi);
    unsigned ExcessBits =
      MVT::getSizeInBits(Op.getValueType()) - MVT::getSizeInBits(NVT);
    Hi = DAG.getNode(ISD::SIGN_EXTEND_INREG, Hi.getValueType(), Hi,
                     DAG.getValueType(MVT::getIntegerType(ExcessBits)));
  }
}

void DAGTypeLegalizer::ExpandResult_BIT_CONVERT(SDNode *N,
                                                SDOperand &Lo, SDOperand &Hi) {
  // Lower the bit-convert to a store/load from the stack, then expand the load.
  SDOperand Op = CreateStackStoreLoad(N->getOperand(0), N->getValueType(0));
  ExpandResult_LOAD(cast<LoadSDNode>(Op.Val), Lo, Hi);
}

void DAGTypeLegalizer::
ExpandResult_SIGN_EXTEND_INREG(SDNode *N, SDOperand &Lo, SDOperand &Hi) {
  GetExpandedOp(N->getOperand(0), Lo, Hi);
  MVT::ValueType EVT = cast<VTSDNode>(N->getOperand(1))->getVT();

  if (MVT::getSizeInBits(EVT) <= MVT::getSizeInBits(Lo.getValueType())) {
    // sext_inreg the low part if needed.
    Lo = DAG.getNode(ISD::SIGN_EXTEND_INREG, Lo.getValueType(), Lo,
                     N->getOperand(1));

    // The high part gets the sign extension from the lo-part.  This handles
    // things like sextinreg V:i64 from i8.
    Hi = DAG.getNode(ISD::SRA, Hi.getValueType(), Lo,
                     DAG.getConstant(MVT::getSizeInBits(Hi.getValueType())-1,
                                     TLI.getShiftAmountTy()));
  } else {
    // For example, extension of an i48 to an i64.  Leave the low part alone,
    // sext_inreg the high part.
    unsigned ExcessBits =
      MVT::getSizeInBits(EVT) - MVT::getSizeInBits(Lo.getValueType());
    Hi = DAG.getNode(ISD::SIGN_EXTEND_INREG, Hi.getValueType(), Hi,
                     DAG.getValueType(MVT::getIntegerType(ExcessBits)));
  }
}

void DAGTypeLegalizer::ExpandResult_LOAD(LoadSDNode *N,
                                         SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  SDOperand Ch  = N->getChain();    // Legalize the chain.
  SDOperand Ptr = N->getBasePtr();  // Legalize the pointer.
  ISD::LoadExtType ExtType = N->getExtensionType();
  int SVOffset = N->getSrcValueOffset();
  unsigned Alignment = N->getAlignment();
  bool isVolatile = N->isVolatile();

  assert(!(MVT::getSizeInBits(NVT) & 7) && "Expanded type not byte sized!");

  if (ExtType == ISD::NON_EXTLOAD) {
    Lo = DAG.getLoad(NVT, Ch, Ptr, N->getSrcValue(), SVOffset,
                     isVolatile, Alignment);
    // Increment the pointer to the other half.
    unsigned IncrementSize = MVT::getSizeInBits(NVT)/8;
    Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                      getIntPtrConstant(IncrementSize));
    Hi = DAG.getLoad(NVT, Ch, Ptr, N->getSrcValue(), SVOffset+IncrementSize,
                     isVolatile, MinAlign(Alignment, IncrementSize));

    // Build a factor node to remember that this load is independent of the
    // other one.
    Ch = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                     Hi.getValue(1));

    // Handle endianness of the load.
    if (!TLI.isLittleEndian())
      std::swap(Lo, Hi);
  } else if (MVT::getSizeInBits(N->getLoadedVT()) <= MVT::getSizeInBits(NVT)) {
    MVT::ValueType EVT = N->getLoadedVT();

    Lo = DAG.getExtLoad(ExtType, NVT, Ch, Ptr, N->getSrcValue(), SVOffset, EVT,
                        isVolatile, Alignment);

    // Remember the chain.
    Ch = Lo.getValue(1);

    if (ExtType == ISD::SEXTLOAD) {
      // The high part is obtained by SRA'ing all but one of the bits of the
      // lo part.
      unsigned LoSize = MVT::getSizeInBits(Lo.getValueType());
      Hi = DAG.getNode(ISD::SRA, NVT, Lo,
                       DAG.getConstant(LoSize-1, TLI.getShiftAmountTy()));
    } else if (ExtType == ISD::ZEXTLOAD) {
      // The high part is just a zero.
      Hi = DAG.getConstant(0, NVT);
    } else {
      assert(ExtType == ISD::EXTLOAD && "Unknown extload!");
      // The high part is undefined.
      Hi = DAG.getNode(ISD::UNDEF, NVT);
    }
  } else if (TLI.isLittleEndian()) {
    // Little-endian - low bits are at low addresses.
    Lo = DAG.getLoad(NVT, Ch, Ptr, N->getSrcValue(), SVOffset,
                     isVolatile, Alignment);

    unsigned ExcessBits =
      MVT::getSizeInBits(N->getLoadedVT()) - MVT::getSizeInBits(NVT);
    MVT::ValueType NEVT = MVT::getIntegerType(ExcessBits);

    // Increment the pointer to the other half.
    unsigned IncrementSize = MVT::getSizeInBits(NVT)/8;
    Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                      getIntPtrConstant(IncrementSize));
    Hi = DAG.getExtLoad(ExtType, NVT, Ch, Ptr, N->getSrcValue(),
                        SVOffset+IncrementSize, NEVT,
                        isVolatile, MinAlign(Alignment, IncrementSize));

    // Build a factor node to remember that this load is independent of the
    // other one.
    Ch = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                     Hi.getValue(1));
  } else {
    // Big-endian - high bits are at low addresses.  Favor aligned loads at
    // the cost of some bit-fiddling.
    MVT::ValueType EVT = N->getLoadedVT();
    unsigned EBytes = MVT::getStoreSizeInBits(EVT)/8;
    unsigned IncrementSize = MVT::getSizeInBits(NVT)/8;
    unsigned ExcessBits = (EBytes - IncrementSize)*8;

    // Load both the high bits and maybe some of the low bits.
    Hi = DAG.getExtLoad(ExtType, NVT, Ch, Ptr, N->getSrcValue(), SVOffset,
                        MVT::getIntegerType(MVT::getSizeInBits(EVT)-ExcessBits),
                        isVolatile, Alignment);

    // Increment the pointer to the other half.
    Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                      getIntPtrConstant(IncrementSize));
    // Load the rest of the low bits.
    Lo = DAG.getExtLoad(ISD::ZEXTLOAD, NVT, Ch, Ptr, N->getSrcValue(),
                        SVOffset+IncrementSize, MVT::getIntegerType(ExcessBits),
                        isVolatile, MinAlign(Alignment, IncrementSize));

    // Build a factor node to remember that this load is independent of the
    // other one.
    Ch = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                     Hi.getValue(1));

    if (ExcessBits < MVT::getSizeInBits(NVT)) {
      // Transfer low bits from the bottom of Hi to the top of Lo.
      Lo = DAG.getNode(ISD::OR, NVT, Lo,
                       DAG.getNode(ISD::SHL, NVT, Hi,
                                   DAG.getConstant(ExcessBits,
                                                   TLI.getShiftAmountTy())));
      // Move high bits to the right position in Hi.
      Hi = DAG.getNode(ExtType == ISD::SEXTLOAD ? ISD::SRA : ISD::SRL, NVT, Hi,
                       DAG.getConstant(MVT::getSizeInBits(NVT) - ExcessBits,
                                       TLI.getShiftAmountTy()));
    }
  }

  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDOperand(N, 1), Ch);
}

void DAGTypeLegalizer::ExpandResult_Logical(SDNode *N,
                                            SDOperand &Lo, SDOperand &Hi) {
  SDOperand LL, LH, RL, RH;
  GetExpandedOp(N->getOperand(0), LL, LH);
  GetExpandedOp(N->getOperand(1), RL, RH);
  Lo = DAG.getNode(N->getOpcode(), LL.getValueType(), LL, RL);
  Hi = DAG.getNode(N->getOpcode(), LL.getValueType(), LH, RH);
}

void DAGTypeLegalizer::ExpandResult_BSWAP(SDNode *N,
                                          SDOperand &Lo, SDOperand &Hi) {
  GetExpandedOp(N->getOperand(0), Hi, Lo);  // Note swapped operands.
  Lo = DAG.getNode(ISD::BSWAP, Lo.getValueType(), Lo);
  Hi = DAG.getNode(ISD::BSWAP, Hi.getValueType(), Hi);
}

void DAGTypeLegalizer::ExpandResult_SELECT(SDNode *N,
                                           SDOperand &Lo, SDOperand &Hi) {
  SDOperand LL, LH, RL, RH;
  GetExpandedOp(N->getOperand(1), LL, LH);
  GetExpandedOp(N->getOperand(2), RL, RH);
  Lo = DAG.getNode(ISD::SELECT, LL.getValueType(), N->getOperand(0), LL, RL);
  
  assert(N->getOperand(0).getValueType() != MVT::f32 &&
         "FIXME: softfp shouldn't use expand!");
  Hi = DAG.getNode(ISD::SELECT, LL.getValueType(), N->getOperand(0), LH, RH);
}

void DAGTypeLegalizer::ExpandResult_SELECT_CC(SDNode *N,
                                              SDOperand &Lo, SDOperand &Hi) {
  SDOperand LL, LH, RL, RH;
  GetExpandedOp(N->getOperand(2), LL, LH);
  GetExpandedOp(N->getOperand(3), RL, RH);
  Lo = DAG.getNode(ISD::SELECT_CC, LL.getValueType(), N->getOperand(0), 
                   N->getOperand(1), LL, RL, N->getOperand(4));
  
  assert(N->getOperand(0).getValueType() != MVT::f32 &&
         "FIXME: softfp shouldn't use expand!");
  Hi = DAG.getNode(ISD::SELECT_CC, LL.getValueType(), N->getOperand(0), 
                   N->getOperand(1), LH, RH, N->getOperand(4));
}

void DAGTypeLegalizer::ExpandResult_ADDSUB(SDNode *N,
                                           SDOperand &Lo, SDOperand &Hi) {
  // Expand the subcomponents.
  SDOperand LHSL, LHSH, RHSL, RHSH;
  GetExpandedOp(N->getOperand(0), LHSL, LHSH);
  GetExpandedOp(N->getOperand(1), RHSL, RHSH);
  SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
  SDOperand LoOps[2] = { LHSL, RHSL };
  SDOperand HiOps[3] = { LHSH, RHSH };

  if (N->getOpcode() == ISD::ADD) {
    Lo = DAG.getNode(ISD::ADDC, VTList, LoOps, 2);
    HiOps[2] = Lo.getValue(1);
    Hi = DAG.getNode(ISD::ADDE, VTList, HiOps, 3);
  } else {
    Lo = DAG.getNode(ISD::SUBC, VTList, LoOps, 2);
    HiOps[2] = Lo.getValue(1);
    Hi = DAG.getNode(ISD::SUBE, VTList, HiOps, 3);
  }
}

void DAGTypeLegalizer::ExpandResult_ADDSUBC(SDNode *N,
                                            SDOperand &Lo, SDOperand &Hi) {
  // Expand the subcomponents.
  SDOperand LHSL, LHSH, RHSL, RHSH;
  GetExpandedOp(N->getOperand(0), LHSL, LHSH);
  GetExpandedOp(N->getOperand(1), RHSL, RHSH);
  SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
  SDOperand LoOps[2] = { LHSL, RHSL };
  SDOperand HiOps[3] = { LHSH, RHSH };

  if (N->getOpcode() == ISD::ADDC) {
    Lo = DAG.getNode(ISD::ADDC, VTList, LoOps, 2);
    HiOps[2] = Lo.getValue(1);
    Hi = DAG.getNode(ISD::ADDE, VTList, HiOps, 3);
  } else {
    Lo = DAG.getNode(ISD::SUBC, VTList, LoOps, 2);
    HiOps[2] = Lo.getValue(1);
    Hi = DAG.getNode(ISD::SUBE, VTList, HiOps, 3);
  }

  // Legalized the flag result - switch anything that used the old flag to
  // use the new one.
  ReplaceValueWith(SDOperand(N, 1), Hi.getValue(1));
}

void DAGTypeLegalizer::ExpandResult_ADDSUBE(SDNode *N,
                                            SDOperand &Lo, SDOperand &Hi) {
  // Expand the subcomponents.
  SDOperand LHSL, LHSH, RHSL, RHSH;
  GetExpandedOp(N->getOperand(0), LHSL, LHSH);
  GetExpandedOp(N->getOperand(1), RHSL, RHSH);
  SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
  SDOperand LoOps[3] = { LHSL, RHSL, N->getOperand(2) };
  SDOperand HiOps[3] = { LHSH, RHSH };

  Lo = DAG.getNode(N->getOpcode(), VTList, LoOps, 3);
  HiOps[2] = Lo.getValue(1);
  Hi = DAG.getNode(N->getOpcode(), VTList, HiOps, 3);

  // Legalized the flag result - switch anything that used the old flag to
  // use the new one.
  ReplaceValueWith(SDOperand(N, 1), Hi.getValue(1));
}

void DAGTypeLegalizer::ExpandResult_MUL(SDNode *N,
                                        SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  
  bool HasMULHS = TLI.isOperationLegal(ISD::MULHS, NVT);
  bool HasMULHU = TLI.isOperationLegal(ISD::MULHU, NVT);
  bool HasSMUL_LOHI = TLI.isOperationLegal(ISD::SMUL_LOHI, NVT);
  bool HasUMUL_LOHI = TLI.isOperationLegal(ISD::UMUL_LOHI, NVT);
  if (HasMULHU || HasMULHS || HasUMUL_LOHI || HasSMUL_LOHI) {
    SDOperand LL, LH, RL, RH;
    GetExpandedOp(N->getOperand(0), LL, LH);
    GetExpandedOp(N->getOperand(1), RL, RH);
    unsigned BitSize = MVT::getSizeInBits(NVT);
    unsigned LHSSB = DAG.ComputeNumSignBits(N->getOperand(0));
    unsigned RHSSB = DAG.ComputeNumSignBits(N->getOperand(1));
    
    // FIXME: generalize this to handle other bit sizes
    if (LHSSB == 32 && RHSSB == 32 &&
        DAG.MaskedValueIsZero(N->getOperand(0), 0xFFFFFFFF00000000ULL) &&
        DAG.MaskedValueIsZero(N->getOperand(1), 0xFFFFFFFF00000000ULL)) {
      // The inputs are both zero-extended.
      if (HasUMUL_LOHI) {
        // We can emit a umul_lohi.
        Lo = DAG.getNode(ISD::UMUL_LOHI, DAG.getVTList(NVT, NVT), LL, RL);
        Hi = SDOperand(Lo.Val, 1);
        return;
      }
      if (HasMULHU) {
        // We can emit a mulhu+mul.
        Lo = DAG.getNode(ISD::MUL, NVT, LL, RL);
        Hi = DAG.getNode(ISD::MULHU, NVT, LL, RL);
        return;
      }
    }
    if (LHSSB > BitSize && RHSSB > BitSize) {
      // The input values are both sign-extended.
      if (HasSMUL_LOHI) {
        // We can emit a smul_lohi.
        Lo = DAG.getNode(ISD::SMUL_LOHI, DAG.getVTList(NVT, NVT), LL, RL);
        Hi = SDOperand(Lo.Val, 1);
        return;
      }
      if (HasMULHS) {
        // We can emit a mulhs+mul.
        Lo = DAG.getNode(ISD::MUL, NVT, LL, RL);
        Hi = DAG.getNode(ISD::MULHS, NVT, LL, RL);
        return;
      }
    }
    if (HasUMUL_LOHI) {
      // Lo,Hi = umul LHS, RHS.
      SDOperand UMulLOHI = DAG.getNode(ISD::UMUL_LOHI,
                                       DAG.getVTList(NVT, NVT), LL, RL);
      Lo = UMulLOHI;
      Hi = UMulLOHI.getValue(1);
      RH = DAG.getNode(ISD::MUL, NVT, LL, RH);
      LH = DAG.getNode(ISD::MUL, NVT, LH, RL);
      Hi = DAG.getNode(ISD::ADD, NVT, Hi, RH);
      Hi = DAG.getNode(ISD::ADD, NVT, Hi, LH);
      return;
    }
  }
  
  abort();
#if 0 // FIXME!
  // If nothing else, we can make a libcall.
  Lo = ExpandLibCall(TLI.getLibcallName(RTLIB::MUL_I64), N,
                     false/*sign irrelevant*/, Hi);
#endif
}  


void DAGTypeLegalizer::ExpandResult_Shift(SDNode *N,
                                          SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType VT = N->getValueType(0);
  
  // If we can emit an efficient shift operation, do so now.  Check to see if 
  // the RHS is a constant.
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    return ExpandShiftByConstant(N, CN->getValue(), Lo, Hi);

  // If we can determine that the high bit of the shift is zero or one, even if
  // the low bits are variable, emit this shift in an optimized form.
  if (ExpandShiftWithKnownAmountBit(N, Lo, Hi))
    return;
  
  // If this target supports shift_PARTS, use it.  First, map to the _PARTS opc.
  unsigned PartsOpc;
  if (N->getOpcode() == ISD::SHL)
    PartsOpc = ISD::SHL_PARTS;
  else if (N->getOpcode() == ISD::SRL)
    PartsOpc = ISD::SRL_PARTS;
  else {
    assert(N->getOpcode() == ISD::SRA && "Unknown shift!");
    PartsOpc = ISD::SRA_PARTS;
  }
  
  // Next check to see if the target supports this SHL_PARTS operation or if it
  // will custom expand it.
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  TargetLowering::LegalizeAction Action = TLI.getOperationAction(PartsOpc, NVT);
  if ((Action == TargetLowering::Legal && TLI.isTypeLegal(NVT)) ||
      Action == TargetLowering::Custom) {
    // Expand the subcomponents.
    SDOperand LHSL, LHSH;
    GetExpandedOp(N->getOperand(0), LHSL, LHSH);
    
    SDOperand Ops[] = { LHSL, LHSH, N->getOperand(1) };
    MVT::ValueType VT = LHSL.getValueType();
    Lo = DAG.getNode(PartsOpc, DAG.getNodeValueTypes(VT, VT), 2, Ops, 3);
    Hi = Lo.getValue(1);
    return;
  }
  
  abort();
#if 0 // FIXME!
  // Otherwise, emit a libcall.
  unsigned RuntimeCode = ; // SRL -> SRL_I64 etc.
  bool Signed = ;
  Lo = ExpandLibCall(TLI.getLibcallName(RTLIB::SRL_I64), N,
                     false/*lshr is unsigned*/, Hi);
#endif
}  


/// ExpandShiftByConstant - N is a shift by a value that needs to be expanded,
/// and the shift amount is a constant 'Amt'.  Expand the operation.
void DAGTypeLegalizer::ExpandShiftByConstant(SDNode *N, unsigned Amt, 
                                             SDOperand &Lo, SDOperand &Hi) {
  // Expand the incoming operand to be shifted, so that we have its parts
  SDOperand InL, InH;
  GetExpandedOp(N->getOperand(0), InL, InH);
  
  MVT::ValueType NVT = InL.getValueType();
  unsigned VTBits = MVT::getSizeInBits(N->getValueType(0));
  unsigned NVTBits = MVT::getSizeInBits(NVT);
  MVT::ValueType ShTy = N->getOperand(1).getValueType();

  if (N->getOpcode() == ISD::SHL) {
    if (Amt > VTBits) {
      Lo = Hi = DAG.getConstant(0, NVT);
    } else if (Amt > NVTBits) {
      Lo = DAG.getConstant(0, NVT);
      Hi = DAG.getNode(ISD::SHL, NVT, InL, DAG.getConstant(Amt-NVTBits,ShTy));
    } else if (Amt == NVTBits) {
      Lo = DAG.getConstant(0, NVT);
      Hi = InL;
    } else {
      Lo = DAG.getNode(ISD::SHL, NVT, InL, DAG.getConstant(Amt, ShTy));
      Hi = DAG.getNode(ISD::OR, NVT,
                       DAG.getNode(ISD::SHL, NVT, InH,
                                   DAG.getConstant(Amt, ShTy)),
                       DAG.getNode(ISD::SRL, NVT, InL,
                                   DAG.getConstant(NVTBits-Amt, ShTy)));
    }
    return;
  }
  
  if (N->getOpcode() == ISD::SRL) {
    if (Amt > VTBits) {
      Lo = DAG.getConstant(0, NVT);
      Hi = DAG.getConstant(0, NVT);
    } else if (Amt > NVTBits) {
      Lo = DAG.getNode(ISD::SRL, NVT, InH, DAG.getConstant(Amt-NVTBits,ShTy));
      Hi = DAG.getConstant(0, NVT);
    } else if (Amt == NVTBits) {
      Lo = InH;
      Hi = DAG.getConstant(0, NVT);
    } else {
      Lo = DAG.getNode(ISD::OR, NVT,
                       DAG.getNode(ISD::SRL, NVT, InL,
                                   DAG.getConstant(Amt, ShTy)),
                       DAG.getNode(ISD::SHL, NVT, InH,
                                   DAG.getConstant(NVTBits-Amt, ShTy)));
      Hi = DAG.getNode(ISD::SRL, NVT, InH, DAG.getConstant(Amt, ShTy));
    }
    return;
  }
  
  assert(N->getOpcode() == ISD::SRA && "Unknown shift!");
  if (Amt > VTBits) {
    Hi = Lo = DAG.getNode(ISD::SRA, NVT, InH,
                          DAG.getConstant(NVTBits-1, ShTy));
  } else if (Amt > NVTBits) {
    Lo = DAG.getNode(ISD::SRA, NVT, InH,
                     DAG.getConstant(Amt-NVTBits, ShTy));
    Hi = DAG.getNode(ISD::SRA, NVT, InH,
                     DAG.getConstant(NVTBits-1, ShTy));
  } else if (Amt == NVTBits) {
    Lo = InH;
    Hi = DAG.getNode(ISD::SRA, NVT, InH,
                     DAG.getConstant(NVTBits-1, ShTy));
  } else {
    Lo = DAG.getNode(ISD::OR, NVT,
                     DAG.getNode(ISD::SRL, NVT, InL,
                                 DAG.getConstant(Amt, ShTy)),
                     DAG.getNode(ISD::SHL, NVT, InH,
                                 DAG.getConstant(NVTBits-Amt, ShTy)));
    Hi = DAG.getNode(ISD::SRA, NVT, InH, DAG.getConstant(Amt, ShTy));
  }
}

/// ExpandShiftWithKnownAmountBit - Try to determine whether we can simplify
/// this shift based on knowledge of the high bit of the shift amount.  If we
/// can tell this, we know that it is >= 32 or < 32, without knowing the actual
/// shift amount.
bool DAGTypeLegalizer::
ExpandShiftWithKnownAmountBit(SDNode *N, SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  unsigned NVTBits = MVT::getSizeInBits(NVT);
  assert(!(NVTBits & (NVTBits - 1)) &&
         "Expanded integer type size not a power of two!");

  uint64_t HighBitMask = NVTBits, KnownZero, KnownOne;
  DAG.ComputeMaskedBits(N->getOperand(1), HighBitMask, KnownZero, KnownOne);
  
  // If we don't know anything about the high bit, exit.
  if (((KnownZero|KnownOne) & HighBitMask) == 0)
    return false;

  // Get the incoming operand to be shifted.
  SDOperand InL, InH;
  GetExpandedOp(N->getOperand(0), InL, InH);
  SDOperand Amt = N->getOperand(1);

  // If we know that the high bit of the shift amount is one, then we can do
  // this as a couple of simple shifts.
  if (KnownOne & HighBitMask) {
    // Mask out the high bit, which we know is set.
    Amt = DAG.getNode(ISD::AND, Amt.getValueType(), Amt,
                      DAG.getConstant(NVTBits-1, Amt.getValueType()));
    
    switch (N->getOpcode()) {
    default: assert(0 && "Unknown shift");
    case ISD::SHL:
      Lo = DAG.getConstant(0, NVT);              // Low part is zero.
      Hi = DAG.getNode(ISD::SHL, NVT, InL, Amt); // High part from Lo part.
      return true;
    case ISD::SRL:
      Hi = DAG.getConstant(0, NVT);              // Hi part is zero.
      Lo = DAG.getNode(ISD::SRL, NVT, InH, Amt); // Lo part from Hi part.
      return true;
    case ISD::SRA:
      Hi = DAG.getNode(ISD::SRA, NVT, InH,       // Sign extend high part.
                       DAG.getConstant(NVTBits-1, Amt.getValueType()));
      Lo = DAG.getNode(ISD::SRA, NVT, InH, Amt); // Lo part from Hi part.
      return true;
    }
  }
  
  // If we know that the high bit of the shift amount is zero, then we can do
  // this as a couple of simple shifts.
  assert((KnownZero & HighBitMask) && "Bad mask computation above");

  // Compute 32-amt.
  SDOperand Amt2 = DAG.getNode(ISD::SUB, Amt.getValueType(),
                               DAG.getConstant(NVTBits, Amt.getValueType()),
                               Amt);
  unsigned Op1, Op2;
  switch (N->getOpcode()) {
  default: assert(0 && "Unknown shift");
  case ISD::SHL:  Op1 = ISD::SHL; Op2 = ISD::SRL; break;
  case ISD::SRL:
  case ISD::SRA:  Op1 = ISD::SRL; Op2 = ISD::SHL; break;
  }
    
  Lo = DAG.getNode(N->getOpcode(), NVT, InL, Amt);
  Hi = DAG.getNode(ISD::OR, NVT,
                   DAG.getNode(Op1, NVT, InH, Amt),
                   DAG.getNode(Op2, NVT, InL, Amt2));
  return true;
}


//===----------------------------------------------------------------------===//
//  Operand Expansion
//===----------------------------------------------------------------------===//

/// ExpandOperand - This method is called when the specified operand of the
/// specified node is found to need expansion.  At this point, all of the result
/// types of the node are known to be legal, but other operands of the node may
/// need promotion or expansion as well as the specified one.
bool DAGTypeLegalizer::ExpandOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Expand node operand: "; N->dump(&DAG); cerr << "\n");
  SDOperand Res(0, 0);
  
  if (TLI.getOperationAction(N->getOpcode(), N->getValueType(0)) == 
      TargetLowering::Custom)
    Res = TLI.LowerOperation(SDOperand(N, 0), DAG);
  
  if (Res.Val == 0) {
    switch (N->getOpcode()) {
    default:
  #ifndef NDEBUG
      cerr << "ExpandOperand Op #" << OpNo << ": ";
      N->dump(&DAG); cerr << "\n";
  #endif
      assert(0 && "Do not know how to expand this operator's operand!");
      abort();
      
    case ISD::TRUNCATE:        Res = ExpandOperand_TRUNCATE(N); break;
    case ISD::BIT_CONVERT:     Res = ExpandOperand_BIT_CONVERT(N); break;

    case ISD::SINT_TO_FP:
      Res = ExpandOperand_SINT_TO_FP(N->getOperand(0), N->getValueType(0));
      break;
    case ISD::UINT_TO_FP:
      Res = ExpandOperand_UINT_TO_FP(N->getOperand(0), N->getValueType(0)); 
      break;
    case ISD::EXTRACT_ELEMENT: Res = ExpandOperand_EXTRACT_ELEMENT(N); break;
    case ISD::SETCC:           Res = ExpandOperand_SETCC(N); break;

    case ISD::STORE:
      Res = ExpandOperand_STORE(cast<StoreSDNode>(N), OpNo);
      break;
    case ISD::MEMSET:
    case ISD::MEMCPY:
    case ISD::MEMMOVE:     Res = HandleMemIntrinsic(N); break;
    }
  }
  
  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.Val) return false;
  // If the result is N, the sub-method updated N in place.  Check to see if any
  // operands are new, and if so, mark them.
  if (Res.Val == N) {
    // Mark N as new and remark N and its operands.  This allows us to correctly
    // revisit N if it needs another step of promotion and allows us to visit
    // any new operands to N.
    N->setNodeId(NewNode);
    MarkNewNodes(N);
    return true;
  }

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");
  
  ReplaceValueWith(SDOperand(N, 0), Res);
  return false;
}

SDOperand DAGTypeLegalizer::ExpandOperand_TRUNCATE(SDNode *N) {
  SDOperand InL, InH;
  GetExpandedOp(N->getOperand(0), InL, InH);
  // Just truncate the low part of the source.
  return DAG.getNode(ISD::TRUNCATE, N->getValueType(0), InL);
}

SDOperand DAGTypeLegalizer::ExpandOperand_BIT_CONVERT(SDNode *N) {
  return CreateStackStoreLoad(N->getOperand(0), N->getValueType(0));
}

SDOperand DAGTypeLegalizer::ExpandOperand_SINT_TO_FP(SDOperand Source, 
                                                     MVT::ValueType DestTy) {
  // We know the destination is legal, but that the input needs to be expanded.
  assert(Source.getValueType() == MVT::i64 && "Only handle expand from i64!");
  
  // Check to see if the target has a custom way to lower this.  If so, use it.
  switch (TLI.getOperationAction(ISD::SINT_TO_FP, Source.getValueType())) {
  default: assert(0 && "This action not implemented for this operation!");
  case TargetLowering::Legal:
  case TargetLowering::Expand:
    break;   // This case is handled below.
  case TargetLowering::Custom:
    SDOperand NV = TLI.LowerOperation(DAG.getNode(ISD::SINT_TO_FP, DestTy,
                                                  Source), DAG);
    if (NV.Val) return NV;
    break;   // The target lowered this.
  }
  
  RTLIB::Libcall LC;
  if (DestTy == MVT::f32)
    LC = RTLIB::SINTTOFP_I64_F32;
  else {
    assert(DestTy == MVT::f64 && "Unknown fp value type!");
    LC = RTLIB::SINTTOFP_I64_F64;
  }
  
  assert(0 && "FIXME: no libcalls yet!");
  abort();
#if 0
  assert(TLI.getLibcallName(LC) && "Don't know how to expand this SINT_TO_FP!");
  Source = DAG.getNode(ISD::SINT_TO_FP, DestTy, Source);
  SDOperand UnusedHiPart;
  return ExpandLibCall(TLI.getLibcallName(LC), Source.Val, true, UnusedHiPart);
#endif
}

SDOperand DAGTypeLegalizer::ExpandOperand_UINT_TO_FP(SDOperand Source, 
                                                     MVT::ValueType DestTy) {
  // We know the destination is legal, but that the input needs to be expanded.
  assert(getTypeAction(Source.getValueType()) == Expand &&
         "This is not an expansion!");
  assert(Source.getValueType() == MVT::i64 && "Only handle expand from i64!");
  
  // If this is unsigned, and not supported, first perform the conversion to
  // signed, then adjust the result if the sign bit is set.
  SDOperand SignedConv = ExpandOperand_SINT_TO_FP(Source, DestTy);

  // The 64-bit value loaded will be incorrectly if the 'sign bit' of the
  // incoming integer is set.  To handle this, we dynamically test to see if
  // it is set, and, if so, add a fudge factor.
  SDOperand Lo, Hi;
  GetExpandedOp(Source, Lo, Hi);
  
  SDOperand SignSet = DAG.getSetCC(TLI.getSetCCResultTy(), Hi,
                                   DAG.getConstant(0, Hi.getValueType()),
                                   ISD::SETLT);
  SDOperand Zero = getIntPtrConstant(0), Four = getIntPtrConstant(4);
  SDOperand CstOffset = DAG.getNode(ISD::SELECT, Zero.getValueType(),
                                    SignSet, Four, Zero);
  uint64_t FF = 0x5f800000ULL;
  if (TLI.isLittleEndian()) FF <<= 32;
  Constant *FudgeFactor = ConstantInt::get((Type*)Type::Int64Ty, FF);
  
  SDOperand CPIdx = DAG.getConstantPool(FudgeFactor, TLI.getPointerTy());
  CPIdx = DAG.getNode(ISD::ADD, TLI.getPointerTy(), CPIdx, CstOffset);
  SDOperand FudgeInReg;
  if (DestTy == MVT::f32)
    FudgeInReg = DAG.getLoad(MVT::f32, DAG.getEntryNode(), CPIdx, NULL, 0);
  else if (MVT::getSizeInBits(DestTy) > MVT::getSizeInBits(MVT::f32))
    // FIXME: Avoid the extend by construction the right constantpool?
    FudgeInReg = DAG.getExtLoad(ISD::EXTLOAD, DestTy, DAG.getEntryNode(),
                                CPIdx, NULL, 0, MVT::f32);
  else 
    assert(0 && "Unexpected conversion");
  
  return DAG.getNode(ISD::FADD, DestTy, SignedConv, FudgeInReg);
}

SDOperand DAGTypeLegalizer::ExpandOperand_EXTRACT_ELEMENT(SDNode *N) {
  SDOperand Lo, Hi;
  GetExpandedOp(N->getOperand(0), Lo, Hi);
  return cast<ConstantSDNode>(N->getOperand(1))->getValue() ? Hi : Lo;
}

SDOperand DAGTypeLegalizer::ExpandOperand_SETCC(SDNode *N) {
  SDOperand NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(2))->get();
  ExpandSetCCOperands(NewLHS, NewRHS, CCCode);
  
  // If ExpandSetCCOperands returned a scalar, use it.
  if (NewRHS.Val == 0) return NewLHS;

  // Otherwise, update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), NewLHS, NewRHS,
                                DAG.getCondCode(CCCode));
}

/// ExpandSetCCOperands - Expand the operands of a comparison.  This code is
/// shared among BR_CC, SELECT_CC, and SETCC handlers.
void DAGTypeLegalizer::ExpandSetCCOperands(SDOperand &NewLHS, SDOperand &NewRHS,
                                           ISD::CondCode &CCCode) {
  SDOperand LHSLo, LHSHi, RHSLo, RHSHi;
  GetExpandedOp(NewLHS, LHSLo, LHSHi);
  GetExpandedOp(NewRHS, RHSLo, RHSHi);
  
  MVT::ValueType VT = NewLHS.getValueType();
  if (VT == MVT::f32 || VT == MVT::f64) {
    assert(0 && "FIXME: softfp not implemented yet! should be promote not exp");
  }
  
  if (VT == MVT::ppcf128) {
    // FIXME:  This generated code sucks.  We want to generate
    //         FCMP crN, hi1, hi2
    //         BNE crN, L:
    //         FCMP crN, lo1, lo2
    // The following can be improved, but not that much.
    SDOperand Tmp1, Tmp2, Tmp3;
    Tmp1 = DAG.getSetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi, ISD::SETEQ);
    Tmp2 = DAG.getSetCC(TLI.getSetCCResultTy(), LHSLo, RHSLo, CCCode);
    Tmp3 = DAG.getNode(ISD::AND, Tmp1.getValueType(), Tmp1, Tmp2);
    Tmp1 = DAG.getSetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi, ISD::SETNE);
    Tmp2 = DAG.getSetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi, CCCode);
    Tmp1 = DAG.getNode(ISD::AND, Tmp1.getValueType(), Tmp1, Tmp2);
    NewLHS = DAG.getNode(ISD::OR, Tmp1.getValueType(), Tmp1, Tmp3);
    NewRHS = SDOperand();   // LHS is the result, not a compare.
    return;
  }
  
  
  if (CCCode == ISD::SETEQ || CCCode == ISD::SETNE) {
    if (RHSLo == RHSHi)
      if (ConstantSDNode *RHSCST = dyn_cast<ConstantSDNode>(RHSLo))
        if (RHSCST->isAllOnesValue()) {
          // Equality comparison to -1.
          NewLHS = DAG.getNode(ISD::AND, LHSLo.getValueType(), LHSLo, LHSHi);
          NewRHS = RHSLo;
          return;
        }
          
    NewLHS = DAG.getNode(ISD::XOR, LHSLo.getValueType(), LHSLo, RHSLo);
    NewRHS = DAG.getNode(ISD::XOR, LHSLo.getValueType(), LHSHi, RHSHi);
    NewLHS = DAG.getNode(ISD::OR, NewLHS.getValueType(), NewLHS, NewRHS);
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    return;
  }
  
  // If this is a comparison of the sign bit, just look at the top part.
  // X > -1,  x < 0
  if (ConstantSDNode *CST = dyn_cast<ConstantSDNode>(NewRHS))
    if ((CCCode == ISD::SETLT && CST->getValue() == 0) ||   // X < 0
        (CCCode == ISD::SETGT && CST->isAllOnesValue())) {  // X > -1
      NewLHS = LHSHi;
      NewRHS = RHSHi;
      return;
    }
      
  // FIXME: This generated code sucks.
  ISD::CondCode LowCC;
  switch (CCCode) {
  default: assert(0 && "Unknown integer setcc!");
  case ISD::SETLT:
  case ISD::SETULT: LowCC = ISD::SETULT; break;
  case ISD::SETGT:
  case ISD::SETUGT: LowCC = ISD::SETUGT; break;
  case ISD::SETLE:
  case ISD::SETULE: LowCC = ISD::SETULE; break;
  case ISD::SETGE:
  case ISD::SETUGE: LowCC = ISD::SETUGE; break;
  }
  
  // Tmp1 = lo(op1) < lo(op2)   // Always unsigned comparison
  // Tmp2 = hi(op1) < hi(op2)   // Signedness depends on operands
  // dest = hi(op1) == hi(op2) ? Tmp1 : Tmp2;
  
  // NOTE: on targets without efficient SELECT of bools, we can always use
  // this identity: (B1 ? B2 : B3) --> (B1 & B2)|(!B1&B3)
  TargetLowering::DAGCombinerInfo DagCombineInfo(DAG, false, true, NULL);
  SDOperand Tmp1, Tmp2;
  Tmp1 = TLI.SimplifySetCC(TLI.getSetCCResultTy(), LHSLo, RHSLo, LowCC,
                           false, DagCombineInfo);
  if (!Tmp1.Val)
    Tmp1 = DAG.getSetCC(TLI.getSetCCResultTy(), LHSLo, RHSLo, LowCC);
  Tmp2 = TLI.SimplifySetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi,
                           CCCode, false, DagCombineInfo);
  if (!Tmp2.Val)
    Tmp2 = DAG.getNode(ISD::SETCC, TLI.getSetCCResultTy(), LHSHi, RHSHi,
                       DAG.getCondCode(CCCode));
  
  ConstantSDNode *Tmp1C = dyn_cast<ConstantSDNode>(Tmp1.Val);
  ConstantSDNode *Tmp2C = dyn_cast<ConstantSDNode>(Tmp2.Val);
  if ((Tmp1C && Tmp1C->getValue() == 0) ||
      (Tmp2C && Tmp2C->getValue() == 0 &&
       (CCCode == ISD::SETLE || CCCode == ISD::SETGE ||
        CCCode == ISD::SETUGE || CCCode == ISD::SETULE)) ||
      (Tmp2C && Tmp2C->getValue() == 1 &&
       (CCCode == ISD::SETLT || CCCode == ISD::SETGT ||
        CCCode == ISD::SETUGT || CCCode == ISD::SETULT))) {
    // low part is known false, returns high part.
    // For LE / GE, if high part is known false, ignore the low part.
    // For LT / GT, if high part is known true, ignore the low part.
    NewLHS = Tmp2;
    NewRHS = SDOperand();
    return;
  }
  
  NewLHS = TLI.SimplifySetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi,
                             ISD::SETEQ, false, DagCombineInfo);
  if (!NewLHS.Val)
    NewLHS = DAG.getSetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi, ISD::SETEQ);
  NewLHS = DAG.getNode(ISD::SELECT, Tmp1.getValueType(),
                       NewLHS, Tmp1, Tmp2);
  NewRHS = SDOperand();
}

SDOperand DAGTypeLegalizer::ExpandOperand_STORE(StoreSDNode *N, unsigned OpNo) {
  assert(OpNo == 1 && "Can only expand the stored value so far");

  MVT::ValueType VT = N->getOperand(1).getValueType();
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  SDOperand Ch  = N->getChain();
  SDOperand Ptr = N->getBasePtr();
  int SVOffset = N->getSrcValueOffset();
  unsigned Alignment = N->getAlignment();
  bool isVolatile = N->isVolatile();
  SDOperand Lo, Hi;

  assert(!(MVT::getSizeInBits(NVT) & 7) && "Expanded type not byte sized!");

  if (!N->isTruncatingStore()) {
    unsigned IncrementSize = 0;

    // If this is a vector type, then we have to calculate the increment as
    // the product of the element size in bytes, and the number of elements
    // in the high half of the vector.
    if (MVT::isVector(N->getValue().getValueType())) {
      assert(0 && "Vectors not supported yet");
  #if 0
      SDNode *InVal = ST->getValue().Val;
      unsigned NumElems = MVT::getVectorNumElements(InVal->getValueType(0));
      MVT::ValueType EVT = MVT::getVectorElementType(InVal->getValueType(0));

      // Figure out if there is a simple type corresponding to this Vector
      // type.  If so, convert to the vector type.
      MVT::ValueType TVT = MVT::getVectorType(EVT, NumElems);
      if (TLI.isTypeLegal(TVT)) {
        // Turn this into a normal store of the vector type.
        Tmp3 = LegalizeOp(Node->getOperand(1));
        Result = DAG.getStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                              SVOffset, isVolatile, Alignment);
        Result = LegalizeOp(Result);
        break;
      } else if (NumElems == 1) {
        // Turn this into a normal store of the scalar type.
        Tmp3 = ScalarizeVectorOp(Node->getOperand(1));
        Result = DAG.getStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                              SVOffset, isVolatile, Alignment);
        // The scalarized value type may not be legal, e.g. it might require
        // promotion or expansion.  Relegalize the scalar store.
        return LegalizeOp(Result);
      } else {
        SplitVectorOp(Node->getOperand(1), Lo, Hi);
        IncrementSize = NumElems/2 * MVT::getSizeInBits(EVT)/8;
      }
  #endif
    } else {
      GetExpandedOp(N->getValue(), Lo, Hi);
      IncrementSize = Hi.Val ? MVT::getSizeInBits(Hi.getValueType())/8 : 0;

      if (!TLI.isLittleEndian())
        std::swap(Lo, Hi);
    }

    Lo = DAG.getStore(Ch, Lo, Ptr, N->getSrcValue(),
                      SVOffset, isVolatile, Alignment);

    assert(Hi.Val && "FIXME: int <-> float should be handled with promote!");
  #if 0
    if (Hi.Val == NULL) {
      // Must be int <-> float one-to-one expansion.
      return Lo;
    }
  #endif

    Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                      getIntPtrConstant(IncrementSize));
    assert(isTypeLegal(Ptr.getValueType()) && "Pointers must be legal!");
    Hi = DAG.getStore(Ch, Hi, Ptr, N->getSrcValue(), SVOffset+IncrementSize,
                      isVolatile, MinAlign(Alignment, IncrementSize));
    return DAG.getNode(ISD::TokenFactor, MVT::Other, Lo, Hi);
  } else if (MVT::getSizeInBits(N->getStoredVT()) <= MVT::getSizeInBits(NVT)) {
    GetExpandedOp(N->getValue(), Lo, Hi);
    return DAG.getTruncStore(Ch, Lo, Ptr, N->getSrcValue(), SVOffset,
                             N->getStoredVT(), isVolatile, Alignment);
  } else if (TLI.isLittleEndian()) {
    // Little-endian - low bits are at low addresses.
    GetExpandedOp(N->getValue(), Lo, Hi);

    Lo = DAG.getStore(Ch, Lo, Ptr, N->getSrcValue(), SVOffset,
                      isVolatile, Alignment);

    unsigned ExcessBits =
      MVT::getSizeInBits(N->getStoredVT()) - MVT::getSizeInBits(NVT);
    MVT::ValueType NEVT = MVT::getIntegerType(ExcessBits);

    // Increment the pointer to the other half.
    unsigned IncrementSize = MVT::getSizeInBits(NVT)/8;
    Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                      getIntPtrConstant(IncrementSize));
    Hi = DAG.getTruncStore(Ch, Hi, Ptr, N->getSrcValue(),
                           SVOffset+IncrementSize, NEVT,
                           isVolatile, MinAlign(Alignment, IncrementSize));
    return DAG.getNode(ISD::TokenFactor, MVT::Other, Lo, Hi);
  } else {
    // Big-endian - high bits are at low addresses.  Favor aligned stores at
    // the cost of some bit-fiddling.
    GetExpandedOp(N->getValue(), Lo, Hi);

    MVT::ValueType EVT = N->getStoredVT();
    unsigned EBytes = MVT::getStoreSizeInBits(EVT)/8;
    unsigned IncrementSize = MVT::getSizeInBits(NVT)/8;
    unsigned ExcessBits = (EBytes - IncrementSize)*8;
    MVT::ValueType HiVT =
      MVT::getIntegerType(MVT::getSizeInBits(EVT)-ExcessBits);

    if (ExcessBits < MVT::getSizeInBits(NVT)) {
      // Transfer high bits from the top of Lo to the bottom of Hi.
      Hi = DAG.getNode(ISD::SHL, NVT, Hi,
                       DAG.getConstant(MVT::getSizeInBits(NVT) - ExcessBits,
                                       TLI.getShiftAmountTy()));
      Hi = DAG.getNode(ISD::OR, NVT, Hi,
                       DAG.getNode(ISD::SRL, NVT, Lo,
                                   DAG.getConstant(ExcessBits,
                                                   TLI.getShiftAmountTy())));
    }

    // Store both the high bits and maybe some of the low bits.
    Hi = DAG.getTruncStore(Ch, Hi, Ptr, N->getSrcValue(),
                           SVOffset, HiVT, isVolatile, Alignment);

    // Increment the pointer to the other half.
    Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                      getIntPtrConstant(IncrementSize));
    // Store the lowest ExcessBits bits in the second half.
    Lo = DAG.getTruncStore(Ch, Lo, Ptr, N->getSrcValue(),
                           SVOffset+IncrementSize,
                           MVT::getIntegerType(ExcessBits),
                           isVolatile, MinAlign(Alignment, IncrementSize));
    return DAG.getNode(ISD::TokenFactor, MVT::Other, Lo, Hi);
  }
}

