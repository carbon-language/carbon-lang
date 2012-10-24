// llvm/Target/TargetTransformImpl.cpp - Target Loop Trans Info ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetTransformImpl.h"
#include "llvm/Target/TargetLowering.h"
#include <utility>

using namespace llvm;

//===----------------------------------------------------------------------===//
//
// Calls used by scalar transformations.
//
//===----------------------------------------------------------------------===//

bool ScalarTargetTransformImpl::isLegalAddImmediate(int64_t imm) const {
  return TLI->isLegalAddImmediate(imm);
}

bool ScalarTargetTransformImpl::isLegalICmpImmediate(int64_t imm) const {
  return TLI->isLegalICmpImmediate(imm);
}

bool ScalarTargetTransformImpl::isLegalAddressingMode(const AddrMode &AM,
                                                    Type *Ty) const {
  return TLI->isLegalAddressingMode(AM, Ty);
}

bool ScalarTargetTransformImpl::isTruncateFree(Type *Ty1, Type *Ty2) const {
  return TLI->isTruncateFree(Ty1, Ty2);
}

bool ScalarTargetTransformImpl::isTypeLegal(Type *Ty) const {
  EVT T = TLI->getValueType(Ty);
  return TLI->isTypeLegal(T);
}

unsigned ScalarTargetTransformImpl::getJumpBufAlignment() const {
  return TLI->getJumpBufAlignment();
}

unsigned ScalarTargetTransformImpl::getJumpBufSize() const {
  return TLI->getJumpBufSize();
}

//===----------------------------------------------------------------------===//
//
// Calls used by the vectorizers.
//
//===----------------------------------------------------------------------===//
int InstructionOpcodeToISD(unsigned Opcode) {
  static const int OpToISDTbl[] = {
    /*Instruction::Ret           */ 0, // Opcode numbering start at #1.
    /*Instruction::Br            */ 0,
    /*Instruction::Switch        */ 0,
    /*Instruction::IndirectBr    */ 0,
    /*Instruction::Invoke        */ 0,
    /*Instruction::Resume        */ 0,
    /*Instruction::Unreachable   */ 0,
    /*Instruction::Add           */ ISD::ADD,
    /*Instruction::FAdd          */ ISD::FADD,
    /*Instruction::Sub           */ ISD::SUB,
    /*Instruction::FSub          */ ISD::FSUB,
    /*Instruction::Mul           */ ISD::MUL,
    /*Instruction::FMul          */ ISD::FMUL,
    /*Instruction::UDiv          */ ISD::UDIV,
    /*Instruction::SDiv          */ ISD::UDIV,
    /*Instruction::FDiv          */ ISD::FDIV,
    /*Instruction::URem          */ ISD::UREM,
    /*Instruction::SRem          */ ISD::SREM,
    /*Instruction::FRem          */ ISD::FREM,
    /*Instruction::Shl           */ ISD::SHL,
    /*Instruction::LShr          */ ISD::SRL,
    /*Instruction::AShr          */ ISD::SRA,
    /*Instruction::And           */ ISD::AND,
    /*Instruction::Or            */ ISD::OR,
    /*Instruction::Xor           */ ISD::XOR,
    /*Instruction::Alloca        */ 0,
    /*Instruction::Load          */ ISD::LOAD,
    /*Instruction::Store         */ ISD::STORE,
    /*Instruction::GetElementPtr */ 0,
    /*Instruction::Fence         */ 0,
    /*Instruction::AtomicCmpXchg */ 0,
    /*Instruction::AtomicRMW     */ 0,
    /*Instruction::Trunc         */ ISD::TRUNCATE,
    /*Instruction::ZExt          */ ISD::ZERO_EXTEND,
    /*Instruction::SExt          */ ISD::SEXTLOAD,
    /*Instruction::FPToUI        */ ISD::FP_TO_UINT,
    /*Instruction::FPToSI        */ ISD::FP_TO_SINT,
    /*Instruction::UIToFP        */ ISD::UINT_TO_FP,
    /*Instruction::SIToFP        */ ISD::SINT_TO_FP,
    /*Instruction::FPTrunc       */ ISD::FP_ROUND,
    /*Instruction::FPExt         */ ISD::FP_EXTEND,
    /*Instruction::PtrToInt      */ ISD::BITCAST,
    /*Instruction::IntToPtr      */ ISD::BITCAST,
    /*Instruction::BitCast       */ ISD::BITCAST,
    /*Instruction::ICmp          */ ISD::SETCC,
    /*Instruction::FCmp          */ ISD::SETCC,
    /*Instruction::PHI           */ 0,
    /*Instruction::Call          */ 0,
    /*Instruction::Select        */ ISD::SELECT,
    /*Instruction::UserOp1       */ 0,
    /*Instruction::UserOp2       */ 0,
    /*Instruction::VAArg         */ 0,
    /*Instruction::ExtractElement*/ ISD::EXTRACT_VECTOR_ELT,
    /*Instruction::InsertElement */ ISD::INSERT_VECTOR_ELT,
    /*Instruction::ShuffleVector */ ISD::VECTOR_SHUFFLE,
    /*Instruction::ExtractValue  */ ISD::MERGE_VALUES,
    /*Instruction::InsertValue   */ ISD::MERGE_VALUES,
    /*Instruction::LandingPad    */ 0};

  assert((Instruction::Ret == 1) && (Instruction::LandingPad == 58) &&
         "Instruction order had changed");

  // Opcode numbering starts at #1 but the table starts at #0, so we subtract
  // one from the opcode number.
  return OpToISDTbl[Opcode - 1];
}

std::pair<unsigned, EVT> 
VectorTargetTransformImpl::getTypeLegalizationCost(LLVMContext &C,
                                                         EVT Ty) const {
  unsigned Cost = 1;
  // We keep legalizing the type until we find a legal kind. We assume that
  // the only operation that costs anything is the split. After splitting
  // we need to handle two types.
  while (true) {
    TargetLowering::LegalizeKind LK = TLI->getTypeConversion(C, Ty);

    if (LK.first == TargetLowering::TypeLegal)
      return std::make_pair(Cost, LK.second);

    if (LK.first == TargetLowering::TypeSplitVector)
      Cost *= 2;

    // Keep legalizing the type.
    Ty = LK.second;
  }
}

unsigned
VectorTargetTransformImpl::getInstrCost(unsigned Opcode, Type *Ty1,
                                        Type *Ty2) const {
  // Check if any of the operands are vector operands.
  int ISD = InstructionOpcodeToISD(Opcode);

  // Selects on vectors are actually vector selects.
  if (ISD == ISD::SELECT) {
    assert(Ty2 && "Ty2 must hold the select type");
    if (Ty2->isVectorTy())
    ISD = ISD::VSELECT;
  }

  // If we don't have any information about this instruction assume it costs 1.
  if (ISD == 0)
    return 1;

  assert(Ty1 && "We need to have at least one type");

  // From this stage we look at the legalized type.
  std::pair<unsigned, EVT>  LT =
  getTypeLegalizationCost(Ty1->getContext(), TLI->getValueType(Ty1));

  if (TLI->isOperationLegalOrCustom(ISD, LT.second)) {
    // The operation is legal. Assume it costs 1. Multiply
    // by the type-legalization overhead.
    return LT.first * 1;
  }

  unsigned NumElem =
  (LT.second.isVector() ? LT.second.getVectorNumElements() : 1);

  // We will probably scalarize this instruction. Assume that the cost is the
  // number of the vector elements.
  return LT.first * NumElem * 1;
}

unsigned
VectorTargetTransformImpl::getBroadcastCost(Type *Tp) const {
  return 1;
}

unsigned
VectorTargetTransformImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                           unsigned Alignment,
                                           unsigned AddressSpace) const {
  // From this stage we look at the legalized type.
  std::pair<unsigned, EVT>  LT =
  getTypeLegalizationCost(Src->getContext(), TLI->getValueType(Src));
  // Assume that all loads of legal types cost 1.
  return LT.first;
}
