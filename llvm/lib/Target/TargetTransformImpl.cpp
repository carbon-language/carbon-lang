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
static int InstructionOpcodeToISD(unsigned Opcode) {
  enum InstructionOpcodes {
#define HANDLE_INST(NUM, OPCODE, CLASS) OPCODE = NUM,
#define LAST_OTHER_INST(NUM) InstructionOpcodesCount = NUM
#include "llvm/Instruction.def"
  };
  switch (static_cast<InstructionOpcodes>(Opcode)) {
  case Ret:            return 0;
  case Br:             return 0;
  case Switch:         return 0;
  case IndirectBr:     return 0;
  case Invoke:         return 0;
  case Resume:         return 0;
  case Unreachable:    return 0;
  case Add:            return ISD::ADD;
  case FAdd:           return ISD::FADD;
  case Sub:            return ISD::SUB;
  case FSub:           return ISD::FSUB;
  case Mul:            return ISD::MUL;
  case FMul:           return ISD::FMUL;
  case UDiv:           return ISD::UDIV;
  case SDiv:           return ISD::UDIV;
  case FDiv:           return ISD::FDIV;
  case URem:           return ISD::UREM;
  case SRem:           return ISD::SREM;
  case FRem:           return ISD::FREM;
  case Shl:            return ISD::SHL;
  case LShr:           return ISD::SRL;
  case AShr:           return ISD::SRA;
  case And:            return ISD::AND;
  case Or:             return ISD::OR;
  case Xor:            return ISD::XOR;
  case Alloca:         return 0;
  case Load:           return ISD::LOAD;
  case Store:          return ISD::STORE;
  case GetElementPtr:  return 0;
  case Fence:          return 0;
  case AtomicCmpXchg:  return 0;
  case AtomicRMW:      return 0;
  case Trunc:          return ISD::TRUNCATE;
  case ZExt:           return ISD::ZERO_EXTEND;
  case SExt:           return ISD::SEXTLOAD;
  case FPToUI:         return ISD::FP_TO_UINT;
  case FPToSI:         return ISD::FP_TO_SINT;
  case UIToFP:         return ISD::UINT_TO_FP;
  case SIToFP:         return ISD::SINT_TO_FP;
  case FPTrunc:        return ISD::FP_ROUND;
  case FPExt:          return ISD::FP_EXTEND;
  case PtrToInt:       return ISD::BITCAST;
  case IntToPtr:       return ISD::BITCAST;
  case BitCast:        return ISD::BITCAST;
  case ICmp:           return ISD::SETCC;
  case FCmp:           return ISD::SETCC;
  case PHI:            return 0;
  case Call:           return 0;
  case Select:         return ISD::SELECT;
  case UserOp1:        return 0;
  case UserOp2:        return 0;
  case VAArg:          return 0;
  case ExtractElement: return ISD::EXTRACT_VECTOR_ELT;
  case InsertElement:  return ISD::INSERT_VECTOR_ELT;
  case ShuffleVector:  return ISD::VECTOR_SHUFFLE;
  case ExtractValue:   return ISD::MERGE_VALUES;
  case InsertValue:    return ISD::MERGE_VALUES;
  case LandingPad:     return 0;
  }

  llvm_unreachable("Unknown instruction type encountered!");
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

  // If we don't have any information about this instruction assume it costs 1.
  if (ISD == 0)
    return 1;

  // Selects on vectors are actually vector selects.
  if (ISD == ISD::SELECT) {
    assert(Ty2 && "Ty2 must hold the condition type");
    if (Ty2->isVectorTy())
    ISD = ISD::VSELECT;
  }

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

unsigned
VectorTargetTransformImpl::getNumberOfParts(Type *Tp) const {
  std::pair<unsigned, EVT>  LT =
  getTypeLegalizationCost(Tp->getContext(), TLI->getValueType(Tp));
  return LT.first;
}

