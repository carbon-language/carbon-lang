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

unsigned
VectorTargetTransformImpl::getInstrCost(unsigned Opcode, Type *Ty1,
                                        Type *Ty2) const {
  return 1;
}

unsigned
VectorTargetTransformImpl::getBroadcastCost(Type *Tp) const {
  return 1;
}

unsigned
VectorTargetTransformImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                           unsigned Alignment,
                                           unsigned AddressSpace) const {
  return 1;
}
