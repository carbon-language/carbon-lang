//===-- LLVMContext.cpp - Implement LLVMContext -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContext, as a wrapper around the opaque
// class LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instruction.h"
#include "llvm/Metadata.h"
#include "llvm/Support/ManagedStatic.h"
#include "LLVMContextImpl.h"
#include <cstdarg>

using namespace llvm;

static ManagedStatic<LLVMContext> GlobalContext;

LLVMContext& llvm::getGlobalContext() {
  return *GlobalContext;
}

LLVMContext::LLVMContext() : pImpl(new LLVMContextImpl(*this)) { }
LLVMContext::~LLVMContext() { delete pImpl; }

// Constant accessors

// Constructor to create a '0' constant of arbitrary type...
static const uint64_t zero[2] = {0, 0};
Constant* LLVMContext::getNullValue(const Type* Ty) {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID:
    return ConstantInt::get(Ty, 0);
  case Type::FloatTyID:
    return ConstantFP::get(Ty->getContext(), APFloat(APInt(32, 0)));
  case Type::DoubleTyID:
    return ConstantFP::get(Ty->getContext(), APFloat(APInt(64, 0)));
  case Type::X86_FP80TyID:
    return ConstantFP::get(Ty->getContext(), APFloat(APInt(80, 2, zero)));
  case Type::FP128TyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat(APInt(128, 2, zero), true));
  case Type::PPC_FP128TyID:
    return ConstantFP::get(Ty->getContext(), APFloat(APInt(128, 2, zero)));
  case Type::PointerTyID:
    return ConstantPointerNull::get(cast<PointerType>(Ty));
  case Type::StructTyID:
  case Type::ArrayTyID:
  case Type::VectorTyID:
    return ConstantAggregateZero::get(Ty);
  default:
    // Function, Label, or Opaque type?
    assert(!"Cannot create a null constant of that type!");
    return 0;
  }
}

Constant* LLVMContext::getAllOnesValue(const Type* Ty) {
  if (const IntegerType* ITy = dyn_cast<IntegerType>(Ty))
    return ConstantInt::get(*this, APInt::getAllOnesValue(ITy->getBitWidth()));
  
  std::vector<Constant*> Elts;
  const VectorType* VTy = cast<VectorType>(Ty);
  Elts.resize(VTy->getNumElements(), getAllOnesValue(VTy->getElementType()));
  assert(Elts[0] && "Not a vector integer type!");
  return cast<ConstantVector>(ConstantVector::get(Elts));
}

// ConstantInt accessors.
ConstantInt* LLVMContext::getTrue() {
  assert(this && "Context not initialized!");
  assert(pImpl && "Context not initialized!");
  return pImpl->getTrue();
}

ConstantInt* LLVMContext::getFalse() {
  assert(this && "Context not initialized!");
  assert(pImpl && "Context not initialized!");
  return pImpl->getFalse();
}

// MDNode accessors
MDNode* LLVMContext::getMDNode(Value* const* Vals, unsigned NumVals) {
  return pImpl->getMDNode(Vals, NumVals);
}

// MDString accessors
MDString* LLVMContext::getMDString(const StringRef &Str) {
  return pImpl->getMDString(Str.data(), Str.size());
}

void LLVMContext::erase(MDString *M) {
  pImpl->erase(M);
}

void LLVMContext::erase(MDNode *M) {
  pImpl->erase(M);
}
