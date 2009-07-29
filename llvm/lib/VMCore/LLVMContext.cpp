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
    return getConstantPointerNull(cast<PointerType>(Ty));
  case Type::StructTyID:
  case Type::ArrayTyID:
  case Type::VectorTyID:
    return getConstantAggregateZero(Ty);
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

// UndefValue accessors.
UndefValue* LLVMContext::getUndef(const Type* Ty) {
  return UndefValue::get(Ty);
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

// ConstantPointerNull accessors.
ConstantPointerNull* LLVMContext::getConstantPointerNull(const PointerType* T) {
  return ConstantPointerNull::get(T);
}

// ConstantAggregateZero accessors.
ConstantAggregateZero* LLVMContext::getConstantAggregateZero(const Type* Ty) {
  return pImpl->getConstantAggregateZero(Ty);
}

// MDNode accessors
MDNode* LLVMContext::getMDNode(Value* const* Vals, unsigned NumVals) {
  return pImpl->getMDNode(Vals, NumVals);
}

// MDString accessors
MDString* LLVMContext::getMDString(const StringRef &Str) {
  return pImpl->getMDString(Str.data(), Str.size());
}

// FunctionType accessors
FunctionType* LLVMContext::getFunctionType(const Type* Result, bool isVarArg) {
  return FunctionType::get(Result, isVarArg);
}

FunctionType* LLVMContext::getFunctionType(const Type* Result,
                                         const std::vector<const Type*>& Params,
                                         bool isVarArg) {
  return FunctionType::get(Result, Params, isVarArg);
}
                                
// IntegerType accessors
const IntegerType* LLVMContext::getIntegerType(unsigned NumBits) {
  return IntegerType::get(NumBits);
}
  
// OpaqueType accessors
OpaqueType* LLVMContext::getOpaqueType() {
  return OpaqueType::get();
}

// StructType accessors
StructType* LLVMContext::getStructType(bool isPacked) {
  return StructType::get(isPacked);
}

StructType* LLVMContext::getStructType(const std::vector<const Type*>& Params,
                                       bool isPacked) {
  return StructType::get(Params, isPacked);
}

StructType *LLVMContext::getStructType(const Type *type, ...) {
  va_list ap;
  std::vector<const llvm::Type*> StructFields;
  va_start(ap, type);
  while (type) {
    StructFields.push_back(type);
    type = va_arg(ap, llvm::Type*);
  }
  return StructType::get(StructFields);
}

// ArrayType accessors
ArrayType* LLVMContext::getArrayType(const Type* ElementType,
                                     uint64_t NumElements) {
  return ArrayType::get(ElementType, NumElements);
}
  
// PointerType accessors
PointerType* LLVMContext::getPointerType(const Type* ElementType,
                                         unsigned AddressSpace) {
  return PointerType::get(ElementType, AddressSpace);
}

PointerType* LLVMContext::getPointerTypeUnqual(const Type* ElementType) {
  return PointerType::getUnqual(ElementType);
}
  
// VectorType accessors
VectorType* LLVMContext::getVectorType(const Type* ElementType,
                                       unsigned NumElements) {
  return VectorType::get(ElementType, NumElements);
}

VectorType* LLVMContext::getVectorTypeInteger(const VectorType* VTy) {
  return VectorType::getInteger(VTy);  
}

VectorType* LLVMContext::getVectorTypeExtendedElement(const VectorType* VTy) {
  return VectorType::getExtendedElementVectorType(VTy);
}

VectorType* LLVMContext::getVectorTypeTruncatedElement(const VectorType* VTy) {
  return VectorType::getTruncatedElementVectorType(VTy);
}

const Type* LLVMContext::makeCmpResultType(const Type* opnd_type) {
  if (const VectorType* vt = dyn_cast<const VectorType>(opnd_type)) {
    return getVectorType(Type::Int1Ty, vt->getNumElements());
  }
  return Type::Int1Ty;
}

void LLVMContext::erase(MDString *M) {
  pImpl->erase(M);
}

void LLVMContext::erase(MDNode *M) {
  pImpl->erase(M);
}

void LLVMContext::erase(ConstantAggregateZero *Z) {
  pImpl->erase(Z);
}
