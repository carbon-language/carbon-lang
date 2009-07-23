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
#include "llvm/MDNode.h"
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
    return getConstantInt(Ty, 0);
  case Type::FloatTyID:
    return getConstantFP(APFloat(APInt(32, 0)));
  case Type::DoubleTyID:
    return getConstantFP(APFloat(APInt(64, 0)));
  case Type::X86_FP80TyID:
    return getConstantFP(APFloat(APInt(80, 2, zero)));
  case Type::FP128TyID:
    return getConstantFP(APFloat(APInt(128, 2, zero), true));
  case Type::PPC_FP128TyID:
    return getConstantFP(APFloat(APInt(128, 2, zero)));
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
    return getConstantInt(APInt::getAllOnesValue(ITy->getBitWidth()));
  
  std::vector<Constant*> Elts;
  const VectorType* VTy = cast<VectorType>(Ty);
  Elts.resize(VTy->getNumElements(), getAllOnesValue(VTy->getElementType()));
  assert(Elts[0] && "Not a vector integer type!");
  return cast<ConstantVector>(getConstantVector(Elts));
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

Constant* LLVMContext::getConstantInt(const Type* Ty, uint64_t V,
                                         bool isSigned) {
  Constant *C = getConstantInt(cast<IntegerType>(Ty->getScalarType()),
                               V, isSigned);

  // For vectors, broadcast the value.
  if (const VectorType *VTy = dyn_cast<VectorType>(Ty))
    return
      getConstantVector(std::vector<Constant *>(VTy->getNumElements(), C));

  return C;
}


ConstantInt* LLVMContext::getConstantInt(const IntegerType* Ty, uint64_t V,
                                         bool isSigned) {
  return getConstantInt(APInt(Ty->getBitWidth(), V, isSigned));
}

ConstantInt* LLVMContext::getConstantIntSigned(const IntegerType* Ty,
                                               int64_t V) {
  return getConstantInt(Ty, V, true);
}

Constant *LLVMContext::getConstantIntSigned(const Type *Ty, int64_t V) {
  return getConstantInt(Ty, V, true);
}

ConstantInt* LLVMContext::getConstantInt(const APInt& V) {
  return pImpl->getConstantInt(V);
}

Constant* LLVMContext::getConstantInt(const Type* Ty, const APInt& V) {
  ConstantInt *C = getConstantInt(V);
  assert(C->getType() == Ty->getScalarType() &&
         "ConstantInt type doesn't match the type implied by its value!");

  // For vectors, broadcast the value.
  if (const VectorType *VTy = dyn_cast<VectorType>(Ty))
    return
      getConstantVector(std::vector<Constant *>(VTy->getNumElements(), C));

  return C;
}

// ConstantPointerNull accessors.
ConstantPointerNull* LLVMContext::getConstantPointerNull(const PointerType* T) {
  return ConstantPointerNull::get(T);
}


// ConstantStruct accessors.
Constant* LLVMContext::getConstantStruct(const StructType* T,
                                         const std::vector<Constant*>& V) {
  return ConstantStruct::get(T, V);
}

Constant* LLVMContext::getConstantStruct(const std::vector<Constant*>& V,
                                         bool packed) {
  std::vector<const Type*> StructEls;
  StructEls.reserve(V.size());
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    StructEls.push_back(V[i]->getType());
  return getConstantStruct(getStructType(StructEls, packed), V);
}

Constant* LLVMContext::getConstantStruct(Constant* const *Vals,
                                         unsigned NumVals, bool Packed) {
  // FIXME: make this the primary ctor method.
  return getConstantStruct(std::vector<Constant*>(Vals, Vals+NumVals), Packed);
}


// ConstantAggregateZero accessors.
ConstantAggregateZero* LLVMContext::getConstantAggregateZero(const Type* Ty) {
  return pImpl->getConstantAggregateZero(Ty);
}


// ConstantArray accessors.
Constant* LLVMContext::getConstantArray(const ArrayType* T,
                                        const std::vector<Constant*>& V) {
  return pImpl->getConstantArray(T, V);
}

Constant* LLVMContext::getConstantArray(const ArrayType* T,
                                        Constant* const* Vals,
                                        unsigned NumVals) {
  // FIXME: make this the primary ctor method.
  return getConstantArray(T, std::vector<Constant*>(Vals, Vals+NumVals));
}

/// ConstantArray::get(const string&) - Return an array that is initialized to
/// contain the specified string.  If length is zero then a null terminator is 
/// added to the specified string so that it may be used in a natural way. 
/// Otherwise, the length parameter specifies how much of the string to use 
/// and it won't be null terminated.
///
Constant* LLVMContext::getConstantArray(const std::string& Str,
                                        bool AddNull) {
  std::vector<Constant*> ElementVals;
  for (unsigned i = 0; i < Str.length(); ++i)
    ElementVals.push_back(getConstantInt(Type::Int8Ty, Str[i]));

  // Add a null terminator to the string...
  if (AddNull) {
    ElementVals.push_back(getConstantInt(Type::Int8Ty, 0));
  }

  ArrayType *ATy = getArrayType(Type::Int8Ty, ElementVals.size());
  return getConstantArray(ATy, ElementVals);
}


// ConstantExpr accessors.
Constant* LLVMContext::getConstantExpr(unsigned Opcode, Constant* C1,
                                       Constant* C2) {
  return ConstantExpr::get(Opcode, C1, C2);
}

Constant* LLVMContext::getConstantExprTrunc(Constant* C, const Type* Ty) {
  return ConstantExpr::getTrunc(C, Ty);
}

Constant* LLVMContext::getConstantExprSExt(Constant* C, const Type* Ty) {
  return ConstantExpr::getSExt(C, Ty);
}

Constant* LLVMContext::getConstantExprZExt(Constant* C, const Type* Ty) {
  return ConstantExpr::getZExt(C, Ty);  
}

Constant* LLVMContext::getConstantExprFPTrunc(Constant* C, const Type* Ty) {
  return ConstantExpr::getFPTrunc(C, Ty);
}

Constant* LLVMContext::getConstantExprFPExtend(Constant* C, const Type* Ty) {
  return ConstantExpr::getFPExtend(C, Ty);
}

Constant* LLVMContext::getConstantExprUIToFP(Constant* C, const Type* Ty) {
  return ConstantExpr::getUIToFP(C, Ty);
}

Constant* LLVMContext::getConstantExprSIToFP(Constant* C, const Type* Ty) {
  return ConstantExpr::getSIToFP(C, Ty);
}

Constant* LLVMContext::getConstantExprFPToUI(Constant* C, const Type* Ty) {
  return ConstantExpr::getFPToUI(C, Ty);
}

Constant* LLVMContext::getConstantExprFPToSI(Constant* C, const Type* Ty) {
  return ConstantExpr::getFPToSI(C, Ty);
}

Constant* LLVMContext::getConstantExprPtrToInt(Constant* C, const Type* Ty) {
  return ConstantExpr::getPtrToInt(C, Ty);
}

Constant* LLVMContext::getConstantExprIntToPtr(Constant* C, const Type* Ty) {
  return ConstantExpr::getIntToPtr(C, Ty);
}

Constant* LLVMContext::getConstantExprBitCast(Constant* C, const Type* Ty) {
  return ConstantExpr::getBitCast(C, Ty);
}

Constant* LLVMContext::getConstantExprCast(unsigned ops, Constant* C,
                                           const Type* Ty) {
  return ConstantExpr::getCast(ops, C, Ty);
}

Constant* LLVMContext::getConstantExprZExtOrBitCast(Constant* C,
                                                    const Type* Ty) {
  return ConstantExpr::getZExtOrBitCast(C, Ty);
}

Constant* LLVMContext::getConstantExprSExtOrBitCast(Constant* C,
                                                    const Type* Ty) {
  return ConstantExpr::getSExtOrBitCast(C, Ty);
}

Constant* LLVMContext::getConstantExprTruncOrBitCast(Constant* C,
                                                     const Type* Ty) {
  return ConstantExpr::getTruncOrBitCast(C, Ty);  
}

Constant* LLVMContext::getConstantExprPointerCast(Constant* C, const Type* Ty) {
  return ConstantExpr::getPointerCast(C, Ty);
}

Constant* LLVMContext::getConstantExprIntegerCast(Constant* C, const Type* Ty,
                                                  bool isSigned) {
  return ConstantExpr::getIntegerCast(C, Ty, isSigned);
}

Constant* LLVMContext::getConstantExprFPCast(Constant* C, const Type* Ty) {
  return ConstantExpr::getFPCast(C, Ty);
}

Constant* LLVMContext::getConstantExprSelect(Constant* C, Constant* V1,
                                             Constant* V2) {
  return ConstantExpr::getSelect(C, V1, V2);
}

Constant* LLVMContext::getConstantExprAlignOf(const Type* Ty) {
  // alignof is implemented as: (i64) gep ({i8,Ty}*)null, 0, 1
  const Type *AligningTy = getStructType(Type::Int8Ty, Ty, NULL);
  Constant *NullPtr = getNullValue(AligningTy->getPointerTo());
  Constant *Zero = getConstantInt(Type::Int32Ty, 0);
  Constant *One = getConstantInt(Type::Int32Ty, 1);
  Constant *Indices[2] = { Zero, One };
  Constant *GEP = getConstantExprGetElementPtr(NullPtr, Indices, 2);
  return getConstantExprCast(Instruction::PtrToInt, GEP, Type::Int32Ty);
}

Constant* LLVMContext::getConstantExprCompare(unsigned short pred,
                                 Constant* C1, Constant* C2) {
  return ConstantExpr::getCompare(pred, C1, C2);
}

Constant* LLVMContext::getConstantExprNeg(Constant* C) {
  // API compatibility: Adjust integer opcodes to floating-point opcodes.
  if (C->getType()->isFPOrFPVector())
    return getConstantExprFNeg(C);
  assert(C->getType()->isIntOrIntVector() &&
         "Cannot NEG a nonintegral value!");
  return getConstantExpr(Instruction::Sub,
             getZeroValueForNegation(C->getType()),
             C);
}

Constant* LLVMContext::getConstantExprFNeg(Constant* C) {
  assert(C->getType()->isFPOrFPVector() &&
         "Cannot FNEG a non-floating-point value!");
  return getConstantExpr(Instruction::FSub,
             getZeroValueForNegation(C->getType()),
             C);
}

Constant* LLVMContext::getConstantExprNot(Constant* C) {
  assert(C->getType()->isIntOrIntVector() &&
         "Cannot NOT a nonintegral value!");
  return getConstantExpr(Instruction::Xor, C, getAllOnesValue(C->getType()));
}

Constant* LLVMContext::getConstantExprAdd(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::Add, C1, C2);
}

Constant* LLVMContext::getConstantExprFAdd(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::FAdd, C1, C2);
}

Constant* LLVMContext::getConstantExprSub(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::Sub, C1, C2);
}

Constant* LLVMContext::getConstantExprFSub(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::FSub, C1, C2);
}

Constant* LLVMContext::getConstantExprMul(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::Mul, C1, C2);
}

Constant* LLVMContext::getConstantExprFMul(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::FMul, C1, C2);
}

Constant* LLVMContext::getConstantExprUDiv(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::UDiv, C1, C2);
}

Constant* LLVMContext::getConstantExprSDiv(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::SDiv, C1, C2);
}

Constant* LLVMContext::getConstantExprFDiv(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::FDiv, C1, C2);
}

Constant* LLVMContext::getConstantExprURem(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::URem, C1, C2);
}

Constant* LLVMContext::getConstantExprSRem(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::SRem, C1, C2);
}

Constant* LLVMContext::getConstantExprFRem(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::FRem, C1, C2);
}

Constant* LLVMContext::getConstantExprAnd(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::And, C1, C2);
}

Constant* LLVMContext::getConstantExprOr(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::Or, C1, C2);
}

Constant* LLVMContext::getConstantExprXor(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::Xor, C1, C2);
}

Constant* LLVMContext::getConstantExprICmp(unsigned short pred, Constant* LHS,
                              Constant* RHS) {
  return ConstantExpr::getICmp(pred, LHS, RHS);
}

Constant* LLVMContext::getConstantExprFCmp(unsigned short pred, Constant* LHS,
                              Constant* RHS) {
  return ConstantExpr::getFCmp(pred, LHS, RHS);
}

Constant* LLVMContext::getConstantExprShl(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::Shl, C1, C2);
}

Constant* LLVMContext::getConstantExprLShr(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::LShr, C1, C2);
}

Constant* LLVMContext::getConstantExprAShr(Constant* C1, Constant* C2) {
  return getConstantExpr(Instruction::AShr, C1, C2);
}

Constant* LLVMContext::getConstantExprGetElementPtr(Constant* C,
                                                    Constant* const* IdxList, 
                                                    unsigned NumIdx) {
  return ConstantExpr::getGetElementPtr(C, IdxList, NumIdx);
}

Constant* LLVMContext::getConstantExprGetElementPtr(Constant* C,
                                                    Value* const* IdxList, 
                                                    unsigned NumIdx) {
  return ConstantExpr::getGetElementPtr(C, IdxList, NumIdx);
}

Constant* LLVMContext::getConstantExprExtractElement(Constant* Vec,
                                                     Constant* Idx) {
  return ConstantExpr::getExtractElement(Vec, Idx);
}

Constant* LLVMContext::getConstantExprInsertElement(Constant* Vec,
                                                    Constant* Elt,
                                                    Constant* Idx) {
  return ConstantExpr::getInsertElement(Vec, Elt, Idx);
}

Constant* LLVMContext::getConstantExprShuffleVector(Constant* V1, Constant* V2,
                                                    Constant* Mask) {
  return ConstantExpr::getShuffleVector(V1, V2, Mask);
}

Constant* LLVMContext::getConstantExprExtractValue(Constant* Agg,
                                                   const unsigned* IdxList, 
                                                   unsigned NumIdx) {
  return ConstantExpr::getExtractValue(Agg, IdxList, NumIdx);
}

Constant* LLVMContext::getConstantExprInsertValue(Constant* Agg, Constant* Val,
                                                  const unsigned* IdxList,
                                                  unsigned NumIdx) {
  return ConstantExpr::getInsertValue(Agg, Val, IdxList, NumIdx);
}

Constant* LLVMContext::getConstantExprSizeOf(const Type* Ty) {
  // sizeof is implemented as: (i64) gep (Ty*)null, 1
  Constant *GEPIdx = getConstantInt(Type::Int32Ty, 1);
  Constant *GEP = getConstantExprGetElementPtr(
                            getNullValue(getPointerTypeUnqual(Ty)), &GEPIdx, 1);
  return getConstantExprCast(Instruction::PtrToInt, GEP, Type::Int64Ty);
}

Constant* LLVMContext::getZeroValueForNegation(const Type* Ty) {
  if (const VectorType *PTy = dyn_cast<VectorType>(Ty))
    if (PTy->getElementType()->isFloatingPoint()) {
      std::vector<Constant*> zeros(PTy->getNumElements(),
                           getConstantFPNegativeZero(PTy->getElementType()));
      return getConstantVector(PTy, zeros);
    }

  if (Ty->isFloatingPoint()) 
    return getConstantFPNegativeZero(Ty);

  return getNullValue(Ty);
}


// ConstantFP accessors.
ConstantFP* LLVMContext::getConstantFP(const APFloat& V) {
  return pImpl->getConstantFP(V);
}

static const fltSemantics *TypeToFloatSemantics(const Type *Ty) {
  if (Ty == Type::FloatTy)
    return &APFloat::IEEEsingle;
  if (Ty == Type::DoubleTy)
    return &APFloat::IEEEdouble;
  if (Ty == Type::X86_FP80Ty)
    return &APFloat::x87DoubleExtended;
  else if (Ty == Type::FP128Ty)
    return &APFloat::IEEEquad;
  
  assert(Ty == Type::PPC_FP128Ty && "Unknown FP format");
  return &APFloat::PPCDoubleDouble;
}

/// get() - This returns a constant fp for the specified value in the
/// specified type.  This should only be used for simple constant values like
/// 2.0/1.0 etc, that are known-valid both as double and as the target format.
Constant* LLVMContext::getConstantFP(const Type* Ty, double V) {
  APFloat FV(V);
  bool ignored;
  FV.convert(*TypeToFloatSemantics(Ty->getScalarType()),
             APFloat::rmNearestTiesToEven, &ignored);
  Constant *C = getConstantFP(FV);

  // For vectors, broadcast the value.
  if (const VectorType *VTy = dyn_cast<VectorType>(Ty))
    return
      getConstantVector(std::vector<Constant *>(VTy->getNumElements(), C));

  return C;
}

ConstantFP* LLVMContext::getConstantFPNegativeZero(const Type* Ty) {
  APFloat apf = cast <ConstantFP>(getNullValue(Ty))->getValueAPF();
  apf.changeSign();
  return getConstantFP(apf);
}


// ConstantVector accessors.
Constant* LLVMContext::getConstantVector(const VectorType* T,
                            const std::vector<Constant*>& V) {
  return ConstantVector::get(T, V);
}

Constant* LLVMContext::getConstantVector(const std::vector<Constant*>& V) {
  assert(!V.empty() && "Cannot infer type if V is empty");
  return getConstantVector(getVectorType(V.front()->getType(),V.size()), V);
}

Constant* LLVMContext::getConstantVector(Constant* const* Vals,
                                         unsigned NumVals) {
  // FIXME: make this the primary ctor method.
  return getConstantVector(std::vector<Constant*>(Vals, Vals+NumVals));
}

// MDNode accessors
MDNode* LLVMContext::getMDNode(Value* const* Vals, unsigned NumVals) {
  return pImpl->getMDNode(Vals, NumVals);
}

// MDString accessors
MDString* LLVMContext::getMDString(const char *StrBegin, unsigned StrLength) {
  return pImpl->getMDString(StrBegin, StrLength);
}

MDString* LLVMContext::getMDString(const std::string &Str) {
  return getMDString(Str.data(), Str.size());
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

void LLVMContext::erase(ConstantArray *C) {
  pImpl->erase(C);
}

Constant *LLVMContext::replaceUsesOfWithOnConstant(ConstantArray *CA,
                                               Value *From, Value *To, Use *U) {
  return pImpl->replaceUsesOfWithOnConstant(CA, From, To, U);
}
