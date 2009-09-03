//===-- Constants.cpp - Implement Constant nodes --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Constant* classes...
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "LLVMContextImpl.h"
#include "ConstantFold.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Operator.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Mutex.h"
#include "llvm/System/RWMutex.h"
#include "llvm/System/Threading.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <map>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                              Constant Class
//===----------------------------------------------------------------------===//

// Constructor to create a '0' constant of arbitrary type...
static const uint64_t zero[2] = {0, 0};
Constant* Constant::getNullValue(const Type* Ty) {
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

Constant* Constant::getIntegerValue(const Type* Ty, const APInt &V) {
  const Type *ScalarTy = Ty->getScalarType();

  // Create the base integer constant.
  Constant *C = ConstantInt::get(Ty->getContext(), V);

  // Convert an integer to a pointer, if necessary.
  if (const PointerType *PTy = dyn_cast<PointerType>(ScalarTy))
    C = ConstantExpr::getIntToPtr(C, PTy);

  // Broadcast a scalar to a vector, if necessary.
  if (const VectorType *VTy = dyn_cast<VectorType>(Ty))
    C = ConstantVector::get(std::vector<Constant *>(VTy->getNumElements(), C));

  return C;
}

Constant* Constant::getAllOnesValue(const Type* Ty) {
  if (const IntegerType* ITy = dyn_cast<IntegerType>(Ty))
    return ConstantInt::get(Ty->getContext(),
                            APInt::getAllOnesValue(ITy->getBitWidth()));
  
  std::vector<Constant*> Elts;
  const VectorType* VTy = cast<VectorType>(Ty);
  Elts.resize(VTy->getNumElements(), getAllOnesValue(VTy->getElementType()));
  assert(Elts[0] && "Not a vector integer type!");
  return cast<ConstantVector>(ConstantVector::get(Elts));
}

void Constant::destroyConstantImpl() {
  // When a Constant is destroyed, there may be lingering
  // references to the constant by other constants in the constant pool.  These
  // constants are implicitly dependent on the module that is being deleted,
  // but they don't know that.  Because we only find out when the CPV is
  // deleted, we must now notify all of our users (that should only be
  // Constants) that they are, in fact, invalid now and should be deleted.
  //
  while (!use_empty()) {
    Value *V = use_back();
#ifndef NDEBUG      // Only in -g mode...
    if (!isa<Constant>(V)) {
      errs() << "While deleting: " << *this
             << "\n\nUse still stuck around after Def is destroyed: "
             << *V << "\n\n";
    }
#endif
    assert(isa<Constant>(V) && "References remain to Constant being destroyed");
    Constant *CV = cast<Constant>(V);
    CV->destroyConstant();

    // The constant should remove itself from our use list...
    assert((use_empty() || use_back() != V) && "Constant not removed!");
  }

  // Value has no outstanding references it is safe to delete it now...
  delete this;
}

/// canTrap - Return true if evaluation of this constant could trap.  This is
/// true for things like constant expressions that could divide by zero.
bool Constant::canTrap() const {
  assert(getType()->isFirstClassType() && "Cannot evaluate aggregate vals!");
  // The only thing that could possibly trap are constant exprs.
  const ConstantExpr *CE = dyn_cast<ConstantExpr>(this);
  if (!CE) return false;
  
  // ConstantExpr traps if any operands can trap. 
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (getOperand(i)->canTrap()) 
      return true;

  // Otherwise, only specific operations can trap.
  switch (CE->getOpcode()) {
  default:
    return false;
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
    // Div and rem can trap if the RHS is not known to be non-zero.
    if (!isa<ConstantInt>(getOperand(1)) || getOperand(1)->isNullValue())
      return true;
    return false;
  }
}


/// getRelocationInfo - This method classifies the entry according to
/// whether or not it may generate a relocation entry.  This must be
/// conservative, so if it might codegen to a relocatable entry, it should say
/// so.  The return values are:
/// 
///  NoRelocation: This constant pool entry is guaranteed to never have a
///     relocation applied to it (because it holds a simple constant like
///     '4').
///  LocalRelocation: This entry has relocations, but the entries are
///     guaranteed to be resolvable by the static linker, so the dynamic
///     linker will never see them.
///  GlobalRelocations: This entry may have arbitrary relocations.
///
/// FIXME: This really should not be in VMCore.
Constant::PossibleRelocationsTy Constant::getRelocationInfo() const {
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(this)) {
    if (GV->hasLocalLinkage() || GV->hasHiddenVisibility())
      return LocalRelocation;  // Local to this file/library.
    return GlobalRelocations;    // Global reference.
  }
  
  PossibleRelocationsTy Result = NoRelocation;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    Result = std::max(Result, getOperand(i)->getRelocationInfo());
  
  return Result;
}


/// getVectorElements - This method, which is only valid on constant of vector
/// type, returns the elements of the vector in the specified smallvector.
/// This handles breaking down a vector undef into undef elements, etc.  For
/// constant exprs and other cases we can't handle, we return an empty vector.
void Constant::getVectorElements(LLVMContext &Context,
                                 SmallVectorImpl<Constant*> &Elts) const {
  assert(isa<VectorType>(getType()) && "Not a vector constant!");
  
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(this)) {
    for (unsigned i = 0, e = CV->getNumOperands(); i != e; ++i)
      Elts.push_back(CV->getOperand(i));
    return;
  }
  
  const VectorType *VT = cast<VectorType>(getType());
  if (isa<ConstantAggregateZero>(this)) {
    Elts.assign(VT->getNumElements(), 
                Constant::getNullValue(VT->getElementType()));
    return;
  }
  
  if (isa<UndefValue>(this)) {
    Elts.assign(VT->getNumElements(), UndefValue::get(VT->getElementType()));
    return;
  }
  
  // Unknown type, must be constant expr etc.
}



//===----------------------------------------------------------------------===//
//                                ConstantInt
//===----------------------------------------------------------------------===//

ConstantInt::ConstantInt(const IntegerType *Ty, const APInt& V)
  : Constant(Ty, ConstantIntVal, 0, 0), Val(V) {
  assert(V.getBitWidth() == Ty->getBitWidth() && "Invalid constant for type");
}

ConstantInt* ConstantInt::getTrue(LLVMContext &Context) {
  LLVMContextImpl *pImpl = Context.pImpl;
  sys::SmartScopedWriter<true>(pImpl->ConstantsLock);
  if (pImpl->TheTrueVal)
    return pImpl->TheTrueVal;
  else
    return (pImpl->TheTrueVal =
              ConstantInt::get(IntegerType::get(Context, 1), 1));
}

ConstantInt* ConstantInt::getFalse(LLVMContext &Context) {
  LLVMContextImpl *pImpl = Context.pImpl;
  sys::SmartScopedWriter<true>(pImpl->ConstantsLock);
  if (pImpl->TheFalseVal)
    return pImpl->TheFalseVal;
  else
    return (pImpl->TheFalseVal =
              ConstantInt::get(IntegerType::get(Context, 1), 0));
}


// Get a ConstantInt from an APInt. Note that the value stored in the DenseMap 
// as the key, is a DenseMapAPIntKeyInfo::KeyTy which has provided the
// operator== and operator!= to ensure that the DenseMap doesn't attempt to
// compare APInt's of different widths, which would violate an APInt class
// invariant which generates an assertion.
ConstantInt *ConstantInt::get(LLVMContext &Context, const APInt& V) {
  // Get the corresponding integer type for the bit width of the value.
  const IntegerType *ITy = IntegerType::get(Context, V.getBitWidth());
  // get an existing value or the insertion position
  DenseMapAPIntKeyInfo::KeyTy Key(V, ITy);
  
  Context.pImpl->ConstantsLock.reader_acquire();
  ConstantInt *&Slot = Context.pImpl->IntConstants[Key]; 
  Context.pImpl->ConstantsLock.reader_release();
    
  if (!Slot) {
    sys::SmartScopedWriter<true> Writer(Context.pImpl->ConstantsLock);
    ConstantInt *&NewSlot = Context.pImpl->IntConstants[Key]; 
    if (!Slot) {
      NewSlot = new ConstantInt(ITy, V);
    }
    
    return NewSlot;
  } else {
    return Slot;
  }
}

Constant* ConstantInt::get(const Type* Ty, uint64_t V, bool isSigned) {
  Constant *C = get(cast<IntegerType>(Ty->getScalarType()),
                               V, isSigned);

  // For vectors, broadcast the value.
  if (const VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::get(
      std::vector<Constant *>(VTy->getNumElements(), C));

  return C;
}

ConstantInt* ConstantInt::get(const IntegerType* Ty, uint64_t V, 
                              bool isSigned) {
  return get(Ty->getContext(), APInt(Ty->getBitWidth(), V, isSigned));
}

ConstantInt* ConstantInt::getSigned(const IntegerType* Ty, int64_t V) {
  return get(Ty, V, true);
}

Constant *ConstantInt::getSigned(const Type *Ty, int64_t V) {
  return get(Ty, V, true);
}

Constant* ConstantInt::get(const Type* Ty, const APInt& V) {
  ConstantInt *C = get(Ty->getContext(), V);
  assert(C->getType() == Ty->getScalarType() &&
         "ConstantInt type doesn't match the type implied by its value!");

  // For vectors, broadcast the value.
  if (const VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::get(
      std::vector<Constant *>(VTy->getNumElements(), C));

  return C;
}

ConstantInt* ConstantInt::get(const IntegerType* Ty, const StringRef& Str,
                              uint8_t radix) {
  return get(Ty->getContext(), APInt(Ty->getBitWidth(), Str, radix));
}

//===----------------------------------------------------------------------===//
//                                ConstantFP
//===----------------------------------------------------------------------===//

static const fltSemantics *TypeToFloatSemantics(const Type *Ty) {
  if (Ty == Type::getFloatTy(Ty->getContext()))
    return &APFloat::IEEEsingle;
  if (Ty == Type::getDoubleTy(Ty->getContext()))
    return &APFloat::IEEEdouble;
  if (Ty == Type::getX86_FP80Ty(Ty->getContext()))
    return &APFloat::x87DoubleExtended;
  else if (Ty == Type::getFP128Ty(Ty->getContext()))
    return &APFloat::IEEEquad;
  
  assert(Ty == Type::getPPC_FP128Ty(Ty->getContext()) && "Unknown FP format");
  return &APFloat::PPCDoubleDouble;
}

/// get() - This returns a constant fp for the specified value in the
/// specified type.  This should only be used for simple constant values like
/// 2.0/1.0 etc, that are known-valid both as double and as the target format.
Constant* ConstantFP::get(const Type* Ty, double V) {
  LLVMContext &Context = Ty->getContext();
  
  APFloat FV(V);
  bool ignored;
  FV.convert(*TypeToFloatSemantics(Ty->getScalarType()),
             APFloat::rmNearestTiesToEven, &ignored);
  Constant *C = get(Context, FV);

  // For vectors, broadcast the value.
  if (const VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::get(
      std::vector<Constant *>(VTy->getNumElements(), C));

  return C;
}


Constant* ConstantFP::get(const Type* Ty, const StringRef& Str) {
  LLVMContext &Context = Ty->getContext();

  APFloat FV(*TypeToFloatSemantics(Ty->getScalarType()), Str);
  Constant *C = get(Context, FV);

  // For vectors, broadcast the value.
  if (const VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::get(
      std::vector<Constant *>(VTy->getNumElements(), C));

  return C; 
}


ConstantFP* ConstantFP::getNegativeZero(const Type* Ty) {
  LLVMContext &Context = Ty->getContext();
  APFloat apf = cast <ConstantFP>(Constant::getNullValue(Ty))->getValueAPF();
  apf.changeSign();
  return get(Context, apf);
}


Constant* ConstantFP::getZeroValueForNegation(const Type* Ty) {
  if (const VectorType *PTy = dyn_cast<VectorType>(Ty))
    if (PTy->getElementType()->isFloatingPoint()) {
      std::vector<Constant*> zeros(PTy->getNumElements(),
                           getNegativeZero(PTy->getElementType()));
      return ConstantVector::get(PTy, zeros);
    }

  if (Ty->isFloatingPoint()) 
    return getNegativeZero(Ty);

  return Constant::getNullValue(Ty);
}


// ConstantFP accessors.
ConstantFP* ConstantFP::get(LLVMContext &Context, const APFloat& V) {
  DenseMapAPFloatKeyInfo::KeyTy Key(V);
  
  LLVMContextImpl* pImpl = Context.pImpl;
  
  pImpl->ConstantsLock.reader_acquire();
  ConstantFP *&Slot = pImpl->FPConstants[Key];
  pImpl->ConstantsLock.reader_release();
    
  if (!Slot) {
    sys::SmartScopedWriter<true> Writer(pImpl->ConstantsLock);
    ConstantFP *&NewSlot = pImpl->FPConstants[Key];
    if (!NewSlot) {
      const Type *Ty;
      if (&V.getSemantics() == &APFloat::IEEEsingle)
        Ty = Type::getFloatTy(Context);
      else if (&V.getSemantics() == &APFloat::IEEEdouble)
        Ty = Type::getDoubleTy(Context);
      else if (&V.getSemantics() == &APFloat::x87DoubleExtended)
        Ty = Type::getX86_FP80Ty(Context);
      else if (&V.getSemantics() == &APFloat::IEEEquad)
        Ty = Type::getFP128Ty(Context);
      else {
        assert(&V.getSemantics() == &APFloat::PPCDoubleDouble && 
               "Unknown FP format");
        Ty = Type::getPPC_FP128Ty(Context);
      }
      NewSlot = new ConstantFP(Ty, V);
    }
    
    return NewSlot;
  }
  
  return Slot;
}

ConstantFP::ConstantFP(const Type *Ty, const APFloat& V)
  : Constant(Ty, ConstantFPVal, 0, 0), Val(V) {
  assert(&V.getSemantics() == TypeToFloatSemantics(Ty) &&
         "FP type Mismatch");
}

bool ConstantFP::isNullValue() const {
  return Val.isZero() && !Val.isNegative();
}

bool ConstantFP::isExactlyValue(const APFloat& V) const {
  return Val.bitwiseIsEqual(V);
}

//===----------------------------------------------------------------------===//
//                            ConstantXXX Classes
//===----------------------------------------------------------------------===//


ConstantArray::ConstantArray(const ArrayType *T,
                             const std::vector<Constant*> &V)
  : Constant(T, ConstantArrayVal,
             OperandTraits<ConstantArray>::op_end(this) - V.size(),
             V.size()) {
  assert(V.size() == T->getNumElements() &&
         "Invalid initializer vector for constant array");
  Use *OL = OperandList;
  for (std::vector<Constant*>::const_iterator I = V.begin(), E = V.end();
       I != E; ++I, ++OL) {
    Constant *C = *I;
    assert((C->getType() == T->getElementType() ||
            (T->isAbstract() &&
             C->getType()->getTypeID() == T->getElementType()->getTypeID())) &&
           "Initializer for array element doesn't match array element type!");
    *OL = C;
  }
}

Constant *ConstantArray::get(const ArrayType *Ty, 
                             const std::vector<Constant*> &V) {
  LLVMContextImpl *pImpl = Ty->getContext().pImpl;
  // If this is an all-zero array, return a ConstantAggregateZero object
  if (!V.empty()) {
    Constant *C = V[0];
    if (!C->isNullValue()) {
      // Implicitly locked.
      return pImpl->ArrayConstants.getOrCreate(Ty, V);
    }
    for (unsigned i = 1, e = V.size(); i != e; ++i)
      if (V[i] != C) {
        // Implicitly locked.
        return pImpl->ArrayConstants.getOrCreate(Ty, V);
      }
  }
  
  return ConstantAggregateZero::get(Ty);
}


Constant* ConstantArray::get(const ArrayType* T, Constant* const* Vals,
                             unsigned NumVals) {
  // FIXME: make this the primary ctor method.
  return get(T, std::vector<Constant*>(Vals, Vals+NumVals));
}

/// ConstantArray::get(const string&) - Return an array that is initialized to
/// contain the specified string.  If length is zero then a null terminator is 
/// added to the specified string so that it may be used in a natural way. 
/// Otherwise, the length parameter specifies how much of the string to use 
/// and it won't be null terminated.
///
Constant* ConstantArray::get(LLVMContext &Context, const StringRef &Str,
                             bool AddNull) {
  std::vector<Constant*> ElementVals;
  for (unsigned i = 0; i < Str.size(); ++i)
    ElementVals.push_back(ConstantInt::get(Type::getInt8Ty(Context), Str[i]));

  // Add a null terminator to the string...
  if (AddNull) {
    ElementVals.push_back(ConstantInt::get(Type::getInt8Ty(Context), 0));
  }

  ArrayType *ATy = ArrayType::get(Type::getInt8Ty(Context), ElementVals.size());
  return get(ATy, ElementVals);
}



ConstantStruct::ConstantStruct(const StructType *T,
                               const std::vector<Constant*> &V)
  : Constant(T, ConstantStructVal,
             OperandTraits<ConstantStruct>::op_end(this) - V.size(),
             V.size()) {
  assert(V.size() == T->getNumElements() &&
         "Invalid initializer vector for constant structure");
  Use *OL = OperandList;
  for (std::vector<Constant*>::const_iterator I = V.begin(), E = V.end();
       I != E; ++I, ++OL) {
    Constant *C = *I;
    assert((C->getType() == T->getElementType(I-V.begin()) ||
            ((T->getElementType(I-V.begin())->isAbstract() ||
              C->getType()->isAbstract()) &&
             T->getElementType(I-V.begin())->getTypeID() == 
                   C->getType()->getTypeID())) &&
           "Initializer for struct element doesn't match struct element type!");
    *OL = C;
  }
}

// ConstantStruct accessors.
Constant* ConstantStruct::get(const StructType* T,
                              const std::vector<Constant*>& V) {
  LLVMContextImpl* pImpl = T->getContext().pImpl;
  
  // Create a ConstantAggregateZero value if all elements are zeros...
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    if (!V[i]->isNullValue())
      // Implicitly locked.
      return pImpl->StructConstants.getOrCreate(T, V);

  return ConstantAggregateZero::get(T);
}

Constant* ConstantStruct::get(LLVMContext &Context,
                              const std::vector<Constant*>& V, bool packed) {
  std::vector<const Type*> StructEls;
  StructEls.reserve(V.size());
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    StructEls.push_back(V[i]->getType());
  return get(StructType::get(Context, StructEls, packed), V);
}

Constant* ConstantStruct::get(LLVMContext &Context,
                              Constant* const *Vals, unsigned NumVals,
                              bool Packed) {
  // FIXME: make this the primary ctor method.
  return get(Context, std::vector<Constant*>(Vals, Vals+NumVals), Packed);
}

ConstantVector::ConstantVector(const VectorType *T,
                               const std::vector<Constant*> &V)
  : Constant(T, ConstantVectorVal,
             OperandTraits<ConstantVector>::op_end(this) - V.size(),
             V.size()) {
  Use *OL = OperandList;
    for (std::vector<Constant*>::const_iterator I = V.begin(), E = V.end();
         I != E; ++I, ++OL) {
      Constant *C = *I;
      assert((C->getType() == T->getElementType() ||
            (T->isAbstract() &&
             C->getType()->getTypeID() == T->getElementType()->getTypeID())) &&
           "Initializer for vector element doesn't match vector element type!");
    *OL = C;
  }
}

// ConstantVector accessors.
Constant* ConstantVector::get(const VectorType* T,
                              const std::vector<Constant*>& V) {
   assert(!V.empty() && "Vectors can't be empty");
   LLVMContext &Context = T->getContext();
   LLVMContextImpl *pImpl = Context.pImpl;
   
  // If this is an all-undef or alll-zero vector, return a
  // ConstantAggregateZero or UndefValue.
  Constant *C = V[0];
  bool isZero = C->isNullValue();
  bool isUndef = isa<UndefValue>(C);

  if (isZero || isUndef) {
    for (unsigned i = 1, e = V.size(); i != e; ++i)
      if (V[i] != C) {
        isZero = isUndef = false;
        break;
      }
  }
  
  if (isZero)
    return ConstantAggregateZero::get(T);
  if (isUndef)
    return UndefValue::get(T);
    
  // Implicitly locked.
  return pImpl->VectorConstants.getOrCreate(T, V);
}

Constant* ConstantVector::get(const std::vector<Constant*>& V) {
  assert(!V.empty() && "Cannot infer type if V is empty");
  return get(VectorType::get(V.front()->getType(),V.size()), V);
}

Constant* ConstantVector::get(Constant* const* Vals, unsigned NumVals) {
  // FIXME: make this the primary ctor method.
  return get(std::vector<Constant*>(Vals, Vals+NumVals));
}

// Utility function for determining if a ConstantExpr is a CastOp or not. This
// can't be inline because we don't want to #include Instruction.h into
// Constant.h
bool ConstantExpr::isCast() const {
  return Instruction::isCast(getOpcode());
}

bool ConstantExpr::isCompare() const {
  return getOpcode() == Instruction::ICmp || getOpcode() == Instruction::FCmp;
}

bool ConstantExpr::hasIndices() const {
  return getOpcode() == Instruction::ExtractValue ||
         getOpcode() == Instruction::InsertValue;
}

const SmallVector<unsigned, 4> &ConstantExpr::getIndices() const {
  if (const ExtractValueConstantExpr *EVCE =
        dyn_cast<ExtractValueConstantExpr>(this))
    return EVCE->Indices;

  return cast<InsertValueConstantExpr>(this)->Indices;
}

unsigned ConstantExpr::getPredicate() const {
  assert(getOpcode() == Instruction::FCmp || 
         getOpcode() == Instruction::ICmp);
  return ((const CompareConstantExpr*)this)->predicate;
}

/// getWithOperandReplaced - Return a constant expression identical to this
/// one, but with the specified operand set to the specified value.
Constant *
ConstantExpr::getWithOperandReplaced(unsigned OpNo, Constant *Op) const {
  assert(OpNo < getNumOperands() && "Operand num is out of range!");
  assert(Op->getType() == getOperand(OpNo)->getType() &&
         "Replacing operand with value of different type!");
  if (getOperand(OpNo) == Op)
    return const_cast<ConstantExpr*>(this);
  
  Constant *Op0, *Op1, *Op2;
  switch (getOpcode()) {
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    return ConstantExpr::getCast(getOpcode(), Op, getType());
  case Instruction::Select:
    Op0 = (OpNo == 0) ? Op : getOperand(0);
    Op1 = (OpNo == 1) ? Op : getOperand(1);
    Op2 = (OpNo == 2) ? Op : getOperand(2);
    return ConstantExpr::getSelect(Op0, Op1, Op2);
  case Instruction::InsertElement:
    Op0 = (OpNo == 0) ? Op : getOperand(0);
    Op1 = (OpNo == 1) ? Op : getOperand(1);
    Op2 = (OpNo == 2) ? Op : getOperand(2);
    return ConstantExpr::getInsertElement(Op0, Op1, Op2);
  case Instruction::ExtractElement:
    Op0 = (OpNo == 0) ? Op : getOperand(0);
    Op1 = (OpNo == 1) ? Op : getOperand(1);
    return ConstantExpr::getExtractElement(Op0, Op1);
  case Instruction::ShuffleVector:
    Op0 = (OpNo == 0) ? Op : getOperand(0);
    Op1 = (OpNo == 1) ? Op : getOperand(1);
    Op2 = (OpNo == 2) ? Op : getOperand(2);
    return ConstantExpr::getShuffleVector(Op0, Op1, Op2);
  case Instruction::GetElementPtr: {
    SmallVector<Constant*, 8> Ops;
    Ops.resize(getNumOperands()-1);
    for (unsigned i = 1, e = getNumOperands(); i != e; ++i)
      Ops[i-1] = getOperand(i);
    if (OpNo == 0)
      return ConstantExpr::getGetElementPtr(Op, &Ops[0], Ops.size());
    Ops[OpNo-1] = Op;
    return ConstantExpr::getGetElementPtr(getOperand(0), &Ops[0], Ops.size());
  }
  default:
    assert(getNumOperands() == 2 && "Must be binary operator?");
    Op0 = (OpNo == 0) ? Op : getOperand(0);
    Op1 = (OpNo == 1) ? Op : getOperand(1);
    return ConstantExpr::get(getOpcode(), Op0, Op1);
  }
}

/// getWithOperands - This returns the current constant expression with the
/// operands replaced with the specified values.  The specified operands must
/// match count and type with the existing ones.
Constant *ConstantExpr::
getWithOperands(Constant* const *Ops, unsigned NumOps) const {
  assert(NumOps == getNumOperands() && "Operand count mismatch!");
  bool AnyChange = false;
  for (unsigned i = 0; i != NumOps; ++i) {
    assert(Ops[i]->getType() == getOperand(i)->getType() &&
           "Operand type mismatch!");
    AnyChange |= Ops[i] != getOperand(i);
  }
  if (!AnyChange)  // No operands changed, return self.
    return const_cast<ConstantExpr*>(this);

  switch (getOpcode()) {
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    return ConstantExpr::getCast(getOpcode(), Ops[0], getType());
  case Instruction::Select:
    return ConstantExpr::getSelect(Ops[0], Ops[1], Ops[2]);
  case Instruction::InsertElement:
    return ConstantExpr::getInsertElement(Ops[0], Ops[1], Ops[2]);
  case Instruction::ExtractElement:
    return ConstantExpr::getExtractElement(Ops[0], Ops[1]);
  case Instruction::ShuffleVector:
    return ConstantExpr::getShuffleVector(Ops[0], Ops[1], Ops[2]);
  case Instruction::GetElementPtr:
    return ConstantExpr::getGetElementPtr(Ops[0], &Ops[1], NumOps-1);
  case Instruction::ICmp:
  case Instruction::FCmp:
    return ConstantExpr::getCompare(getPredicate(), Ops[0], Ops[1]);
  default:
    assert(getNumOperands() == 2 && "Must be binary operator?");
    return ConstantExpr::get(getOpcode(), Ops[0], Ops[1]);
  }
}


//===----------------------------------------------------------------------===//
//                      isValueValidForType implementations

bool ConstantInt::isValueValidForType(const Type *Ty, uint64_t Val) {
  unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth(); // assert okay
  if (Ty == Type::getInt1Ty(Ty->getContext()))
    return Val == 0 || Val == 1;
  if (NumBits >= 64)
    return true; // always true, has to fit in largest type
  uint64_t Max = (1ll << NumBits) - 1;
  return Val <= Max;
}

bool ConstantInt::isValueValidForType(const Type *Ty, int64_t Val) {
  unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth(); // assert okay
  if (Ty == Type::getInt1Ty(Ty->getContext()))
    return Val == 0 || Val == 1 || Val == -1;
  if (NumBits >= 64)
    return true; // always true, has to fit in largest type
  int64_t Min = -(1ll << (NumBits-1));
  int64_t Max = (1ll << (NumBits-1)) - 1;
  return (Val >= Min && Val <= Max);
}

bool ConstantFP::isValueValidForType(const Type *Ty, const APFloat& Val) {
  // convert modifies in place, so make a copy.
  APFloat Val2 = APFloat(Val);
  bool losesInfo;
  switch (Ty->getTypeID()) {
  default:
    return false;         // These can't be represented as floating point!

  // FIXME rounding mode needs to be more flexible
  case Type::FloatTyID: {
    if (&Val2.getSemantics() == &APFloat::IEEEsingle)
      return true;
    Val2.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven, &losesInfo);
    return !losesInfo;
  }
  case Type::DoubleTyID: {
    if (&Val2.getSemantics() == &APFloat::IEEEsingle ||
        &Val2.getSemantics() == &APFloat::IEEEdouble)
      return true;
    Val2.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &losesInfo);
    return !losesInfo;
  }
  case Type::X86_FP80TyID:
    return &Val2.getSemantics() == &APFloat::IEEEsingle || 
           &Val2.getSemantics() == &APFloat::IEEEdouble ||
           &Val2.getSemantics() == &APFloat::x87DoubleExtended;
  case Type::FP128TyID:
    return &Val2.getSemantics() == &APFloat::IEEEsingle || 
           &Val2.getSemantics() == &APFloat::IEEEdouble ||
           &Val2.getSemantics() == &APFloat::IEEEquad;
  case Type::PPC_FP128TyID:
    return &Val2.getSemantics() == &APFloat::IEEEsingle || 
           &Val2.getSemantics() == &APFloat::IEEEdouble ||
           &Val2.getSemantics() == &APFloat::PPCDoubleDouble;
  }
}

//===----------------------------------------------------------------------===//
//                      Factory Function Implementation

static char getValType(ConstantAggregateZero *CPZ) { return 0; }

ConstantAggregateZero* ConstantAggregateZero::get(const Type* Ty) {
  assert((isa<StructType>(Ty) || isa<ArrayType>(Ty) || isa<VectorType>(Ty)) &&
         "Cannot create an aggregate zero of non-aggregate type!");
  
  LLVMContextImpl *pImpl = Ty->getContext().pImpl;
  // Implicitly locked.
  return pImpl->AggZeroConstants.getOrCreate(Ty, 0);
}

/// destroyConstant - Remove the constant from the constant table...
///
void ConstantAggregateZero::destroyConstant() {
  // Implicitly locked.
  getType()->getContext().pImpl->AggZeroConstants.remove(this);
  destroyConstantImpl();
}

/// destroyConstant - Remove the constant from the constant table...
///
void ConstantArray::destroyConstant() {
  // Implicitly locked.
  getType()->getContext().pImpl->ArrayConstants.remove(this);
  destroyConstantImpl();
}

/// isString - This method returns true if the array is an array of i8, and 
/// if the elements of the array are all ConstantInt's.
bool ConstantArray::isString() const {
  // Check the element type for i8...
  if (getType()->getElementType() != Type::getInt8Ty(getContext()))
    return false;
  // Check the elements to make sure they are all integers, not constant
  // expressions.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (!isa<ConstantInt>(getOperand(i)))
      return false;
  return true;
}

/// isCString - This method returns true if the array is a string (see
/// isString) and it ends in a null byte \\0 and does not contains any other
/// null bytes except its terminator.
bool ConstantArray::isCString() const {
  // Check the element type for i8...
  if (getType()->getElementType() != Type::getInt8Ty(getContext()))
    return false;

  // Last element must be a null.
  if (!getOperand(getNumOperands()-1)->isNullValue())
    return false;
  // Other elements must be non-null integers.
  for (unsigned i = 0, e = getNumOperands()-1; i != e; ++i) {
    if (!isa<ConstantInt>(getOperand(i)))
      return false;
    if (getOperand(i)->isNullValue())
      return false;
  }
  return true;
}


/// getAsString - If the sub-element type of this array is i8
/// then this method converts the array to an std::string and returns it.
/// Otherwise, it asserts out.
///
std::string ConstantArray::getAsString() const {
  assert(isString() && "Not a string!");
  std::string Result;
  Result.reserve(getNumOperands());
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    Result.push_back((char)cast<ConstantInt>(getOperand(i))->getZExtValue());
  return Result;
}


//---- ConstantStruct::get() implementation...
//

namespace llvm {

}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantStruct::destroyConstant() {
  // Implicitly locked.
  getType()->getContext().pImpl->StructConstants.remove(this);
  destroyConstantImpl();
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantVector::destroyConstant() {
  // Implicitly locked.
  getType()->getContext().pImpl->VectorConstants.remove(this);
  destroyConstantImpl();
}

/// This function will return true iff every element in this vector constant
/// is set to all ones.
/// @returns true iff this constant's emements are all set to all ones.
/// @brief Determine if the value is all ones.
bool ConstantVector::isAllOnesValue() const {
  // Check out first element.
  const Constant *Elt = getOperand(0);
  const ConstantInt *CI = dyn_cast<ConstantInt>(Elt);
  if (!CI || !CI->isAllOnesValue()) return false;
  // Then make sure all remaining elements point to the same value.
  for (unsigned I = 1, E = getNumOperands(); I < E; ++I) {
    if (getOperand(I) != Elt) return false;
  }
  return true;
}

/// getSplatValue - If this is a splat constant, where all of the
/// elements have the same value, return that value. Otherwise return null.
Constant *ConstantVector::getSplatValue() {
  // Check out first element.
  Constant *Elt = getOperand(0);
  // Then make sure all remaining elements point to the same value.
  for (unsigned I = 1, E = getNumOperands(); I < E; ++I)
    if (getOperand(I) != Elt) return 0;
  return Elt;
}

//---- ConstantPointerNull::get() implementation...
//

static char getValType(ConstantPointerNull *) {
  return 0;
}


ConstantPointerNull *ConstantPointerNull::get(const PointerType *Ty) {
  // Implicitly locked.
  return Ty->getContext().pImpl->NullPtrConstants.getOrCreate(Ty, 0);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantPointerNull::destroyConstant() {
  // Implicitly locked.
  getType()->getContext().pImpl->NullPtrConstants.remove(this);
  destroyConstantImpl();
}


//---- UndefValue::get() implementation...
//

static char getValType(UndefValue *) {
  return 0;
}

UndefValue *UndefValue::get(const Type *Ty) {
  // Implicitly locked.
  return Ty->getContext().pImpl->UndefValueConstants.getOrCreate(Ty, 0);
}

// destroyConstant - Remove the constant from the constant table.
//
void UndefValue::destroyConstant() {
  // Implicitly locked.
  getType()->getContext().pImpl->UndefValueConstants.remove(this);
  destroyConstantImpl();
}

//---- ConstantExpr::get() implementations...
//

static ExprMapKeyType getValType(ConstantExpr *CE) {
  std::vector<Constant*> Operands;
  Operands.reserve(CE->getNumOperands());
  for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i)
    Operands.push_back(cast<Constant>(CE->getOperand(i)));
  return ExprMapKeyType(CE->getOpcode(), Operands, 
      CE->isCompare() ? CE->getPredicate() : 0,
      CE->hasIndices() ?
        CE->getIndices() : SmallVector<unsigned, 4>());
}

/// This is a utility function to handle folding of casts and lookup of the
/// cast in the ExprConstants map. It is used by the various get* methods below.
static inline Constant *getFoldedCast(
  Instruction::CastOps opc, Constant *C, const Type *Ty) {
  assert(Ty->isFirstClassType() && "Cannot cast to an aggregate type!");
  // Fold a few common cases
  if (Constant *FC = ConstantFoldCastInstruction(Ty->getContext(), opc, C, Ty))
    return FC;

  LLVMContextImpl *pImpl = Ty->getContext().pImpl;

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> argVec(1, C);
  ExprMapKeyType Key(opc, argVec);
  
  // Implicitly locked.
  return pImpl->ExprConstants.getOrCreate(Ty, Key);
}
 
Constant *ConstantExpr::getCast(unsigned oc, Constant *C, const Type *Ty) {
  Instruction::CastOps opc = Instruction::CastOps(oc);
  assert(Instruction::isCast(opc) && "opcode out of range");
  assert(C && Ty && "Null arguments to getCast");
  assert(Ty->isFirstClassType() && "Cannot cast to an aggregate type!");

  switch (opc) {
    default:
      llvm_unreachable("Invalid cast opcode");
      break;
    case Instruction::Trunc:    return getTrunc(C, Ty);
    case Instruction::ZExt:     return getZExt(C, Ty);
    case Instruction::SExt:     return getSExt(C, Ty);
    case Instruction::FPTrunc:  return getFPTrunc(C, Ty);
    case Instruction::FPExt:    return getFPExtend(C, Ty);
    case Instruction::UIToFP:   return getUIToFP(C, Ty);
    case Instruction::SIToFP:   return getSIToFP(C, Ty);
    case Instruction::FPToUI:   return getFPToUI(C, Ty);
    case Instruction::FPToSI:   return getFPToSI(C, Ty);
    case Instruction::PtrToInt: return getPtrToInt(C, Ty);
    case Instruction::IntToPtr: return getIntToPtr(C, Ty);
    case Instruction::BitCast:  return getBitCast(C, Ty);
  }
  return 0;
} 

Constant *ConstantExpr::getZExtOrBitCast(Constant *C, const Type *Ty) {
  if (C->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return getCast(Instruction::BitCast, C, Ty);
  return getCast(Instruction::ZExt, C, Ty);
}

Constant *ConstantExpr::getSExtOrBitCast(Constant *C, const Type *Ty) {
  if (C->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return getCast(Instruction::BitCast, C, Ty);
  return getCast(Instruction::SExt, C, Ty);
}

Constant *ConstantExpr::getTruncOrBitCast(Constant *C, const Type *Ty) {
  if (C->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return getCast(Instruction::BitCast, C, Ty);
  return getCast(Instruction::Trunc, C, Ty);
}

Constant *ConstantExpr::getPointerCast(Constant *S, const Type *Ty) {
  assert(isa<PointerType>(S->getType()) && "Invalid cast");
  assert((Ty->isInteger() || isa<PointerType>(Ty)) && "Invalid cast");

  if (Ty->isInteger())
    return getCast(Instruction::PtrToInt, S, Ty);
  return getCast(Instruction::BitCast, S, Ty);
}

Constant *ConstantExpr::getIntegerCast(Constant *C, const Type *Ty, 
                                       bool isSigned) {
  assert(C->getType()->isIntOrIntVector() &&
         Ty->isIntOrIntVector() && "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::Trunc :
      (isSigned ? Instruction::SExt : Instruction::ZExt)));
  return getCast(opcode, C, Ty);
}

Constant *ConstantExpr::getFPCast(Constant *C, const Type *Ty) {
  assert(C->getType()->isFPOrFPVector() && Ty->isFPOrFPVector() &&
         "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  if (SrcBits == DstBits)
    return C; // Avoid a useless cast
  Instruction::CastOps opcode =
     (SrcBits > DstBits ? Instruction::FPTrunc : Instruction::FPExt);
  return getCast(opcode, C, Ty);
}

Constant *ConstantExpr::getTrunc(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVector() && "Trunc operand must be integer");
  assert(Ty->isIntOrIntVector() && "Trunc produces only integral");
  assert(C->getType()->getScalarSizeInBits() > Ty->getScalarSizeInBits()&&
         "SrcTy must be larger than DestTy for Trunc!");

  return getFoldedCast(Instruction::Trunc, C, Ty);
}

Constant *ConstantExpr::getSExt(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVector() && "SExt operand must be integral");
  assert(Ty->isIntOrIntVector() && "SExt produces only integer");
  assert(C->getType()->getScalarSizeInBits() < Ty->getScalarSizeInBits()&&
         "SrcTy must be smaller than DestTy for SExt!");

  return getFoldedCast(Instruction::SExt, C, Ty);
}

Constant *ConstantExpr::getZExt(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVector() && "ZEXt operand must be integral");
  assert(Ty->isIntOrIntVector() && "ZExt produces only integer");
  assert(C->getType()->getScalarSizeInBits() < Ty->getScalarSizeInBits()&&
         "SrcTy must be smaller than DestTy for ZExt!");

  return getFoldedCast(Instruction::ZExt, C, Ty);
}

Constant *ConstantExpr::getFPTrunc(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVector() && Ty->isFPOrFPVector() &&
         C->getType()->getScalarSizeInBits() > Ty->getScalarSizeInBits()&&
         "This is an illegal floating point truncation!");
  return getFoldedCast(Instruction::FPTrunc, C, Ty);
}

Constant *ConstantExpr::getFPExtend(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVector() && Ty->isFPOrFPVector() &&
         C->getType()->getScalarSizeInBits() < Ty->getScalarSizeInBits()&&
         "This is an illegal floating point extension!");
  return getFoldedCast(Instruction::FPExt, C, Ty);
}

Constant *ConstantExpr::getUIToFP(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVector() && Ty->isFPOrFPVector() &&
         "This is an illegal uint to floating point cast!");
  return getFoldedCast(Instruction::UIToFP, C, Ty);
}

Constant *ConstantExpr::getSIToFP(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVector() && Ty->isFPOrFPVector() &&
         "This is an illegal sint to floating point cast!");
  return getFoldedCast(Instruction::SIToFP, C, Ty);
}

Constant *ConstantExpr::getFPToUI(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVector() && Ty->isIntOrIntVector() &&
         "This is an illegal floating point to uint cast!");
  return getFoldedCast(Instruction::FPToUI, C, Ty);
}

Constant *ConstantExpr::getFPToSI(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVector() && Ty->isIntOrIntVector() &&
         "This is an illegal floating point to sint cast!");
  return getFoldedCast(Instruction::FPToSI, C, Ty);
}

Constant *ConstantExpr::getPtrToInt(Constant *C, const Type *DstTy) {
  assert(isa<PointerType>(C->getType()) && "PtrToInt source must be pointer");
  assert(DstTy->isInteger() && "PtrToInt destination must be integral");
  return getFoldedCast(Instruction::PtrToInt, C, DstTy);
}

Constant *ConstantExpr::getIntToPtr(Constant *C, const Type *DstTy) {
  assert(C->getType()->isInteger() && "IntToPtr source must be integral");
  assert(isa<PointerType>(DstTy) && "IntToPtr destination must be a pointer");
  return getFoldedCast(Instruction::IntToPtr, C, DstTy);
}

Constant *ConstantExpr::getBitCast(Constant *C, const Type *DstTy) {
  // BitCast implies a no-op cast of type only. No bits change.  However, you 
  // can't cast pointers to anything but pointers.
#ifndef NDEBUG
  const Type *SrcTy = C->getType();
  assert((isa<PointerType>(SrcTy) == isa<PointerType>(DstTy)) &&
         "BitCast cannot cast pointer to non-pointer and vice versa");

  // Now we know we're not dealing with mismatched pointer casts (ptr->nonptr
  // or nonptr->ptr). For all the other types, the cast is okay if source and 
  // destination bit widths are identical.
  unsigned SrcBitSize = SrcTy->getPrimitiveSizeInBits();
  unsigned DstBitSize = DstTy->getPrimitiveSizeInBits();
#endif
  assert(SrcBitSize == DstBitSize && "BitCast requires types of same width");
  
  // It is common to ask for a bitcast of a value to its own type, handle this
  // speedily.
  if (C->getType() == DstTy) return C;
  
  return getFoldedCast(Instruction::BitCast, C, DstTy);
}

Constant *ConstantExpr::getTy(const Type *ReqTy, unsigned Opcode,
                              Constant *C1, Constant *C2) {
  // Check the operands for consistency first
  assert(Opcode >= Instruction::BinaryOpsBegin &&
         Opcode <  Instruction::BinaryOpsEnd   &&
         "Invalid opcode in binary constant expression");
  assert(C1->getType() == C2->getType() &&
         "Operand types in binary constant expression should match");

  if (ReqTy == C1->getType() || ReqTy == Type::getInt1Ty(ReqTy->getContext()))
    if (Constant *FC = ConstantFoldBinaryInstruction(ReqTy->getContext(),
                                                     Opcode, C1, C2))
      return FC;          // Fold a few common cases...

  std::vector<Constant*> argVec(1, C1); argVec.push_back(C2);
  ExprMapKeyType Key(Opcode, argVec);
  
  LLVMContextImpl *pImpl = ReqTy->getContext().pImpl;
  
  // Implicitly locked.
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getCompareTy(unsigned short predicate,
                                     Constant *C1, Constant *C2) {
  switch (predicate) {
    default: llvm_unreachable("Invalid CmpInst predicate");
    case CmpInst::FCMP_FALSE: case CmpInst::FCMP_OEQ: case CmpInst::FCMP_OGT:
    case CmpInst::FCMP_OGE:   case CmpInst::FCMP_OLT: case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_ONE:   case CmpInst::FCMP_ORD: case CmpInst::FCMP_UNO:
    case CmpInst::FCMP_UEQ:   case CmpInst::FCMP_UGT: case CmpInst::FCMP_UGE:
    case CmpInst::FCMP_ULT:   case CmpInst::FCMP_ULE: case CmpInst::FCMP_UNE:
    case CmpInst::FCMP_TRUE:
      return getFCmp(predicate, C1, C2);

    case CmpInst::ICMP_EQ:  case CmpInst::ICMP_NE:  case CmpInst::ICMP_UGT:
    case CmpInst::ICMP_UGE: case CmpInst::ICMP_ULT: case CmpInst::ICMP_ULE:
    case CmpInst::ICMP_SGT: case CmpInst::ICMP_SGE: case CmpInst::ICMP_SLT:
    case CmpInst::ICMP_SLE:
      return getICmp(predicate, C1, C2);
  }
}

Constant *ConstantExpr::get(unsigned Opcode, Constant *C1, Constant *C2) {
  // API compatibility: Adjust integer opcodes to floating-point opcodes.
  if (C1->getType()->isFPOrFPVector()) {
    if (Opcode == Instruction::Add) Opcode = Instruction::FAdd;
    else if (Opcode == Instruction::Sub) Opcode = Instruction::FSub;
    else if (Opcode == Instruction::Mul) Opcode = Instruction::FMul;
  }
#ifndef NDEBUG
  switch (Opcode) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVector() &&
           "Tried to create an integer operation on a non-integer type!");
    break;
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isFPOrFPVector() &&
           "Tried to create a floating-point operation on a "
           "non-floating-point type!");
    break;
  case Instruction::UDiv: 
  case Instruction::SDiv: 
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVector() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::FDiv:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isFPOrFPVector() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::URem: 
  case Instruction::SRem: 
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVector() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::FRem:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isFPOrFPVector() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVector() &&
           "Tried to create a logical operation on a non-integral type!");
    break;
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVector() &&
           "Tried to create a shift operation on a non-integer type!");
    break;
  default:
    break;
  }
#endif

  return getTy(C1->getType(), Opcode, C1, C2);
}

Constant* ConstantExpr::getSizeOf(const Type* Ty) {
  // sizeof is implemented as: (i64) gep (Ty*)null, 1
  // Note that a non-inbounds gep is used, as null isn't within any object.
  Constant *GEPIdx = ConstantInt::get(Type::getInt32Ty(Ty->getContext()), 1);
  Constant *GEP = getGetElementPtr(
                 Constant::getNullValue(PointerType::getUnqual(Ty)), &GEPIdx, 1);
  return getCast(Instruction::PtrToInt, GEP, 
                 Type::getInt64Ty(Ty->getContext()));
}

Constant* ConstantExpr::getAlignOf(const Type* Ty) {
  // alignof is implemented as: (i64) gep ({i8,Ty}*)null, 0, 1
  // Note that a non-inbounds gep is used, as null isn't within any object.
  const Type *AligningTy = StructType::get(Ty->getContext(),
                                   Type::getInt8Ty(Ty->getContext()), Ty, NULL);
  Constant *NullPtr = Constant::getNullValue(AligningTy->getPointerTo());
  Constant *Zero = ConstantInt::get(Type::getInt32Ty(Ty->getContext()), 0);
  Constant *One = ConstantInt::get(Type::getInt32Ty(Ty->getContext()), 1);
  Constant *Indices[2] = { Zero, One };
  Constant *GEP = getGetElementPtr(NullPtr, Indices, 2);
  return getCast(Instruction::PtrToInt, GEP,
                 Type::getInt32Ty(Ty->getContext()));
}

Constant* ConstantExpr::getOffsetOf(const StructType* STy, unsigned FieldNo) {
  // offsetof is implemented as: (i64) gep (Ty*)null, 0, FieldNo
  // Note that a non-inbounds gep is used, as null isn't within any object.
  Constant *GEPIdx[] = {
    ConstantInt::get(Type::getInt64Ty(STy->getContext()), 0),
    ConstantInt::get(Type::getInt32Ty(STy->getContext()), FieldNo)
  };
  Constant *GEP = getGetElementPtr(
                Constant::getNullValue(PointerType::getUnqual(STy)), GEPIdx, 2);
  return getCast(Instruction::PtrToInt, GEP,
                 Type::getInt64Ty(STy->getContext()));
}

Constant *ConstantExpr::getCompare(unsigned short pred, 
                            Constant *C1, Constant *C2) {
  assert(C1->getType() == C2->getType() && "Op types should be identical!");
  return getCompareTy(pred, C1, C2);
}

Constant *ConstantExpr::getSelectTy(const Type *ReqTy, Constant *C,
                                    Constant *V1, Constant *V2) {
  assert(!SelectInst::areInvalidOperands(C, V1, V2)&&"Invalid select operands");

  if (ReqTy == V1->getType())
    if (Constant *SC = ConstantFoldSelectInstruction(
                                                ReqTy->getContext(), C, V1, V2))
      return SC;        // Fold common cases

  std::vector<Constant*> argVec(3, C);
  argVec[1] = V1;
  argVec[2] = V2;
  ExprMapKeyType Key(Instruction::Select, argVec);
  
  LLVMContextImpl *pImpl = ReqTy->getContext().pImpl;
  
  // Implicitly locked.
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getGetElementPtrTy(const Type *ReqTy, Constant *C,
                                           Value* const *Idxs,
                                           unsigned NumIdx) {
  assert(GetElementPtrInst::getIndexedType(C->getType(), Idxs,
                                           Idxs+NumIdx) ==
         cast<PointerType>(ReqTy)->getElementType() &&
         "GEP indices invalid!");

  if (Constant *FC = ConstantFoldGetElementPtr(
                              ReqTy->getContext(), C, (Constant**)Idxs, NumIdx))
    return FC;          // Fold a few common cases...

  assert(isa<PointerType>(C->getType()) &&
         "Non-pointer type for constant GetElementPtr expression");
  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.reserve(NumIdx+1);
  ArgVec.push_back(C);
  for (unsigned i = 0; i != NumIdx; ++i)
    ArgVec.push_back(cast<Constant>(Idxs[i]));
  const ExprMapKeyType Key(Instruction::GetElementPtr, ArgVec);

  LLVMContextImpl *pImpl = ReqTy->getContext().pImpl;

  // Implicitly locked.
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getGetElementPtr(Constant *C, Value* const *Idxs,
                                         unsigned NumIdx) {
  // Get the result type of the getelementptr!
  const Type *Ty = 
    GetElementPtrInst::getIndexedType(C->getType(), Idxs, Idxs+NumIdx);
  assert(Ty && "GEP indices invalid!");
  unsigned As = cast<PointerType>(C->getType())->getAddressSpace();
  return getGetElementPtrTy(PointerType::get(Ty, As), C, Idxs, NumIdx);
}

Constant *ConstantExpr::getGetElementPtr(Constant *C, Constant* const *Idxs,
                                         unsigned NumIdx) {
  return getGetElementPtr(C, (Value* const *)Idxs, NumIdx);
}

Constant *
ConstantExpr::getICmp(unsigned short pred, Constant* LHS, Constant* RHS) {
  assert(LHS->getType() == RHS->getType());
  assert(pred >= ICmpInst::FIRST_ICMP_PREDICATE && 
         pred <= ICmpInst::LAST_ICMP_PREDICATE && "Invalid ICmp Predicate");

  if (Constant *FC = ConstantFoldCompareInstruction(
                                             LHS->getContext(), pred, LHS, RHS))
    return FC;          // Fold a few common cases...

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.push_back(LHS);
  ArgVec.push_back(RHS);
  // Get the key type with both the opcode and predicate
  const ExprMapKeyType Key(Instruction::ICmp, ArgVec, pred);

  LLVMContextImpl *pImpl = LHS->getType()->getContext().pImpl;

  // Implicitly locked.
  return
      pImpl->ExprConstants.getOrCreate(Type::getInt1Ty(LHS->getContext()), Key);
}

Constant *
ConstantExpr::getFCmp(unsigned short pred, Constant* LHS, Constant* RHS) {
  assert(LHS->getType() == RHS->getType());
  assert(pred <= FCmpInst::LAST_FCMP_PREDICATE && "Invalid FCmp Predicate");

  if (Constant *FC = ConstantFoldCompareInstruction(
                                            LHS->getContext(), pred, LHS, RHS))
    return FC;          // Fold a few common cases...

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.push_back(LHS);
  ArgVec.push_back(RHS);
  // Get the key type with both the opcode and predicate
  const ExprMapKeyType Key(Instruction::FCmp, ArgVec, pred);
  
  LLVMContextImpl *pImpl = LHS->getType()->getContext().pImpl;
  
  // Implicitly locked.
  return
      pImpl->ExprConstants.getOrCreate(Type::getInt1Ty(LHS->getContext()), Key);
}

Constant *ConstantExpr::getExtractElementTy(const Type *ReqTy, Constant *Val,
                                            Constant *Idx) {
  if (Constant *FC = ConstantFoldExtractElementInstruction(
                                                ReqTy->getContext(), Val, Idx))
    return FC;          // Fold a few common cases...
  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec(1, Val);
  ArgVec.push_back(Idx);
  const ExprMapKeyType Key(Instruction::ExtractElement,ArgVec);
  
  LLVMContextImpl *pImpl = ReqTy->getContext().pImpl;
  
  // Implicitly locked.
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getExtractElement(Constant *Val, Constant *Idx) {
  assert(isa<VectorType>(Val->getType()) &&
         "Tried to create extractelement operation on non-vector type!");
  assert(Idx->getType() == Type::getInt32Ty(Val->getContext()) &&
         "Extractelement index must be i32 type!");
  return getExtractElementTy(cast<VectorType>(Val->getType())->getElementType(),
                             Val, Idx);
}

Constant *ConstantExpr::getInsertElementTy(const Type *ReqTy, Constant *Val,
                                           Constant *Elt, Constant *Idx) {
  if (Constant *FC = ConstantFoldInsertElementInstruction(
                                            ReqTy->getContext(), Val, Elt, Idx))
    return FC;          // Fold a few common cases...
  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec(1, Val);
  ArgVec.push_back(Elt);
  ArgVec.push_back(Idx);
  const ExprMapKeyType Key(Instruction::InsertElement,ArgVec);
  
  LLVMContextImpl *pImpl = ReqTy->getContext().pImpl;
  
  // Implicitly locked.
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getInsertElement(Constant *Val, Constant *Elt, 
                                         Constant *Idx) {
  assert(isa<VectorType>(Val->getType()) &&
         "Tried to create insertelement operation on non-vector type!");
  assert(Elt->getType() == cast<VectorType>(Val->getType())->getElementType()
         && "Insertelement types must match!");
  assert(Idx->getType() == Type::getInt32Ty(Val->getContext()) &&
         "Insertelement index must be i32 type!");
  return getInsertElementTy(Val->getType(), Val, Elt, Idx);
}

Constant *ConstantExpr::getShuffleVectorTy(const Type *ReqTy, Constant *V1,
                                           Constant *V2, Constant *Mask) {
  if (Constant *FC = ConstantFoldShuffleVectorInstruction(
                                            ReqTy->getContext(), V1, V2, Mask))
    return FC;          // Fold a few common cases...
  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec(1, V1);
  ArgVec.push_back(V2);
  ArgVec.push_back(Mask);
  const ExprMapKeyType Key(Instruction::ShuffleVector,ArgVec);
  
  LLVMContextImpl *pImpl = ReqTy->getContext().pImpl;
  
  // Implicitly locked.
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getShuffleVector(Constant *V1, Constant *V2, 
                                         Constant *Mask) {
  assert(ShuffleVectorInst::isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector constant expr operands!");

  unsigned NElts = cast<VectorType>(Mask->getType())->getNumElements();
  const Type *EltTy = cast<VectorType>(V1->getType())->getElementType();
  const Type *ShufTy = VectorType::get(EltTy, NElts);
  return getShuffleVectorTy(ShufTy, V1, V2, Mask);
}

Constant *ConstantExpr::getInsertValueTy(const Type *ReqTy, Constant *Agg,
                                         Constant *Val,
                                        const unsigned *Idxs, unsigned NumIdx) {
  assert(ExtractValueInst::getIndexedType(Agg->getType(), Idxs,
                                          Idxs+NumIdx) == Val->getType() &&
         "insertvalue indices invalid!");
  assert(Agg->getType() == ReqTy &&
         "insertvalue type invalid!");
  assert(Agg->getType()->isFirstClassType() &&
         "Non-first-class type for constant InsertValue expression");
  Constant *FC = ConstantFoldInsertValueInstruction(
                                  ReqTy->getContext(), Agg, Val, Idxs, NumIdx);
  assert(FC && "InsertValue constant expr couldn't be folded!");
  return FC;
}

Constant *ConstantExpr::getInsertValue(Constant *Agg, Constant *Val,
                                     const unsigned *IdxList, unsigned NumIdx) {
  assert(Agg->getType()->isFirstClassType() &&
         "Tried to create insertelement operation on non-first-class type!");

  const Type *ReqTy = Agg->getType();
#ifndef NDEBUG
  const Type *ValTy =
    ExtractValueInst::getIndexedType(Agg->getType(), IdxList, IdxList+NumIdx);
#endif
  assert(ValTy == Val->getType() && "insertvalue indices invalid!");
  return getInsertValueTy(ReqTy, Agg, Val, IdxList, NumIdx);
}

Constant *ConstantExpr::getExtractValueTy(const Type *ReqTy, Constant *Agg,
                                        const unsigned *Idxs, unsigned NumIdx) {
  assert(ExtractValueInst::getIndexedType(Agg->getType(), Idxs,
                                          Idxs+NumIdx) == ReqTy &&
         "extractvalue indices invalid!");
  assert(Agg->getType()->isFirstClassType() &&
         "Non-first-class type for constant extractvalue expression");
  Constant *FC = ConstantFoldExtractValueInstruction(
                                        ReqTy->getContext(), Agg, Idxs, NumIdx);
  assert(FC && "ExtractValue constant expr couldn't be folded!");
  return FC;
}

Constant *ConstantExpr::getExtractValue(Constant *Agg,
                                     const unsigned *IdxList, unsigned NumIdx) {
  assert(Agg->getType()->isFirstClassType() &&
         "Tried to create extractelement operation on non-first-class type!");

  const Type *ReqTy =
    ExtractValueInst::getIndexedType(Agg->getType(), IdxList, IdxList+NumIdx);
  assert(ReqTy && "extractvalue indices invalid!");
  return getExtractValueTy(ReqTy, Agg, IdxList, NumIdx);
}

Constant* ConstantExpr::getNeg(Constant* C) {
  // API compatibility: Adjust integer opcodes to floating-point opcodes.
  if (C->getType()->isFPOrFPVector())
    return getFNeg(C);
  assert(C->getType()->isIntOrIntVector() &&
         "Cannot NEG a nonintegral value!");
  return get(Instruction::Sub,
             ConstantFP::getZeroValueForNegation(C->getType()),
             C);
}

Constant* ConstantExpr::getFNeg(Constant* C) {
  assert(C->getType()->isFPOrFPVector() &&
         "Cannot FNEG a non-floating-point value!");
  return get(Instruction::FSub,
             ConstantFP::getZeroValueForNegation(C->getType()),
             C);
}

Constant* ConstantExpr::getNot(Constant* C) {
  assert(C->getType()->isIntOrIntVector() &&
         "Cannot NOT a nonintegral value!");
  return get(Instruction::Xor, C, Constant::getAllOnesValue(C->getType()));
}

Constant* ConstantExpr::getAdd(Constant* C1, Constant* C2) {
  return get(Instruction::Add, C1, C2);
}

Constant* ConstantExpr::getFAdd(Constant* C1, Constant* C2) {
  return get(Instruction::FAdd, C1, C2);
}

Constant* ConstantExpr::getSub(Constant* C1, Constant* C2) {
  return get(Instruction::Sub, C1, C2);
}

Constant* ConstantExpr::getFSub(Constant* C1, Constant* C2) {
  return get(Instruction::FSub, C1, C2);
}

Constant* ConstantExpr::getMul(Constant* C1, Constant* C2) {
  return get(Instruction::Mul, C1, C2);
}

Constant* ConstantExpr::getFMul(Constant* C1, Constant* C2) {
  return get(Instruction::FMul, C1, C2);
}

Constant* ConstantExpr::getUDiv(Constant* C1, Constant* C2) {
  return get(Instruction::UDiv, C1, C2);
}

Constant* ConstantExpr::getSDiv(Constant* C1, Constant* C2) {
  return get(Instruction::SDiv, C1, C2);
}

Constant* ConstantExpr::getFDiv(Constant* C1, Constant* C2) {
  return get(Instruction::FDiv, C1, C2);
}

Constant* ConstantExpr::getURem(Constant* C1, Constant* C2) {
  return get(Instruction::URem, C1, C2);
}

Constant* ConstantExpr::getSRem(Constant* C1, Constant* C2) {
  return get(Instruction::SRem, C1, C2);
}

Constant* ConstantExpr::getFRem(Constant* C1, Constant* C2) {
  return get(Instruction::FRem, C1, C2);
}

Constant* ConstantExpr::getAnd(Constant* C1, Constant* C2) {
  return get(Instruction::And, C1, C2);
}

Constant* ConstantExpr::getOr(Constant* C1, Constant* C2) {
  return get(Instruction::Or, C1, C2);
}

Constant* ConstantExpr::getXor(Constant* C1, Constant* C2) {
  return get(Instruction::Xor, C1, C2);
}

Constant* ConstantExpr::getShl(Constant* C1, Constant* C2) {
  return get(Instruction::Shl, C1, C2);
}

Constant* ConstantExpr::getLShr(Constant* C1, Constant* C2) {
  return get(Instruction::LShr, C1, C2);
}

Constant* ConstantExpr::getAShr(Constant* C1, Constant* C2) {
  return get(Instruction::AShr, C1, C2);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantExpr::destroyConstant() {
  // Implicitly locked.
  LLVMContextImpl *pImpl = getType()->getContext().pImpl;
  pImpl->ExprConstants.remove(this);
  destroyConstantImpl();
}

const char *ConstantExpr::getOpcodeName() const {
  return Instruction::getOpcodeName(getOpcode());
}

//===----------------------------------------------------------------------===//
//                replaceUsesOfWithOnConstant implementations

/// replaceUsesOfWithOnConstant - Update this constant array to change uses of
/// 'From' to be uses of 'To'.  This must update the uniquing data structures
/// etc.
///
/// Note that we intentionally replace all uses of From with To here.  Consider
/// a large array that uses 'From' 1000 times.  By handling this case all here,
/// ConstantArray::replaceUsesOfWithOnConstant is only invoked once, and that
/// single invocation handles all 1000 uses.  Handling them one at a time would
/// work, but would be really slow because it would have to unique each updated
/// array instance.

static std::vector<Constant*> getValType(ConstantArray *CA) {
  std::vector<Constant*> Elements;
  Elements.reserve(CA->getNumOperands());
  for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
    Elements.push_back(cast<Constant>(CA->getOperand(i)));
  return Elements;
}


void ConstantArray::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  LLVMContext &Context = getType()->getContext();
  LLVMContextImpl *pImpl = Context.pImpl;

  std::pair<LLVMContextImpl::ArrayConstantsTy::MapKey, Constant*> Lookup;
  Lookup.first.first = getType();
  Lookup.second = this;

  std::vector<Constant*> &Values = Lookup.first.second;
  Values.reserve(getNumOperands());  // Build replacement array.

  // Fill values with the modified operands of the constant array.  Also, 
  // compute whether this turns into an all-zeros array.
  bool isAllZeros = false;
  unsigned NumUpdated = 0;
  if (!ToC->isNullValue()) {
    for (Use *O = OperandList, *E = OperandList+getNumOperands(); O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      if (Val == From) {
        Val = ToC;
        ++NumUpdated;
      }
      Values.push_back(Val);
    }
  } else {
    isAllZeros = true;
    for (Use *O = OperandList, *E = OperandList+getNumOperands();O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      if (Val == From) {
        Val = ToC;
        ++NumUpdated;
      }
      Values.push_back(Val);
      if (isAllZeros) isAllZeros = Val->isNullValue();
    }
  }
  
  Constant *Replacement = 0;
  if (isAllZeros) {
    Replacement = ConstantAggregateZero::get(getType());
  } else {
    // Check to see if we have this array type already.
    sys::SmartScopedWriter<true> Writer(pImpl->ConstantsLock);
    bool Exists;
    LLVMContextImpl::ArrayConstantsTy::MapTy::iterator I =
      pImpl->ArrayConstants.InsertOrGetItem(Lookup, Exists);
    
    if (Exists) {
      Replacement = I->second;
    } else {
      // Okay, the new shape doesn't exist in the system yet.  Instead of
      // creating a new constant array, inserting it, replaceallusesof'ing the
      // old with the new, then deleting the old... just update the current one
      // in place!
      pImpl->ArrayConstants.MoveConstantToNewSlot(this, I);
      
      // Update to the new value.  Optimize for the case when we have a single
      // operand that we're changing, but handle bulk updates efficiently.
      if (NumUpdated == 1) {
        unsigned OperandToUpdate = U - OperandList;
        assert(getOperand(OperandToUpdate) == From &&
               "ReplaceAllUsesWith broken!");
        setOperand(OperandToUpdate, ToC);
      } else {
        for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
          if (getOperand(i) == From)
            setOperand(i, ToC);
      }
      return;
    }
  }
 
  // Otherwise, I do need to replace this with an existing value.
  assert(Replacement != this && "I didn't contain From!");
  
  // Everyone using this now uses the replacement.
  uncheckedReplaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}

static std::vector<Constant*> getValType(ConstantStruct *CS) {
  std::vector<Constant*> Elements;
  Elements.reserve(CS->getNumOperands());
  for (unsigned i = 0, e = CS->getNumOperands(); i != e; ++i)
    Elements.push_back(cast<Constant>(CS->getOperand(i)));
  return Elements;
}

void ConstantStruct::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                 Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  unsigned OperandToUpdate = U-OperandList;
  assert(getOperand(OperandToUpdate) == From && "ReplaceAllUsesWith broken!");

  std::pair<LLVMContextImpl::StructConstantsTy::MapKey, Constant*> Lookup;
  Lookup.first.first = getType();
  Lookup.second = this;
  std::vector<Constant*> &Values = Lookup.first.second;
  Values.reserve(getNumOperands());  // Build replacement struct.
  
  
  // Fill values with the modified operands of the constant struct.  Also, 
  // compute whether this turns into an all-zeros struct.
  bool isAllZeros = false;
  if (!ToC->isNullValue()) {
    for (Use *O = OperandList, *E = OperandList + getNumOperands(); O != E; ++O)
      Values.push_back(cast<Constant>(O->get()));
  } else {
    isAllZeros = true;
    for (Use *O = OperandList, *E = OperandList+getNumOperands(); O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      Values.push_back(Val);
      if (isAllZeros) isAllZeros = Val->isNullValue();
    }
  }
  Values[OperandToUpdate] = ToC;
  
  LLVMContext &Context = getType()->getContext();
  LLVMContextImpl *pImpl = Context.pImpl;
  
  Constant *Replacement = 0;
  if (isAllZeros) {
    Replacement = ConstantAggregateZero::get(getType());
  } else {
    // Check to see if we have this array type already.
    sys::SmartScopedWriter<true> Writer(pImpl->ConstantsLock);
    bool Exists;
    LLVMContextImpl::StructConstantsTy::MapTy::iterator I =
      pImpl->StructConstants.InsertOrGetItem(Lookup, Exists);
    
    if (Exists) {
      Replacement = I->second;
    } else {
      // Okay, the new shape doesn't exist in the system yet.  Instead of
      // creating a new constant struct, inserting it, replaceallusesof'ing the
      // old with the new, then deleting the old... just update the current one
      // in place!
      pImpl->StructConstants.MoveConstantToNewSlot(this, I);
      
      // Update to the new value.
      setOperand(OperandToUpdate, ToC);
      return;
    }
  }
  
  assert(Replacement != this && "I didn't contain From!");
  
  // Everyone using this now uses the replacement.
  uncheckedReplaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}

static std::vector<Constant*> getValType(ConstantVector *CP) {
  std::vector<Constant*> Elements;
  Elements.reserve(CP->getNumOperands());
  for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
    Elements.push_back(CP->getOperand(i));
  return Elements;
}

void ConstantVector::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                 Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  
  std::vector<Constant*> Values;
  Values.reserve(getNumOperands());  // Build replacement array...
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    Constant *Val = getOperand(i);
    if (Val == From) Val = cast<Constant>(To);
    Values.push_back(Val);
  }
  
  Constant *Replacement = get(getType(), Values);
  assert(Replacement != this && "I didn't contain From!");
  
  // Everyone using this now uses the replacement.
  uncheckedReplaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}

void ConstantExpr::replaceUsesOfWithOnConstant(Value *From, Value *ToV,
                                               Use *U) {
  assert(isa<Constant>(ToV) && "Cannot make Constant refer to non-constant!");
  Constant *To = cast<Constant>(ToV);
  
  Constant *Replacement = 0;
  if (getOpcode() == Instruction::GetElementPtr) {
    SmallVector<Constant*, 8> Indices;
    Constant *Pointer = getOperand(0);
    Indices.reserve(getNumOperands()-1);
    if (Pointer == From) Pointer = To;
    
    for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
      Constant *Val = getOperand(i);
      if (Val == From) Val = To;
      Indices.push_back(Val);
    }
    Replacement = ConstantExpr::getGetElementPtr(Pointer,
                                                 &Indices[0], Indices.size());
  } else if (getOpcode() == Instruction::ExtractValue) {
    Constant *Agg = getOperand(0);
    if (Agg == From) Agg = To;
    
    const SmallVector<unsigned, 4> &Indices = getIndices();
    Replacement = ConstantExpr::getExtractValue(Agg,
                                                &Indices[0], Indices.size());
  } else if (getOpcode() == Instruction::InsertValue) {
    Constant *Agg = getOperand(0);
    Constant *Val = getOperand(1);
    if (Agg == From) Agg = To;
    if (Val == From) Val = To;
    
    const SmallVector<unsigned, 4> &Indices = getIndices();
    Replacement = ConstantExpr::getInsertValue(Agg, Val,
                                               &Indices[0], Indices.size());
  } else if (isCast()) {
    assert(getOperand(0) == From && "Cast only has one use!");
    Replacement = ConstantExpr::getCast(getOpcode(), To, getType());
  } else if (getOpcode() == Instruction::Select) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    Constant *C3 = getOperand(2);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    if (C3 == From) C3 = To;
    Replacement = ConstantExpr::getSelect(C1, C2, C3);
  } else if (getOpcode() == Instruction::ExtractElement) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    Replacement = ConstantExpr::getExtractElement(C1, C2);
  } else if (getOpcode() == Instruction::InsertElement) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    Constant *C3 = getOperand(1);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    if (C3 == From) C3 = To;
    Replacement = ConstantExpr::getInsertElement(C1, C2, C3);
  } else if (getOpcode() == Instruction::ShuffleVector) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    Constant *C3 = getOperand(2);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    if (C3 == From) C3 = To;
    Replacement = ConstantExpr::getShuffleVector(C1, C2, C3);
  } else if (isCompare()) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    if (getOpcode() == Instruction::ICmp)
      Replacement = ConstantExpr::getICmp(getPredicate(), C1, C2);
    else {
      assert(getOpcode() == Instruction::FCmp);
      Replacement = ConstantExpr::getFCmp(getPredicate(), C1, C2);
    }
  } else if (getNumOperands() == 2) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    Replacement = ConstantExpr::get(getOpcode(), C1, C2);
  } else {
    llvm_unreachable("Unknown ConstantExpr type!");
    return;
  }
  
  assert(Replacement != this && "I didn't contain From!");
  
  // Everyone using this now uses the replacement.
  uncheckedReplaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}

