//===-- Constants.cpp - Implement Constant nodes --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Constant* classes.
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
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cstdarg>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                              Constant Class
//===----------------------------------------------------------------------===//

bool Constant::isNegativeZeroValue() const {
  // Floating point values have an explicit -0.0 value.
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(this))
    return CFP->isZero() && CFP->isNegative();
  
  // Otherwise, just use +0.0.
  return isNullValue();
}

bool Constant::isNullValue() const {
  // 0 is null.
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(this))
    return CI->isZero();
  
  // +0.0 is null.
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(this))
    return CFP->isZero() && !CFP->isNegative();

  // constant zero is zero for aggregates and cpnull is null for pointers.
  return isa<ConstantAggregateZero>(this) || isa<ConstantPointerNull>(this);
}

// Constructor to create a '0' constant of arbitrary type...
Constant *Constant::getNullValue(Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID:
    return ConstantInt::get(Ty, 0);
  case Type::FloatTyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat::getZero(APFloat::IEEEsingle));
  case Type::DoubleTyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat::getZero(APFloat::IEEEdouble));
  case Type::X86_FP80TyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat::getZero(APFloat::x87DoubleExtended));
  case Type::FP128TyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat::getZero(APFloat::IEEEquad));
  case Type::PPC_FP128TyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat(APInt::getNullValue(128)));
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

Constant *Constant::getIntegerValue(Type *Ty, const APInt &V) {
  Type *ScalarTy = Ty->getScalarType();

  // Create the base integer constant.
  Constant *C = ConstantInt::get(Ty->getContext(), V);

  // Convert an integer to a pointer, if necessary.
  if (PointerType *PTy = dyn_cast<PointerType>(ScalarTy))
    C = ConstantExpr::getIntToPtr(C, PTy);

  // Broadcast a scalar to a vector, if necessary.
  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    C = ConstantVector::get(std::vector<Constant *>(VTy->getNumElements(), C));

  return C;
}

Constant *Constant::getAllOnesValue(Type *Ty) {
  if (IntegerType *ITy = dyn_cast<IntegerType>(Ty))
    return ConstantInt::get(Ty->getContext(),
                            APInt::getAllOnesValue(ITy->getBitWidth()));

  if (Ty->isFloatingPointTy()) {
    APFloat FL = APFloat::getAllOnesValue(Ty->getPrimitiveSizeInBits(),
                                          !Ty->isPPC_FP128Ty());
    return ConstantFP::get(Ty->getContext(), FL);
  }

  SmallVector<Constant*, 16> Elts;
  VectorType *VTy = cast<VectorType>(Ty);
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
      dbgs() << "While deleting: " << *this
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
    if (CE->getOperand(i)->canTrap()) 
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
    if (!isa<ConstantInt>(CE->getOperand(1)) ||CE->getOperand(1)->isNullValue())
      return true;
    return false;
  }
}

/// isConstantUsed - Return true if the constant has users other than constant
/// exprs and other dangling things.
bool Constant::isConstantUsed() const {
  for (const_use_iterator UI = use_begin(), E = use_end(); UI != E; ++UI) {
    const Constant *UC = dyn_cast<Constant>(*UI);
    if (UC == 0 || isa<GlobalValue>(UC))
      return true;
    
    if (UC->isConstantUsed())
      return true;
  }
  return false;
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
  
  if (const BlockAddress *BA = dyn_cast<BlockAddress>(this))
    return BA->getFunction()->getRelocationInfo();
  
  // While raw uses of blockaddress need to be relocated, differences between
  // two of them don't when they are for labels in the same function.  This is a
  // common idiom when creating a table for the indirect goto extension, so we
  // handle it efficiently here.
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(this))
    if (CE->getOpcode() == Instruction::Sub) {
      ConstantExpr *LHS = dyn_cast<ConstantExpr>(CE->getOperand(0));
      ConstantExpr *RHS = dyn_cast<ConstantExpr>(CE->getOperand(1));
      if (LHS && RHS &&
          LHS->getOpcode() == Instruction::PtrToInt &&
          RHS->getOpcode() == Instruction::PtrToInt &&
          isa<BlockAddress>(LHS->getOperand(0)) &&
          isa<BlockAddress>(RHS->getOperand(0)) &&
          cast<BlockAddress>(LHS->getOperand(0))->getFunction() ==
            cast<BlockAddress>(RHS->getOperand(0))->getFunction())
        return NoRelocation;
    }
  
  PossibleRelocationsTy Result = NoRelocation;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    Result = std::max(Result,
                      cast<Constant>(getOperand(i))->getRelocationInfo());
  
  return Result;
}


/// getVectorElements - This method, which is only valid on constant of vector
/// type, returns the elements of the vector in the specified smallvector.
/// This handles breaking down a vector undef into undef elements, etc.  For
/// constant exprs and other cases we can't handle, we return an empty vector.
void Constant::getVectorElements(SmallVectorImpl<Constant*> &Elts) const {
  assert(getType()->isVectorTy() && "Not a vector constant!");
  
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(this)) {
    for (unsigned i = 0, e = CV->getNumOperands(); i != e; ++i)
      Elts.push_back(CV->getOperand(i));
    return;
  }
  
  VectorType *VT = cast<VectorType>(getType());
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


/// removeDeadUsersOfConstant - If the specified constantexpr is dead, remove
/// it.  This involves recursively eliminating any dead users of the
/// constantexpr.
static bool removeDeadUsersOfConstant(const Constant *C) {
  if (isa<GlobalValue>(C)) return false; // Cannot remove this
  
  while (!C->use_empty()) {
    const Constant *User = dyn_cast<Constant>(C->use_back());
    if (!User) return false; // Non-constant usage;
    if (!removeDeadUsersOfConstant(User))
      return false; // Constant wasn't dead
  }
  
  const_cast<Constant*>(C)->destroyConstant();
  return true;
}


/// removeDeadConstantUsers - If there are any dead constant users dangling
/// off of this constant, remove them.  This method is useful for clients
/// that want to check to see if a global is unused, but don't want to deal
/// with potentially dead constants hanging off of the globals.
void Constant::removeDeadConstantUsers() const {
  Value::const_use_iterator I = use_begin(), E = use_end();
  Value::const_use_iterator LastNonDeadUser = E;
  while (I != E) {
    const Constant *User = dyn_cast<Constant>(*I);
    if (User == 0) {
      LastNonDeadUser = I;
      ++I;
      continue;
    }
    
    if (!removeDeadUsersOfConstant(User)) {
      // If the constant wasn't dead, remember that this was the last live use
      // and move on to the next constant.
      LastNonDeadUser = I;
      ++I;
      continue;
    }
    
    // If the constant was dead, then the iterator is invalidated.
    if (LastNonDeadUser == E) {
      I = use_begin();
      if (I == E) break;
    } else {
      I = LastNonDeadUser;
      ++I;
    }
  }
}



//===----------------------------------------------------------------------===//
//                                ConstantInt
//===----------------------------------------------------------------------===//

ConstantInt::ConstantInt(IntegerType *Ty, const APInt& V)
  : Constant(Ty, ConstantIntVal, 0, 0), Val(V) {
  assert(V.getBitWidth() == Ty->getBitWidth() && "Invalid constant for type");
}

ConstantInt *ConstantInt::getTrue(LLVMContext &Context) {
  LLVMContextImpl *pImpl = Context.pImpl;
  if (!pImpl->TheTrueVal)
    pImpl->TheTrueVal = ConstantInt::get(Type::getInt1Ty(Context), 1);
  return pImpl->TheTrueVal;
}

ConstantInt *ConstantInt::getFalse(LLVMContext &Context) {
  LLVMContextImpl *pImpl = Context.pImpl;
  if (!pImpl->TheFalseVal)
    pImpl->TheFalseVal = ConstantInt::get(Type::getInt1Ty(Context), 0);
  return pImpl->TheFalseVal;
}

Constant *ConstantInt::getTrue(Type *Ty) {
  VectorType *VTy = dyn_cast<VectorType>(Ty);
  if (!VTy) {
    assert(Ty->isIntegerTy(1) && "True must be i1 or vector of i1.");
    return ConstantInt::getTrue(Ty->getContext());
  }
  assert(VTy->getElementType()->isIntegerTy(1) &&
         "True must be vector of i1 or i1.");
  SmallVector<Constant*, 16> Splat(VTy->getNumElements(),
                                   ConstantInt::getTrue(Ty->getContext()));
  return ConstantVector::get(Splat);
}

Constant *ConstantInt::getFalse(Type *Ty) {
  VectorType *VTy = dyn_cast<VectorType>(Ty);
  if (!VTy) {
    assert(Ty->isIntegerTy(1) && "False must be i1 or vector of i1.");
    return ConstantInt::getFalse(Ty->getContext());
  }
  assert(VTy->getElementType()->isIntegerTy(1) &&
         "False must be vector of i1 or i1.");
  SmallVector<Constant*, 16> Splat(VTy->getNumElements(),
                                   ConstantInt::getFalse(Ty->getContext()));
  return ConstantVector::get(Splat);
}


// Get a ConstantInt from an APInt. Note that the value stored in the DenseMap 
// as the key, is a DenseMapAPIntKeyInfo::KeyTy which has provided the
// operator== and operator!= to ensure that the DenseMap doesn't attempt to
// compare APInt's of different widths, which would violate an APInt class
// invariant which generates an assertion.
ConstantInt *ConstantInt::get(LLVMContext &Context, const APInt &V) {
  // Get the corresponding integer type for the bit width of the value.
  IntegerType *ITy = IntegerType::get(Context, V.getBitWidth());
  // get an existing value or the insertion position
  DenseMapAPIntKeyInfo::KeyTy Key(V, ITy);
  ConstantInt *&Slot = Context.pImpl->IntConstants[Key]; 
  if (!Slot) Slot = new ConstantInt(ITy, V);
  return Slot;
}

Constant *ConstantInt::get(Type *Ty, uint64_t V, bool isSigned) {
  Constant *C = get(cast<IntegerType>(Ty->getScalarType()), V, isSigned);

  // For vectors, broadcast the value.
  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::get(SmallVector<Constant*,
                                           16>(VTy->getNumElements(), C));

  return C;
}

ConstantInt* ConstantInt::get(IntegerType* Ty, uint64_t V, 
                              bool isSigned) {
  return get(Ty->getContext(), APInt(Ty->getBitWidth(), V, isSigned));
}

ConstantInt* ConstantInt::getSigned(IntegerType* Ty, int64_t V) {
  return get(Ty, V, true);
}

Constant *ConstantInt::getSigned(Type *Ty, int64_t V) {
  return get(Ty, V, true);
}

Constant *ConstantInt::get(Type* Ty, const APInt& V) {
  ConstantInt *C = get(Ty->getContext(), V);
  assert(C->getType() == Ty->getScalarType() &&
         "ConstantInt type doesn't match the type implied by its value!");

  // For vectors, broadcast the value.
  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::get(
      SmallVector<Constant *, 16>(VTy->getNumElements(), C));

  return C;
}

ConstantInt* ConstantInt::get(IntegerType* Ty, StringRef Str,
                              uint8_t radix) {
  return get(Ty->getContext(), APInt(Ty->getBitWidth(), Str, radix));
}

//===----------------------------------------------------------------------===//
//                                ConstantFP
//===----------------------------------------------------------------------===//

static const fltSemantics *TypeToFloatSemantics(Type *Ty) {
  if (Ty->isFloatTy())
    return &APFloat::IEEEsingle;
  if (Ty->isDoubleTy())
    return &APFloat::IEEEdouble;
  if (Ty->isX86_FP80Ty())
    return &APFloat::x87DoubleExtended;
  else if (Ty->isFP128Ty())
    return &APFloat::IEEEquad;
  
  assert(Ty->isPPC_FP128Ty() && "Unknown FP format");
  return &APFloat::PPCDoubleDouble;
}

/// get() - This returns a constant fp for the specified value in the
/// specified type.  This should only be used for simple constant values like
/// 2.0/1.0 etc, that are known-valid both as double and as the target format.
Constant *ConstantFP::get(Type* Ty, double V) {
  LLVMContext &Context = Ty->getContext();
  
  APFloat FV(V);
  bool ignored;
  FV.convert(*TypeToFloatSemantics(Ty->getScalarType()),
             APFloat::rmNearestTiesToEven, &ignored);
  Constant *C = get(Context, FV);

  // For vectors, broadcast the value.
  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::get(
      SmallVector<Constant *, 16>(VTy->getNumElements(), C));

  return C;
}


Constant *ConstantFP::get(Type* Ty, StringRef Str) {
  LLVMContext &Context = Ty->getContext();

  APFloat FV(*TypeToFloatSemantics(Ty->getScalarType()), Str);
  Constant *C = get(Context, FV);

  // For vectors, broadcast the value.
  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::get(
      SmallVector<Constant *, 16>(VTy->getNumElements(), C));

  return C; 
}


ConstantFP* ConstantFP::getNegativeZero(Type* Ty) {
  LLVMContext &Context = Ty->getContext();
  APFloat apf = cast <ConstantFP>(Constant::getNullValue(Ty))->getValueAPF();
  apf.changeSign();
  return get(Context, apf);
}


Constant *ConstantFP::getZeroValueForNegation(Type* Ty) {
  if (VectorType *PTy = dyn_cast<VectorType>(Ty))
    if (PTy->getElementType()->isFloatingPointTy()) {
      SmallVector<Constant*, 16> zeros(PTy->getNumElements(),
                           getNegativeZero(PTy->getElementType()));
      return ConstantVector::get(zeros);
    }

  if (Ty->isFloatingPointTy()) 
    return getNegativeZero(Ty);

  return Constant::getNullValue(Ty);
}


// ConstantFP accessors.
ConstantFP* ConstantFP::get(LLVMContext &Context, const APFloat& V) {
  DenseMapAPFloatKeyInfo::KeyTy Key(V);
  
  LLVMContextImpl* pImpl = Context.pImpl;
  
  ConstantFP *&Slot = pImpl->FPConstants[Key];
    
  if (!Slot) {
    Type *Ty;
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
    Slot = new ConstantFP(Ty, V);
  }
  
  return Slot;
}

ConstantFP *ConstantFP::getInfinity(Type *Ty, bool Negative) {
  const fltSemantics &Semantics = *TypeToFloatSemantics(Ty);
  return ConstantFP::get(Ty->getContext(),
                         APFloat::getInf(Semantics, Negative));
}

ConstantFP::ConstantFP(Type *Ty, const APFloat& V)
  : Constant(Ty, ConstantFPVal, 0, 0), Val(V) {
  assert(&V.getSemantics() == TypeToFloatSemantics(Ty) &&
         "FP type Mismatch");
}

bool ConstantFP::isExactlyValue(const APFloat &V) const {
  return Val.bitwiseIsEqual(V);
}

//===----------------------------------------------------------------------===//
//                            ConstantXXX Classes
//===----------------------------------------------------------------------===//


ConstantArray::ConstantArray(ArrayType *T,
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
    assert(C->getType() == T->getElementType() &&
           "Initializer for array element doesn't match array element type!");
    *OL = C;
  }
}

Constant *ConstantArray::get(ArrayType *Ty, ArrayRef<Constant*> V) {
  for (unsigned i = 0, e = V.size(); i != e; ++i) {
    assert(V[i]->getType() == Ty->getElementType() &&
           "Wrong type in array element initializer");
  }
  LLVMContextImpl *pImpl = Ty->getContext().pImpl;
  // If this is an all-zero array, return a ConstantAggregateZero object
  if (!V.empty()) {
    Constant *C = V[0];
    if (!C->isNullValue())
      return pImpl->ArrayConstants.getOrCreate(Ty, V);
    
    for (unsigned i = 1, e = V.size(); i != e; ++i)
      if (V[i] != C)
        return pImpl->ArrayConstants.getOrCreate(Ty, V);
  }
  
  return ConstantAggregateZero::get(Ty);
}

/// ConstantArray::get(const string&) - Return an array that is initialized to
/// contain the specified string.  If length is zero then a null terminator is 
/// added to the specified string so that it may be used in a natural way. 
/// Otherwise, the length parameter specifies how much of the string to use 
/// and it won't be null terminated.
///
Constant *ConstantArray::get(LLVMContext &Context, StringRef Str,
                             bool AddNull) {
  std::vector<Constant*> ElementVals;
  ElementVals.reserve(Str.size() + size_t(AddNull));
  for (unsigned i = 0; i < Str.size(); ++i)
    ElementVals.push_back(ConstantInt::get(Type::getInt8Ty(Context), Str[i]));

  // Add a null terminator to the string...
  if (AddNull) {
    ElementVals.push_back(ConstantInt::get(Type::getInt8Ty(Context), 0));
  }

  ArrayType *ATy = ArrayType::get(Type::getInt8Ty(Context), ElementVals.size());
  return get(ATy, ElementVals);
}

/// getTypeForElements - Return an anonymous struct type to use for a constant
/// with the specified set of elements.  The list must not be empty.
StructType *ConstantStruct::getTypeForElements(LLVMContext &Context,
                                               ArrayRef<Constant*> V,
                                               bool Packed) {
  SmallVector<Type*, 16> EltTypes;
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    EltTypes.push_back(V[i]->getType());
  
  return StructType::get(Context, EltTypes, Packed);
}


StructType *ConstantStruct::getTypeForElements(ArrayRef<Constant*> V,
                                               bool Packed) {
  assert(!V.empty() &&
         "ConstantStruct::getTypeForElements cannot be called on empty list");
  return getTypeForElements(V[0]->getContext(), V, Packed);
}


ConstantStruct::ConstantStruct(StructType *T,
                               const std::vector<Constant*> &V)
  : Constant(T, ConstantStructVal,
             OperandTraits<ConstantStruct>::op_end(this) - V.size(),
             V.size()) {
  assert((T->isOpaque() || V.size() == T->getNumElements()) &&
         "Invalid initializer vector for constant structure");
  Use *OL = OperandList;
  for (std::vector<Constant*>::const_iterator I = V.begin(), E = V.end();
       I != E; ++I, ++OL) {
    Constant *C = *I;
    assert((T->isOpaque() || C->getType() == T->getElementType(I-V.begin())) &&
           "Initializer for struct element doesn't match struct element type!");
    *OL = C;
  }
}

// ConstantStruct accessors.
Constant *ConstantStruct::get(StructType *ST, ArrayRef<Constant*> V) {
  // Create a ConstantAggregateZero value if all elements are zeros.
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    if (!V[i]->isNullValue())
      return ST->getContext().pImpl->StructConstants.getOrCreate(ST, V);

  assert((ST->isOpaque() || ST->getNumElements() == V.size()) &&
         "Incorrect # elements specified to ConstantStruct::get");
  return ConstantAggregateZero::get(ST);
}

Constant* ConstantStruct::get(StructType *T, ...) {
  va_list ap;
  SmallVector<Constant*, 8> Values;
  va_start(ap, T);
  while (Constant *Val = va_arg(ap, llvm::Constant*))
    Values.push_back(Val);
  va_end(ap);
  return get(T, Values);
}

ConstantVector::ConstantVector(VectorType *T,
                               const std::vector<Constant*> &V)
  : Constant(T, ConstantVectorVal,
             OperandTraits<ConstantVector>::op_end(this) - V.size(),
             V.size()) {
  Use *OL = OperandList;
  for (std::vector<Constant*>::const_iterator I = V.begin(), E = V.end();
       I != E; ++I, ++OL) {
    Constant *C = *I;
    assert(C->getType() == T->getElementType() &&
           "Initializer for vector element doesn't match vector element type!");
    *OL = C;
  }
}

// ConstantVector accessors.
Constant *ConstantVector::get(ArrayRef<Constant*> V) {
  assert(!V.empty() && "Vectors can't be empty");
  VectorType *T = VectorType::get(V.front()->getType(), V.size());
  LLVMContextImpl *pImpl = T->getContext().pImpl;

  // If this is an all-undef or all-zero vector, return a
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
    
  return pImpl->VectorConstants.getOrCreate(T, V);
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

bool ConstantExpr::isGEPWithNoNotionalOverIndexing() const {
  if (getOpcode() != Instruction::GetElementPtr) return false;

  gep_type_iterator GEPI = gep_type_begin(this), E = gep_type_end(this);
  User::const_op_iterator OI = llvm::next(this->op_begin());

  // Skip the first index, as it has no static limit.
  ++GEPI;
  ++OI;

  // The remaining indices must be compile-time known integers within the
  // bounds of the corresponding notional static array types.
  for (; GEPI != E; ++GEPI, ++OI) {
    ConstantInt *CI = dyn_cast<ConstantInt>(*OI);
    if (!CI) return false;
    if (ArrayType *ATy = dyn_cast<ArrayType>(*GEPI))
      if (CI->getValue().getActiveBits() > 64 ||
          CI->getZExtValue() >= ATy->getNumElements())
        return false;
  }

  // All the indices checked out.
  return true;
}

bool ConstantExpr::hasIndices() const {
  return getOpcode() == Instruction::ExtractValue ||
         getOpcode() == Instruction::InsertValue;
}

ArrayRef<unsigned> ConstantExpr::getIndices() const {
  if (const ExtractValueConstantExpr *EVCE =
        dyn_cast<ExtractValueConstantExpr>(this))
    return EVCE->Indices;

  return cast<InsertValueConstantExpr>(this)->Indices;
}

unsigned ConstantExpr::getPredicate() const {
  assert(isCompare());
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
      return cast<GEPOperator>(this)->isInBounds() ?
        ConstantExpr::getInBoundsGetElementPtr(Op, &Ops[0], Ops.size()) :
        ConstantExpr::getGetElementPtr(Op, &Ops[0], Ops.size());
    Ops[OpNo-1] = Op;
    return cast<GEPOperator>(this)->isInBounds() ?
      ConstantExpr::getInBoundsGetElementPtr(getOperand(0), &Ops[0],Ops.size()):
      ConstantExpr::getGetElementPtr(getOperand(0), &Ops[0], Ops.size());
  }
  default:
    assert(getNumOperands() == 2 && "Must be binary operator?");
    Op0 = (OpNo == 0) ? Op : getOperand(0);
    Op1 = (OpNo == 1) ? Op : getOperand(1);
    return ConstantExpr::get(getOpcode(), Op0, Op1, SubclassOptionalData);
  }
}

/// getWithOperands - This returns the current constant expression with the
/// operands replaced with the specified values.  The specified array must
/// have the same number of operands as our current one.
Constant *ConstantExpr::
getWithOperands(ArrayRef<Constant*> Ops, Type *Ty) const {
  assert(Ops.size() == getNumOperands() && "Operand count mismatch!");
  bool AnyChange = Ty != getType();
  for (unsigned i = 0; i != Ops.size(); ++i)
    AnyChange |= Ops[i] != getOperand(i);
  
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
    return ConstantExpr::getCast(getOpcode(), Ops[0], Ty);
  case Instruction::Select:
    return ConstantExpr::getSelect(Ops[0], Ops[1], Ops[2]);
  case Instruction::InsertElement:
    return ConstantExpr::getInsertElement(Ops[0], Ops[1], Ops[2]);
  case Instruction::ExtractElement:
    return ConstantExpr::getExtractElement(Ops[0], Ops[1]);
  case Instruction::ShuffleVector:
    return ConstantExpr::getShuffleVector(Ops[0], Ops[1], Ops[2]);
  case Instruction::GetElementPtr:
    return cast<GEPOperator>(this)->isInBounds() ?
      ConstantExpr::getInBoundsGetElementPtr(Ops[0], &Ops[1], Ops.size()-1) :
      ConstantExpr::getGetElementPtr(Ops[0], &Ops[1], Ops.size()-1);
  case Instruction::ICmp:
  case Instruction::FCmp:
    return ConstantExpr::getCompare(getPredicate(), Ops[0], Ops[1]);
  default:
    assert(getNumOperands() == 2 && "Must be binary operator?");
    return ConstantExpr::get(getOpcode(), Ops[0], Ops[1], SubclassOptionalData);
  }
}


//===----------------------------------------------------------------------===//
//                      isValueValidForType implementations

bool ConstantInt::isValueValidForType(Type *Ty, uint64_t Val) {
  unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth(); // assert okay
  if (Ty == Type::getInt1Ty(Ty->getContext()))
    return Val == 0 || Val == 1;
  if (NumBits >= 64)
    return true; // always true, has to fit in largest type
  uint64_t Max = (1ll << NumBits) - 1;
  return Val <= Max;
}

bool ConstantInt::isValueValidForType(Type *Ty, int64_t Val) {
  unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth(); // assert okay
  if (Ty == Type::getInt1Ty(Ty->getContext()))
    return Val == 0 || Val == 1 || Val == -1;
  if (NumBits >= 64)
    return true; // always true, has to fit in largest type
  int64_t Min = -(1ll << (NumBits-1));
  int64_t Max = (1ll << (NumBits-1)) - 1;
  return (Val >= Min && Val <= Max);
}

bool ConstantFP::isValueValidForType(Type *Ty, const APFloat& Val) {
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

ConstantAggregateZero* ConstantAggregateZero::get(Type* Ty) {
  assert((Ty->isStructTy() || Ty->isArrayTy() || Ty->isVectorTy()) &&
         "Cannot create an aggregate zero of non-aggregate type!");
  
  LLVMContextImpl *pImpl = Ty->getContext().pImpl;
  return pImpl->AggZeroConstants.getOrCreate(Ty, 0);
}

/// destroyConstant - Remove the constant from the constant table...
///
void ConstantAggregateZero::destroyConstant() {
  getType()->getContext().pImpl->AggZeroConstants.remove(this);
  destroyConstantImpl();
}

/// destroyConstant - Remove the constant from the constant table...
///
void ConstantArray::destroyConstant() {
  getType()->getContext().pImpl->ArrayConstants.remove(this);
  destroyConstantImpl();
}

/// isString - This method returns true if the array is an array of i8, and 
/// if the elements of the array are all ConstantInt's.
bool ConstantArray::isString() const {
  // Check the element type for i8...
  if (!getType()->getElementType()->isIntegerTy(8))
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
  if (!getType()->getElementType()->isIntegerTy(8))
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


/// convertToString - Helper function for getAsString() and getAsCString().
static std::string convertToString(const User *U, unsigned len) {
  std::string Result;
  Result.reserve(len);
  for (unsigned i = 0; i != len; ++i)
    Result.push_back((char)cast<ConstantInt>(U->getOperand(i))->getZExtValue());
  return Result;
}

/// getAsString - If this array is isString(), then this method converts the
/// array to an std::string and returns it.  Otherwise, it asserts out.
///
std::string ConstantArray::getAsString() const {
  assert(isString() && "Not a string!");
  return convertToString(this, getNumOperands());
}


/// getAsCString - If this array is isCString(), then this method converts the
/// array (without the trailing null byte) to an std::string and returns it.
/// Otherwise, it asserts out.
///
std::string ConstantArray::getAsCString() const {
  assert(isCString() && "Not a string!");
  return convertToString(this, getNumOperands() - 1);
}


//---- ConstantStruct::get() implementation...
//

// destroyConstant - Remove the constant from the constant table...
//
void ConstantStruct::destroyConstant() {
  getType()->getContext().pImpl->StructConstants.remove(this);
  destroyConstantImpl();
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantVector::destroyConstant() {
  getType()->getContext().pImpl->VectorConstants.remove(this);
  destroyConstantImpl();
}

/// This function will return true iff every element in this vector constant
/// is set to all ones.
/// @returns true iff this constant's elements are all set to all ones.
/// @brief Determine if the value is all ones.
bool ConstantVector::isAllOnesValue() const {
  // Check out first element.
  const Constant *Elt = getOperand(0);
  const ConstantInt *CI = dyn_cast<ConstantInt>(Elt);
  if (!CI || !CI->isAllOnesValue()) return false;
  // Then make sure all remaining elements point to the same value.
  for (unsigned I = 1, E = getNumOperands(); I < E; ++I)
    if (getOperand(I) != Elt)
      return false;
  
  return true;
}

/// getSplatValue - If this is a splat constant, where all of the
/// elements have the same value, return that value. Otherwise return null.
Constant *ConstantVector::getSplatValue() const {
  // Check out first element.
  Constant *Elt = getOperand(0);
  // Then make sure all remaining elements point to the same value.
  for (unsigned I = 1, E = getNumOperands(); I < E; ++I)
    if (getOperand(I) != Elt)
      return 0;
  return Elt;
}

//---- ConstantPointerNull::get() implementation.
//

ConstantPointerNull *ConstantPointerNull::get(PointerType *Ty) {
  return Ty->getContext().pImpl->NullPtrConstants.getOrCreate(Ty, 0);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantPointerNull::destroyConstant() {
  getType()->getContext().pImpl->NullPtrConstants.remove(this);
  destroyConstantImpl();
}


//---- UndefValue::get() implementation.
//

UndefValue *UndefValue::get(Type *Ty) {
  return Ty->getContext().pImpl->UndefValueConstants.getOrCreate(Ty, 0);
}

// destroyConstant - Remove the constant from the constant table.
//
void UndefValue::destroyConstant() {
  getType()->getContext().pImpl->UndefValueConstants.remove(this);
  destroyConstantImpl();
}

//---- BlockAddress::get() implementation.
//

BlockAddress *BlockAddress::get(BasicBlock *BB) {
  assert(BB->getParent() != 0 && "Block must have a parent");
  return get(BB->getParent(), BB);
}

BlockAddress *BlockAddress::get(Function *F, BasicBlock *BB) {
  BlockAddress *&BA =
    F->getContext().pImpl->BlockAddresses[std::make_pair(F, BB)];
  if (BA == 0)
    BA = new BlockAddress(F, BB);
  
  assert(BA->getFunction() == F && "Basic block moved between functions");
  return BA;
}

BlockAddress::BlockAddress(Function *F, BasicBlock *BB)
: Constant(Type::getInt8PtrTy(F->getContext()), Value::BlockAddressVal,
           &Op<0>(), 2) {
  setOperand(0, F);
  setOperand(1, BB);
  BB->AdjustBlockAddressRefCount(1);
}


// destroyConstant - Remove the constant from the constant table.
//
void BlockAddress::destroyConstant() {
  getFunction()->getType()->getContext().pImpl
    ->BlockAddresses.erase(std::make_pair(getFunction(), getBasicBlock()));
  getBasicBlock()->AdjustBlockAddressRefCount(-1);
  destroyConstantImpl();
}

void BlockAddress::replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U) {
  // This could be replacing either the Basic Block or the Function.  In either
  // case, we have to remove the map entry.
  Function *NewF = getFunction();
  BasicBlock *NewBB = getBasicBlock();
  
  if (U == &Op<0>())
    NewF = cast<Function>(To);
  else
    NewBB = cast<BasicBlock>(To);
  
  // See if the 'new' entry already exists, if not, just update this in place
  // and return early.
  BlockAddress *&NewBA =
    getContext().pImpl->BlockAddresses[std::make_pair(NewF, NewBB)];
  if (NewBA == 0) {
    getBasicBlock()->AdjustBlockAddressRefCount(-1);
    
    // Remove the old entry, this can't cause the map to rehash (just a
    // tombstone will get added).
    getContext().pImpl->BlockAddresses.erase(std::make_pair(getFunction(),
                                                            getBasicBlock()));
    NewBA = this;
    setOperand(0, NewF);
    setOperand(1, NewBB);
    getBasicBlock()->AdjustBlockAddressRefCount(1);
    return;
  }

  // Otherwise, I do need to replace this with an existing value.
  assert(NewBA != this && "I didn't contain From!");
  
  // Everyone using this now uses the replacement.
  replaceAllUsesWith(NewBA);
  
  destroyConstant();
}

//---- ConstantExpr::get() implementations.
//

/// This is a utility function to handle folding of casts and lookup of the
/// cast in the ExprConstants map. It is used by the various get* methods below.
static inline Constant *getFoldedCast(
  Instruction::CastOps opc, Constant *C, Type *Ty) {
  assert(Ty->isFirstClassType() && "Cannot cast to an aggregate type!");
  // Fold a few common cases
  if (Constant *FC = ConstantFoldCastInstruction(opc, C, Ty))
    return FC;

  LLVMContextImpl *pImpl = Ty->getContext().pImpl;

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> argVec(1, C);
  ExprMapKeyType Key(opc, argVec);
  
  return pImpl->ExprConstants.getOrCreate(Ty, Key);
}
 
Constant *ConstantExpr::getCast(unsigned oc, Constant *C, Type *Ty) {
  Instruction::CastOps opc = Instruction::CastOps(oc);
  assert(Instruction::isCast(opc) && "opcode out of range");
  assert(C && Ty && "Null arguments to getCast");
  assert(CastInst::castIsValid(opc, C, Ty) && "Invalid constantexpr cast!");

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

Constant *ConstantExpr::getZExtOrBitCast(Constant *C, Type *Ty) {
  if (C->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return getBitCast(C, Ty);
  return getZExt(C, Ty);
}

Constant *ConstantExpr::getSExtOrBitCast(Constant *C, Type *Ty) {
  if (C->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return getBitCast(C, Ty);
  return getSExt(C, Ty);
}

Constant *ConstantExpr::getTruncOrBitCast(Constant *C, Type *Ty) {
  if (C->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return getBitCast(C, Ty);
  return getTrunc(C, Ty);
}

Constant *ConstantExpr::getPointerCast(Constant *S, Type *Ty) {
  assert(S->getType()->isPointerTy() && "Invalid cast");
  assert((Ty->isIntegerTy() || Ty->isPointerTy()) && "Invalid cast");

  if (Ty->isIntegerTy())
    return getPtrToInt(S, Ty);
  return getBitCast(S, Ty);
}

Constant *ConstantExpr::getIntegerCast(Constant *C, Type *Ty, 
                                       bool isSigned) {
  assert(C->getType()->isIntOrIntVectorTy() &&
         Ty->isIntOrIntVectorTy() && "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::Trunc :
      (isSigned ? Instruction::SExt : Instruction::ZExt)));
  return getCast(opcode, C, Ty);
}

Constant *ConstantExpr::getFPCast(Constant *C, Type *Ty) {
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isFPOrFPVectorTy() &&
         "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  if (SrcBits == DstBits)
    return C; // Avoid a useless cast
  Instruction::CastOps opcode =
    (SrcBits > DstBits ? Instruction::FPTrunc : Instruction::FPExt);
  return getCast(opcode, C, Ty);
}

Constant *ConstantExpr::getTrunc(Constant *C, Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVectorTy() && "Trunc operand must be integer");
  assert(Ty->isIntOrIntVectorTy() && "Trunc produces only integral");
  assert(C->getType()->getScalarSizeInBits() > Ty->getScalarSizeInBits()&&
         "SrcTy must be larger than DestTy for Trunc!");

  return getFoldedCast(Instruction::Trunc, C, Ty);
}

Constant *ConstantExpr::getSExt(Constant *C, Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVectorTy() && "SExt operand must be integral");
  assert(Ty->isIntOrIntVectorTy() && "SExt produces only integer");
  assert(C->getType()->getScalarSizeInBits() < Ty->getScalarSizeInBits()&&
         "SrcTy must be smaller than DestTy for SExt!");

  return getFoldedCast(Instruction::SExt, C, Ty);
}

Constant *ConstantExpr::getZExt(Constant *C, Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVectorTy() && "ZEXt operand must be integral");
  assert(Ty->isIntOrIntVectorTy() && "ZExt produces only integer");
  assert(C->getType()->getScalarSizeInBits() < Ty->getScalarSizeInBits()&&
         "SrcTy must be smaller than DestTy for ZExt!");

  return getFoldedCast(Instruction::ZExt, C, Ty);
}

Constant *ConstantExpr::getFPTrunc(Constant *C, Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isFPOrFPVectorTy() &&
         C->getType()->getScalarSizeInBits() > Ty->getScalarSizeInBits()&&
         "This is an illegal floating point truncation!");
  return getFoldedCast(Instruction::FPTrunc, C, Ty);
}

Constant *ConstantExpr::getFPExtend(Constant *C, Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isFPOrFPVectorTy() &&
         C->getType()->getScalarSizeInBits() < Ty->getScalarSizeInBits()&&
         "This is an illegal floating point extension!");
  return getFoldedCast(Instruction::FPExt, C, Ty);
}

Constant *ConstantExpr::getUIToFP(Constant *C, Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVectorTy() && Ty->isFPOrFPVectorTy() &&
         "This is an illegal uint to floating point cast!");
  return getFoldedCast(Instruction::UIToFP, C, Ty);
}

Constant *ConstantExpr::getSIToFP(Constant *C, Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVectorTy() && Ty->isFPOrFPVectorTy() &&
         "This is an illegal sint to floating point cast!");
  return getFoldedCast(Instruction::SIToFP, C, Ty);
}

Constant *ConstantExpr::getFPToUI(Constant *C, Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isIntOrIntVectorTy() &&
         "This is an illegal floating point to uint cast!");
  return getFoldedCast(Instruction::FPToUI, C, Ty);
}

Constant *ConstantExpr::getFPToSI(Constant *C, Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isIntOrIntVectorTy() &&
         "This is an illegal floating point to sint cast!");
  return getFoldedCast(Instruction::FPToSI, C, Ty);
}

Constant *ConstantExpr::getPtrToInt(Constant *C, Type *DstTy) {
  assert(C->getType()->isPointerTy() && "PtrToInt source must be pointer");
  assert(DstTy->isIntegerTy() && "PtrToInt destination must be integral");
  return getFoldedCast(Instruction::PtrToInt, C, DstTy);
}

Constant *ConstantExpr::getIntToPtr(Constant *C, Type *DstTy) {
  assert(C->getType()->isIntegerTy() && "IntToPtr source must be integral");
  assert(DstTy->isPointerTy() && "IntToPtr destination must be a pointer");
  return getFoldedCast(Instruction::IntToPtr, C, DstTy);
}

Constant *ConstantExpr::getBitCast(Constant *C, Type *DstTy) {
  assert(CastInst::castIsValid(Instruction::BitCast, C, DstTy) &&
         "Invalid constantexpr bitcast!");
  
  // It is common to ask for a bitcast of a value to its own type, handle this
  // speedily.
  if (C->getType() == DstTy) return C;
  
  return getFoldedCast(Instruction::BitCast, C, DstTy);
}

Constant *ConstantExpr::get(unsigned Opcode, Constant *C1, Constant *C2,
                            unsigned Flags) {
  // Check the operands for consistency first.
  assert(Opcode >= Instruction::BinaryOpsBegin &&
         Opcode <  Instruction::BinaryOpsEnd   &&
         "Invalid opcode in binary constant expression");
  assert(C1->getType() == C2->getType() &&
         "Operand types in binary constant expression should match");
  
#ifndef NDEBUG
  switch (Opcode) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVectorTy() &&
           "Tried to create an integer operation on a non-integer type!");
    break;
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isFPOrFPVectorTy() &&
           "Tried to create a floating-point operation on a "
           "non-floating-point type!");
    break;
  case Instruction::UDiv: 
  case Instruction::SDiv: 
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVectorTy() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::FDiv:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isFPOrFPVectorTy() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::URem: 
  case Instruction::SRem: 
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVectorTy() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::FRem:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isFPOrFPVectorTy() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVectorTy() &&
           "Tried to create a logical operation on a non-integral type!");
    break;
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVectorTy() &&
           "Tried to create a shift operation on a non-integer type!");
    break;
  default:
    break;
  }
#endif

  if (Constant *FC = ConstantFoldBinaryInstruction(Opcode, C1, C2))
    return FC;          // Fold a few common cases.
  
  std::vector<Constant*> argVec(1, C1);
  argVec.push_back(C2);
  ExprMapKeyType Key(Opcode, argVec, 0, Flags);
  
  LLVMContextImpl *pImpl = C1->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(C1->getType(), Key);
}

Constant *ConstantExpr::getSizeOf(Type* Ty) {
  // sizeof is implemented as: (i64) gep (Ty*)null, 1
  // Note that a non-inbounds gep is used, as null isn't within any object.
  Constant *GEPIdx = ConstantInt::get(Type::getInt32Ty(Ty->getContext()), 1);
  Constant *GEP = getGetElementPtr(
                 Constant::getNullValue(PointerType::getUnqual(Ty)), &GEPIdx, 1);
  return getPtrToInt(GEP, 
                     Type::getInt64Ty(Ty->getContext()));
}

Constant *ConstantExpr::getAlignOf(Type* Ty) {
  // alignof is implemented as: (i64) gep ({i1,Ty}*)null, 0, 1
  // Note that a non-inbounds gep is used, as null isn't within any object.
  Type *AligningTy = 
    StructType::get(Type::getInt1Ty(Ty->getContext()), Ty, NULL);
  Constant *NullPtr = Constant::getNullValue(AligningTy->getPointerTo());
  Constant *Zero = ConstantInt::get(Type::getInt64Ty(Ty->getContext()), 0);
  Constant *One = ConstantInt::get(Type::getInt32Ty(Ty->getContext()), 1);
  Constant *Indices[2] = { Zero, One };
  Constant *GEP = getGetElementPtr(NullPtr, Indices, 2);
  return getPtrToInt(GEP,
                     Type::getInt64Ty(Ty->getContext()));
}

Constant *ConstantExpr::getOffsetOf(StructType* STy, unsigned FieldNo) {
  return getOffsetOf(STy, ConstantInt::get(Type::getInt32Ty(STy->getContext()),
                                           FieldNo));
}

Constant *ConstantExpr::getOffsetOf(Type* Ty, Constant *FieldNo) {
  // offsetof is implemented as: (i64) gep (Ty*)null, 0, FieldNo
  // Note that a non-inbounds gep is used, as null isn't within any object.
  Constant *GEPIdx[] = {
    ConstantInt::get(Type::getInt64Ty(Ty->getContext()), 0),
    FieldNo
  };
  Constant *GEP = getGetElementPtr(
                Constant::getNullValue(PointerType::getUnqual(Ty)), GEPIdx, 2);
  return getPtrToInt(GEP,
                     Type::getInt64Ty(Ty->getContext()));
}

Constant *ConstantExpr::getCompare(unsigned short Predicate, 
                                   Constant *C1, Constant *C2) {
  assert(C1->getType() == C2->getType() && "Op types should be identical!");
  
  switch (Predicate) {
  default: llvm_unreachable("Invalid CmpInst predicate");
  case CmpInst::FCMP_FALSE: case CmpInst::FCMP_OEQ: case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_OGE:   case CmpInst::FCMP_OLT: case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_ONE:   case CmpInst::FCMP_ORD: case CmpInst::FCMP_UNO:
  case CmpInst::FCMP_UEQ:   case CmpInst::FCMP_UGT: case CmpInst::FCMP_UGE:
  case CmpInst::FCMP_ULT:   case CmpInst::FCMP_ULE: case CmpInst::FCMP_UNE:
  case CmpInst::FCMP_TRUE:
    return getFCmp(Predicate, C1, C2);
    
  case CmpInst::ICMP_EQ:  case CmpInst::ICMP_NE:  case CmpInst::ICMP_UGT:
  case CmpInst::ICMP_UGE: case CmpInst::ICMP_ULT: case CmpInst::ICMP_ULE:
  case CmpInst::ICMP_SGT: case CmpInst::ICMP_SGE: case CmpInst::ICMP_SLT:
  case CmpInst::ICMP_SLE:
    return getICmp(Predicate, C1, C2);
  }
}

Constant *ConstantExpr::getSelect(Constant *C, Constant *V1, Constant *V2) {
  assert(!SelectInst::areInvalidOperands(C, V1, V2)&&"Invalid select operands");

  if (Constant *SC = ConstantFoldSelectInstruction(C, V1, V2))
    return SC;        // Fold common cases

  std::vector<Constant*> argVec(3, C);
  argVec[1] = V1;
  argVec[2] = V2;
  ExprMapKeyType Key(Instruction::Select, argVec);
  
  LLVMContextImpl *pImpl = C->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(V1->getType(), Key);
}

Constant *ConstantExpr::getGetElementPtr(Constant *C, Value* const *Idxs,
                                         unsigned NumIdx, bool InBounds) {
  if (Constant *FC = ConstantFoldGetElementPtr(C, InBounds,
                                               makeArrayRef(Idxs, NumIdx)))
    return FC;          // Fold a few common cases.

  // Get the result type of the getelementptr!
  Type *Ty = 
    GetElementPtrInst::getIndexedType(C->getType(), Idxs, Idxs+NumIdx);
  assert(Ty && "GEP indices invalid!");
  unsigned AS = cast<PointerType>(C->getType())->getAddressSpace();
  Type *ReqTy = Ty->getPointerTo(AS);
  
  assert(C->getType()->isPointerTy() &&
         "Non-pointer type for constant GetElementPtr expression");
  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.reserve(NumIdx+1);
  ArgVec.push_back(C);
  for (unsigned i = 0; i != NumIdx; ++i)
    ArgVec.push_back(cast<Constant>(Idxs[i]));
  const ExprMapKeyType Key(Instruction::GetElementPtr, ArgVec, 0,
                           InBounds ? GEPOperator::IsInBounds : 0);
  
  LLVMContextImpl *pImpl = C->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *
ConstantExpr::getICmp(unsigned short pred, Constant *LHS, Constant *RHS) {
  assert(LHS->getType() == RHS->getType());
  assert(pred >= ICmpInst::FIRST_ICMP_PREDICATE && 
         pred <= ICmpInst::LAST_ICMP_PREDICATE && "Invalid ICmp Predicate");

  if (Constant *FC = ConstantFoldCompareInstruction(pred, LHS, RHS))
    return FC;          // Fold a few common cases...

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.push_back(LHS);
  ArgVec.push_back(RHS);
  // Get the key type with both the opcode and predicate
  const ExprMapKeyType Key(Instruction::ICmp, ArgVec, pred);

  Type *ResultTy = Type::getInt1Ty(LHS->getContext());
  if (VectorType *VT = dyn_cast<VectorType>(LHS->getType()))
    ResultTy = VectorType::get(ResultTy, VT->getNumElements());

  LLVMContextImpl *pImpl = LHS->getType()->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(ResultTy, Key);
}

Constant *
ConstantExpr::getFCmp(unsigned short pred, Constant *LHS, Constant *RHS) {
  assert(LHS->getType() == RHS->getType());
  assert(pred <= FCmpInst::LAST_FCMP_PREDICATE && "Invalid FCmp Predicate");

  if (Constant *FC = ConstantFoldCompareInstruction(pred, LHS, RHS))
    return FC;          // Fold a few common cases...

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.push_back(LHS);
  ArgVec.push_back(RHS);
  // Get the key type with both the opcode and predicate
  const ExprMapKeyType Key(Instruction::FCmp, ArgVec, pred);

  Type *ResultTy = Type::getInt1Ty(LHS->getContext());
  if (VectorType *VT = dyn_cast<VectorType>(LHS->getType()))
    ResultTy = VectorType::get(ResultTy, VT->getNumElements());

  LLVMContextImpl *pImpl = LHS->getType()->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(ResultTy, Key);
}

Constant *ConstantExpr::getExtractElement(Constant *Val, Constant *Idx) {
  assert(Val->getType()->isVectorTy() &&
         "Tried to create extractelement operation on non-vector type!");
  assert(Idx->getType()->isIntegerTy(32) &&
         "Extractelement index must be i32 type!");
  
  if (Constant *FC = ConstantFoldExtractElementInstruction(Val, Idx))
    return FC;          // Fold a few common cases.
  
  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec(1, Val);
  ArgVec.push_back(Idx);
  const ExprMapKeyType Key(Instruction::ExtractElement,ArgVec);
  
  LLVMContextImpl *pImpl = Val->getContext().pImpl;
  Type *ReqTy = cast<VectorType>(Val->getType())->getElementType();
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getInsertElement(Constant *Val, Constant *Elt, 
                                         Constant *Idx) {
  assert(Val->getType()->isVectorTy() &&
         "Tried to create insertelement operation on non-vector type!");
  assert(Elt->getType() == cast<VectorType>(Val->getType())->getElementType()
         && "Insertelement types must match!");
  assert(Idx->getType()->isIntegerTy(32) &&
         "Insertelement index must be i32 type!");

  if (Constant *FC = ConstantFoldInsertElementInstruction(Val, Elt, Idx))
    return FC;          // Fold a few common cases.
  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec(1, Val);
  ArgVec.push_back(Elt);
  ArgVec.push_back(Idx);
  const ExprMapKeyType Key(Instruction::InsertElement,ArgVec);
  
  LLVMContextImpl *pImpl = Val->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(Val->getType(), Key);
}

Constant *ConstantExpr::getShuffleVector(Constant *V1, Constant *V2, 
                                         Constant *Mask) {
  assert(ShuffleVectorInst::isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector constant expr operands!");

  if (Constant *FC = ConstantFoldShuffleVectorInstruction(V1, V2, Mask))
    return FC;          // Fold a few common cases.

  unsigned NElts = cast<VectorType>(Mask->getType())->getNumElements();
  Type *EltTy = cast<VectorType>(V1->getType())->getElementType();
  Type *ShufTy = VectorType::get(EltTy, NElts);

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec(1, V1);
  ArgVec.push_back(V2);
  ArgVec.push_back(Mask);
  const ExprMapKeyType Key(Instruction::ShuffleVector,ArgVec);
  
  LLVMContextImpl *pImpl = ShufTy->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(ShufTy, Key);
}

Constant *ConstantExpr::getInsertValue(Constant *Agg, Constant *Val,
                                       ArrayRef<unsigned> Idxs) {
  assert(ExtractValueInst::getIndexedType(Agg->getType(),
                                          Idxs) == Val->getType() &&
         "insertvalue indices invalid!");
  assert(Agg->getType()->isFirstClassType() &&
         "Non-first-class type for constant insertvalue expression");
  Constant *FC = ConstantFoldInsertValueInstruction(Agg, Val, Idxs);
  assert(FC && "insertvalue constant expr couldn't be folded!");
  return FC;
}

Constant *ConstantExpr::getExtractValue(Constant *Agg,
                                        ArrayRef<unsigned> Idxs) {
  assert(Agg->getType()->isFirstClassType() &&
         "Tried to create extractelement operation on non-first-class type!");

  Type *ReqTy = ExtractValueInst::getIndexedType(Agg->getType(), Idxs);
  (void)ReqTy;
  assert(ReqTy && "extractvalue indices invalid!");
  
  assert(Agg->getType()->isFirstClassType() &&
         "Non-first-class type for constant extractvalue expression");
  Constant *FC = ConstantFoldExtractValueInstruction(Agg, Idxs);
  assert(FC && "ExtractValue constant expr couldn't be folded!");
  return FC;
}

Constant *ConstantExpr::getNeg(Constant *C, bool HasNUW, bool HasNSW) {
  assert(C->getType()->isIntOrIntVectorTy() &&
         "Cannot NEG a nonintegral value!");
  return getSub(ConstantFP::getZeroValueForNegation(C->getType()),
                C, HasNUW, HasNSW);
}

Constant *ConstantExpr::getFNeg(Constant *C) {
  assert(C->getType()->isFPOrFPVectorTy() &&
         "Cannot FNEG a non-floating-point value!");
  return getFSub(ConstantFP::getZeroValueForNegation(C->getType()), C);
}

Constant *ConstantExpr::getNot(Constant *C) {
  assert(C->getType()->isIntOrIntVectorTy() &&
         "Cannot NOT a nonintegral value!");
  return get(Instruction::Xor, C, Constant::getAllOnesValue(C->getType()));
}

Constant *ConstantExpr::getAdd(Constant *C1, Constant *C2,
                               bool HasNUW, bool HasNSW) {
  unsigned Flags = (HasNUW ? OverflowingBinaryOperator::NoUnsignedWrap : 0) |
                   (HasNSW ? OverflowingBinaryOperator::NoSignedWrap   : 0);
  return get(Instruction::Add, C1, C2, Flags);
}

Constant *ConstantExpr::getFAdd(Constant *C1, Constant *C2) {
  return get(Instruction::FAdd, C1, C2);
}

Constant *ConstantExpr::getSub(Constant *C1, Constant *C2,
                               bool HasNUW, bool HasNSW) {
  unsigned Flags = (HasNUW ? OverflowingBinaryOperator::NoUnsignedWrap : 0) |
                   (HasNSW ? OverflowingBinaryOperator::NoSignedWrap   : 0);
  return get(Instruction::Sub, C1, C2, Flags);
}

Constant *ConstantExpr::getFSub(Constant *C1, Constant *C2) {
  return get(Instruction::FSub, C1, C2);
}

Constant *ConstantExpr::getMul(Constant *C1, Constant *C2,
                               bool HasNUW, bool HasNSW) {
  unsigned Flags = (HasNUW ? OverflowingBinaryOperator::NoUnsignedWrap : 0) |
                   (HasNSW ? OverflowingBinaryOperator::NoSignedWrap   : 0);
  return get(Instruction::Mul, C1, C2, Flags);
}

Constant *ConstantExpr::getFMul(Constant *C1, Constant *C2) {
  return get(Instruction::FMul, C1, C2);
}

Constant *ConstantExpr::getUDiv(Constant *C1, Constant *C2, bool isExact) {
  return get(Instruction::UDiv, C1, C2,
             isExact ? PossiblyExactOperator::IsExact : 0);
}

Constant *ConstantExpr::getSDiv(Constant *C1, Constant *C2, bool isExact) {
  return get(Instruction::SDiv, C1, C2,
             isExact ? PossiblyExactOperator::IsExact : 0);
}

Constant *ConstantExpr::getFDiv(Constant *C1, Constant *C2) {
  return get(Instruction::FDiv, C1, C2);
}

Constant *ConstantExpr::getURem(Constant *C1, Constant *C2) {
  return get(Instruction::URem, C1, C2);
}

Constant *ConstantExpr::getSRem(Constant *C1, Constant *C2) {
  return get(Instruction::SRem, C1, C2);
}

Constant *ConstantExpr::getFRem(Constant *C1, Constant *C2) {
  return get(Instruction::FRem, C1, C2);
}

Constant *ConstantExpr::getAnd(Constant *C1, Constant *C2) {
  return get(Instruction::And, C1, C2);
}

Constant *ConstantExpr::getOr(Constant *C1, Constant *C2) {
  return get(Instruction::Or, C1, C2);
}

Constant *ConstantExpr::getXor(Constant *C1, Constant *C2) {
  return get(Instruction::Xor, C1, C2);
}

Constant *ConstantExpr::getShl(Constant *C1, Constant *C2,
                               bool HasNUW, bool HasNSW) {
  unsigned Flags = (HasNUW ? OverflowingBinaryOperator::NoUnsignedWrap : 0) |
                   (HasNSW ? OverflowingBinaryOperator::NoSignedWrap   : 0);
  return get(Instruction::Shl, C1, C2, Flags);
}

Constant *ConstantExpr::getLShr(Constant *C1, Constant *C2, bool isExact) {
  return get(Instruction::LShr, C1, C2,
             isExact ? PossiblyExactOperator::IsExact : 0);
}

Constant *ConstantExpr::getAShr(Constant *C1, Constant *C2, bool isExact) {
  return get(Instruction::AShr, C1, C2,
             isExact ? PossiblyExactOperator::IsExact : 0);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantExpr::destroyConstant() {
  getType()->getContext().pImpl->ExprConstants.remove(this);
  destroyConstantImpl();
}

const char *ConstantExpr::getOpcodeName() const {
  return Instruction::getOpcodeName(getOpcode());
}



GetElementPtrConstantExpr::
GetElementPtrConstantExpr(Constant *C, const std::vector<Constant*> &IdxList,
                          Type *DestTy)
  : ConstantExpr(DestTy, Instruction::GetElementPtr,
                 OperandTraits<GetElementPtrConstantExpr>::op_end(this)
                 - (IdxList.size()+1), IdxList.size()+1) {
  OperandList[0] = C;
  for (unsigned i = 0, E = IdxList.size(); i != E; ++i)
    OperandList[i+1] = IdxList[i];
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
///
void ConstantArray::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  LLVMContextImpl *pImpl = getType()->getContext().pImpl;

  std::pair<LLVMContextImpl::ArrayConstantsTy::MapKey, ConstantArray*> Lookup;
  Lookup.first.first = cast<ArrayType>(getType());
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
  replaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}

void ConstantStruct::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                 Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  unsigned OperandToUpdate = U-OperandList;
  assert(getOperand(OperandToUpdate) == From && "ReplaceAllUsesWith broken!");

  std::pair<LLVMContextImpl::StructConstantsTy::MapKey, ConstantStruct*> Lookup;
  Lookup.first.first = cast<StructType>(getType());
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
  
  LLVMContextImpl *pImpl = getContext().pImpl;
  
  Constant *Replacement = 0;
  if (isAllZeros) {
    Replacement = ConstantAggregateZero::get(getType());
  } else {
    // Check to see if we have this struct type already.
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
  replaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
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
  
  Constant *Replacement = get(Values);
  assert(Replacement != this && "I didn't contain From!");
  
  // Everyone using this now uses the replacement.
  replaceAllUsesWith(Replacement);
  
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
                                                 &Indices[0], Indices.size(),
                                         cast<GEPOperator>(this)->isInBounds());
  } else if (getOpcode() == Instruction::ExtractValue) {
    Constant *Agg = getOperand(0);
    if (Agg == From) Agg = To;
    
    ArrayRef<unsigned> Indices = getIndices();
    Replacement = ConstantExpr::getExtractValue(Agg, Indices);
  } else if (getOpcode() == Instruction::InsertValue) {
    Constant *Agg = getOperand(0);
    Constant *Val = getOperand(1);
    if (Agg == From) Agg = To;
    if (Val == From) Val = To;
    
    ArrayRef<unsigned> Indices = getIndices();
    Replacement = ConstantExpr::getInsertValue(Agg, Val, Indices);
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
    Replacement = ConstantExpr::get(getOpcode(), C1, C2, SubclassOptionalData);
  } else {
    llvm_unreachable("Unknown ConstantExpr type!");
    return;
  }
  
  assert(Replacement != this && "I didn't contain From!");
  
  // Everyone using this now uses the replacement.
  replaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}
