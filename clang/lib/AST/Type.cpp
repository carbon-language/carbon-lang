//===--- Type.cpp - Type representation and manipulation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements type-related functionality.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/StringExtras.h"
#include <sstream>
using namespace clang;

void Type::Destroy(ASTContext& C) { delete this; }

void FunctionTypeProto::Destroy(ASTContext& C) {
  // Destroy the object, but don't call delete.  These are malloc'd.
  this->~FunctionTypeProto();
  free(this);  
}

void VariableArrayType::Destroy(ASTContext& C) {
  SizeExpr->Destroy(C);
  delete this;  
}

/// isVoidType - Helper method to determine if this is the 'void' type.
bool Type::isVoidType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Void;
  return false;
}

bool Type::isObjectType() const {
  if (isa<FunctionType>(CanonicalType))
    return false;
  else if (CanonicalType->isIncompleteType())
    return false;
  else
    return true;
}

bool Type::isDerivedType() const {
  switch (CanonicalType->getTypeClass()) {
  case Pointer:
  case VariableArray:
  case ConstantArray:
  case IncompleteArray:
  case FunctionProto:
  case FunctionNoProto:
  case Reference:
    return true;
  case Tagged: {
    const TagType *TT = cast<TagType>(CanonicalType);
    return !TT->getDecl()->isEnum();
  }
  default:
    return false;
  }
}

bool Type::isClassType() const {
  if (const RecordType *RT = dyn_cast<RecordType>(CanonicalType))
    if (RT->getDecl()->isClass())
      return true;
  return false;
}
bool Type::isStructureType() const {
  if (const RecordType *RT = dyn_cast<RecordType>(CanonicalType))
    if (RT->getDecl()->isStruct())
      return true;
  return false;
}
bool Type::isUnionType() const {
  if (const RecordType *RT = dyn_cast<RecordType>(CanonicalType))
    if (RT->getDecl()->isUnion())
      return true;
  return false;
}

bool Type::isComplexType() const {
  if (const ComplexType *CT = dyn_cast<ComplexType>(CanonicalType))
    return CT->getElementType()->isFloatingType();
  return false;
}

bool Type::isComplexIntegerType() const {
  // Check for GCC complex integer extension.
  if (const ComplexType *CT = dyn_cast<ComplexType>(CanonicalType))
    return CT->getElementType()->isIntegerType();
  return false;
}

const ComplexType *Type::getAsComplexIntegerType() const {
  // Are we directly a complex type?
  if (const ComplexType *CTy = dyn_cast<ComplexType>(this)) {
    if (CTy->getElementType()->isIntegerType())
      return CTy;
  }
  // If the canonical form of this type isn't the right kind, reject it.
  const ComplexType *CTy = dyn_cast<ComplexType>(CanonicalType);
  if (!CTy || !CTy->getElementType()->isIntegerType())
    return 0;

  // If this is a typedef for a complex type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsComplexIntegerType();
}

/// getDesugaredType - Return the specified type with any "sugar" removed from
/// type type.  This takes off typedefs, typeof's etc.  If the outer level of
/// the type is already concrete, it returns it unmodified.  This is similar
/// to getting the canonical type, but it doesn't remove *all* typedefs.  For
/// example, it return "T*" as "T*", (not as "int*"), because the pointer is
/// concrete.
const Type *Type::getDesugaredType() const {
  if (const TypedefType *TDT = dyn_cast<TypedefType>(this))
    return TDT->LookThroughTypedefs().getTypePtr();
  if (const TypeOfExpr *TOE = dyn_cast<TypeOfExpr>(this))
    return TOE->getUnderlyingExpr()->getType().getTypePtr();
  if (const TypeOfType *TOT = dyn_cast<TypeOfType>(this))
    return TOT->getUnderlyingType().getTypePtr();
  return this;
}


const BuiltinType *Type::getAsBuiltinType() const {
  // If this is directly a builtin type, return it.
  if (const BuiltinType *BTy = dyn_cast<BuiltinType>(this))
    return BTy;

  // If the canonical form of this type isn't a builtin type, reject it.
  if (!isa<BuiltinType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<BuiltinType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsBuiltinType();
    return 0;
  }

  // If this is a typedef for a builtin type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsBuiltinType();
}

const FunctionType *Type::getAsFunctionType() const {
  // If this is directly a function type, return it.
  if (const FunctionType *FTy = dyn_cast<FunctionType>(this))
    return FTy;

  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<FunctionType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<FunctionType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsFunctionType();
    return 0;
  }
  
  // If this is a typedef for a function type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsFunctionType();
}

const PointerLikeType *Type::getAsPointerLikeType() const {
  // If this is directly a pointer-like type, return it.
  if (const PointerLikeType *PTy = dyn_cast<PointerLikeType>(this))
    return PTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<PointerLikeType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<PointerLikeType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsPointerLikeType();
    return 0;
  }
  
  // If this is a typedef for a pointer type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsPointerLikeType();
}

const PointerType *Type::getAsPointerType() const {
  // If this is directly a pointer type, return it.
  if (const PointerType *PTy = dyn_cast<PointerType>(this))
    return PTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<PointerType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<PointerType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsPointerType();
    return 0;
  }

  // If this is a typedef for a pointer type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsPointerType();
}

const ReferenceType *Type::getAsReferenceType() const {
  // If this is directly a reference type, return it.
  if (const ReferenceType *RTy = dyn_cast<ReferenceType>(this))
    return RTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ReferenceType>(CanonicalType)) {    
    // Look through type qualifiers
    if (isa<ReferenceType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsReferenceType();
    return 0;
  }

  // If this is a typedef for a reference type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsReferenceType();
}

const ArrayType *Type::getAsArrayType() const {
  // If this is directly an array type, return it.
  if (const ArrayType *ATy = dyn_cast<ArrayType>(this))
    return ATy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ArrayType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<ArrayType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsArrayType();
    return 0;
  }
  
  // If this is a typedef for an array type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsArrayType();
}

const ConstantArrayType *Type::getAsConstantArrayType() const {
  // If this is directly a constant array type, return it.
  if (const ConstantArrayType *ATy = dyn_cast<ConstantArrayType>(this))
    return ATy;

  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ConstantArrayType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<ConstantArrayType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsConstantArrayType();
    return 0;
  }
  
  // If this is a typedef for a constant array type, strip the typedef off
  // without losing all typedef information.
  return getDesugaredType()->getAsConstantArrayType();
}

const VariableArrayType *Type::getAsVariableArrayType() const {
  // If this is directly a variable array type, return it.
  if (const VariableArrayType *ATy = dyn_cast<VariableArrayType>(this))
    return ATy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<VariableArrayType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<VariableArrayType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsVariableArrayType();
    return 0;
  }

  // If this is a typedef for a variable array type, strip the typedef off
  // without losing all typedef information.
  return getDesugaredType()->getAsVariableArrayType();
}

/// isVariablyModifiedType (C99 6.7.5p3) - Return true for variable length
/// array types and types that contain variable array types in their
/// declarator
bool Type::isVariablyModifiedType() const {
  // A VLA is a veriably modified type
  if (getAsVariableArrayType())
    return true;

  // An array can contain a variably modified type
  if (const ArrayType* AT = getAsArrayType())
    return AT->getElementType()->isVariablyModifiedType();

  // A pointer can point to a variably modified type
  if (const PointerType* PT = getAsPointerType())
    return PT->getPointeeType()->isVariablyModifiedType();

  // A function can return a variably modified type
  // This one isn't completely obvious, but it follows from the
  // definition in C99 6.7.5p3. Because of this rule, it's
  // illegal to declare a function returning a variably modified type.
  if (const FunctionType* FT = getAsFunctionType())
    return FT->getResultType()->isVariablyModifiedType();

  return false;
}

bool Type::isIncompleteArrayType() const {
  return isa<IncompleteArrayType>(CanonicalType);
}

const IncompleteArrayType *Type::getAsIncompleteArrayType() const {
  // If this is directly a variable array type, return it.
  if (const IncompleteArrayType *ATy = dyn_cast<IncompleteArrayType>(this))
    return ATy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<IncompleteArrayType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<IncompleteArrayType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsIncompleteArrayType();
    return 0;
  }

  // If this is a typedef for a variable array type, strip the typedef off
  // without losing all typedef information.
  return getDesugaredType()->getAsIncompleteArrayType();
}

const RecordType *Type::getAsRecordType() const {
  // If this is directly a reference type, return it.
  if (const RecordType *RTy = dyn_cast<RecordType>(this))
    return RTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<RecordType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<RecordType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsRecordType();
    return 0;
  }

  // If this is a typedef for a record type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsRecordType();
}

const RecordType *Type::getAsStructureType() const {
  // If this is directly a structure type, return it.
  if (const RecordType *RT = dyn_cast<RecordType>(this)) {
    if (RT->getDecl()->isStruct())
      return RT;
  }

  // If the canonical form of this type isn't the right kind, reject it.
  if (const RecordType *RT = dyn_cast<RecordType>(CanonicalType)) {
    if (!RT->getDecl()->isStruct())
      return 0;
    
    // If this is a typedef for a structure type, strip the typedef off without
    // losing all typedef information.
    return getDesugaredType()->getAsStructureType();
  }
  // Look through type qualifiers
  if (isa<RecordType>(CanonicalType.getUnqualifiedType()))
    return CanonicalType.getUnqualifiedType()->getAsStructureType();
  return 0;
}

const RecordType *Type::getAsUnionType() const { 
  // If this is directly a union type, return it.
  if (const RecordType *RT = dyn_cast<RecordType>(this)) {
    if (RT->getDecl()->isUnion())
      return RT;
  }
    
  // If the canonical form of this type isn't the right kind, reject it.
  if (const RecordType *RT = dyn_cast<RecordType>(CanonicalType)) {
    if (!RT->getDecl()->isUnion())
      return 0;

    // If this is a typedef for a union type, strip the typedef off without
    // losing all typedef information.
    return getDesugaredType()->getAsUnionType();
  }
  
  // Look through type qualifiers
  if (isa<RecordType>(CanonicalType.getUnqualifiedType()))
    return CanonicalType.getUnqualifiedType()->getAsUnionType();
  return 0;
}

const EnumType *Type::getAsEnumType() const {
  // Check the canonicalized unqualified type directly; the more complex
  // version is unnecessary because there isn't any typedef information
  // to preserve.
  return dyn_cast<EnumType>(CanonicalType.getUnqualifiedType());
}

const ComplexType *Type::getAsComplexType() const {
  // Are we directly a complex type?
  if (const ComplexType *CTy = dyn_cast<ComplexType>(this))
    return CTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ComplexType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<ComplexType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsComplexType();
    return 0;
  }

  // If this is a typedef for a complex type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsComplexType();
}

const VectorType *Type::getAsVectorType() const {
  // Are we directly a vector type?
  if (const VectorType *VTy = dyn_cast<VectorType>(this))
    return VTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<VectorType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<VectorType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsVectorType();
    return 0;
  }

  // If this is a typedef for a vector type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsVectorType();
}

const ExtVectorType *Type::getAsExtVectorType() const {
  // Are we directly an OpenCU vector type?
  if (const ExtVectorType *VTy = dyn_cast<ExtVectorType>(this))
    return VTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ExtVectorType>(CanonicalType)) {  
    // Look through type qualifiers
    if (isa<ExtVectorType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsExtVectorType();
    return 0;
  }

  // If this is a typedef for an extended vector type, strip the typedef off
  // without losing all typedef information.
  return getDesugaredType()->getAsExtVectorType();
}

const ObjCInterfaceType *Type::getAsObjCInterfaceType() const {
  // There is no sugar for ObjCInterfaceType's, just return the canonical
  // type pointer if it is the right class.
  return dyn_cast<ObjCInterfaceType>(CanonicalType);
}

const ObjCQualifiedInterfaceType *
Type::getAsObjCQualifiedInterfaceType() const {
  // There is no sugar for ObjCQualifiedInterfaceType's, just return the canonical
  // type pointer if it is the right class.
  return dyn_cast<ObjCQualifiedInterfaceType>(CanonicalType);
}

const ObjCQualifiedIdType *Type::getAsObjCQualifiedIdType() const {
  // There is no sugar for ObjCQualifiedIdType's, just return the canonical
  // type pointer if it is the right class.
  return dyn_cast<ObjCQualifiedIdType>(CanonicalType);
}


bool Type::isIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::LongLong;
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    if (TT->getDecl()->isEnum())
      return true;
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isIntegerType();
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isIntegerType();
  return false;
}

bool Type::isIntegralType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
    BT->getKind() <= BuiltinType::LongLong;
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    if (TT->getDecl()->isEnum())
      return true;
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isIntegralType();
  return false;
}

bool Type::isEnumeralType() const {
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    return TT->getDecl()->isEnum();
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isEnumeralType();
  return false;
}

bool Type::isBooleanType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Bool;
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isBooleanType();
  return false;
}

bool Type::isCharType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Char_U ||
           BT->getKind() == BuiltinType::UChar ||
           BT->getKind() == BuiltinType::Char_S ||
           BT->getKind() == BuiltinType::SChar;
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isCharType();
  return false;
}

/// isSignedIntegerType - Return true if this is an integer type that is
/// signed, according to C99 6.2.5p4 [char, signed char, short, int, long..],
/// an enum decl which has a signed representation, or a vector of signed
/// integer element type.
bool Type::isSignedIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType)) {
    return BT->getKind() >= BuiltinType::Char_S &&
           BT->getKind() <= BuiltinType::LongLong;
  }
  
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
    return ET->getDecl()->getIntegerType()->isSignedIntegerType();
  
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isSignedIntegerType();
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isSignedIntegerType();
  return false;
}

/// isUnsignedIntegerType - Return true if this is an integer type that is
/// unsigned, according to C99 6.2.5p6 [which returns true for _Bool], an enum
/// decl which has an unsigned representation, or a vector of unsigned integer
/// element type.
bool Type::isUnsignedIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType)) {
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::ULongLong;
  }

  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
    return ET->getDecl()->getIntegerType()->isUnsignedIntegerType();

  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isUnsignedIntegerType();
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isUnsignedIntegerType();
  return false;
}

bool Type::isFloatingType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Float &&
           BT->getKind() <= BuiltinType::LongDouble;
  if (const ComplexType *CT = dyn_cast<ComplexType>(CanonicalType))
    return CT->getElementType()->isFloatingType();
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isFloatingType();
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isFloatingType();
  return false;
}

bool Type::isRealFloatingType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Float &&
           BT->getKind() <= BuiltinType::LongDouble;
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isRealFloatingType();
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isRealFloatingType();
  return false;
}

bool Type::isRealType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::LongDouble;
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    return TT->getDecl()->isEnum();
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isRealType();
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isRealType();
  return false;
}

bool Type::isArithmeticType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() != BuiltinType::Void;
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
    // GCC allows forward declaration of enum types (forbid by C99 6.7.2.3p2).
    // If a body isn't seen by the time we get here, return false.
    return ET->getDecl()->isDefinition();
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isArithmeticType();
  return isa<ComplexType>(CanonicalType) || isa<VectorType>(CanonicalType);
}

bool Type::isScalarType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() != BuiltinType::Void;
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType)) {
    if (TT->getDecl()->isEnum())
      return true;
    return false;
  }
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isScalarType();
  return isa<PointerType>(CanonicalType) || isa<ComplexType>(CanonicalType) ||
         isa<ObjCQualifiedIdType>(CanonicalType);
}

bool Type::isAggregateType() const {
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType)) {
    if (TT->getDecl()->isStruct())
      return true;
    return false;
  }
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isAggregateType();
  return isa<ArrayType>(CanonicalType);
}

/// isConstantSizeType - Return true if this is not a variable sized type,
/// according to the rules of C99 6.7.5p3.  It is not legal to call this on
/// incomplete types.
bool Type::isConstantSizeType() const {
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isConstantSizeType();
  assert(!isIncompleteType() && "This doesn't make sense for incomplete types");
  // The VAT must have a size, as it is known to be complete.
  return !isa<VariableArrayType>(CanonicalType);
}

/// isIncompleteType - Return true if this is an incomplete type (C99 6.2.5p1)
/// - a type that can describe objects, but which lacks information needed to
/// determine its size.
bool Type::isIncompleteType() const { 
  switch (CanonicalType->getTypeClass()) { 
  default: return false;
  case ASQual:
    return cast<ASQualType>(CanonicalType)->getBaseType()->isIncompleteType();
  case Builtin:
    // Void is the only incomplete builtin type.  Per C99 6.2.5p19, it can never
    // be completed.
    return isVoidType();
  case Tagged:
    // A tagged type (struct/union/enum/class) is incomplete if the decl is a
    // forward declaration, but not a full definition (C99 6.2.5p22).
    return !cast<TagType>(CanonicalType)->getDecl()->isDefinition();
  case IncompleteArray:
    // An array of unknown size is an incomplete type (C99 6.2.5p22).
    return true;
  }
}

bool Type::isPromotableIntegerType() const {
  if (const ASQualType *ASQT = dyn_cast<ASQualType>(CanonicalType))
    return ASQT->getBaseType()->isPromotableIntegerType();
  const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType);
  if (!BT) return false;
  switch (BT->getKind()) {
  case BuiltinType::Bool:
  case BuiltinType::Char_S:
  case BuiltinType::Char_U:
  case BuiltinType::SChar:
  case BuiltinType::UChar:
  case BuiltinType::Short:
  case BuiltinType::UShort:
    return true;
  default: 
    return false;
  }
}

const char *BuiltinType::getName() const {
  switch (getKind()) {
  default: assert(0 && "Unknown builtin type!");
  case Void:              return "void";
  case Bool:              return "_Bool";
  case Char_S:            return "char";
  case Char_U:            return "char";
  case SChar:             return "signed char";
  case Short:             return "short";
  case Int:               return "int";
  case Long:              return "long";
  case LongLong:          return "long long";
  case UChar:             return "unsigned char";
  case UShort:            return "unsigned short";
  case UInt:              return "unsigned int";
  case ULong:             return "unsigned long";
  case ULongLong:         return "unsigned long long";
  case Float:             return "float";
  case Double:            return "double";
  case LongDouble:        return "long double";
  }
}

void FunctionTypeProto::Profile(llvm::FoldingSetNodeID &ID, QualType Result,
                                arg_type_iterator ArgTys,
                                unsigned NumArgs, bool isVariadic) {
  ID.AddPointer(Result.getAsOpaquePtr());
  for (unsigned i = 0; i != NumArgs; ++i)
    ID.AddPointer(ArgTys[i].getAsOpaquePtr());
  ID.AddInteger(isVariadic);
}

void FunctionTypeProto::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getResultType(), arg_type_begin(), NumArgs, isVariadic());
}

void ObjCQualifiedInterfaceType::Profile(llvm::FoldingSetNodeID &ID,
                                         const ObjCInterfaceDecl *Decl,
                                         ObjCProtocolDecl **protocols, 
                                         unsigned NumProtocols) {
  ID.AddPointer(Decl);
  for (unsigned i = 0; i != NumProtocols; i++)
    ID.AddPointer(protocols[i]);
}

void ObjCQualifiedInterfaceType::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getDecl(), &Protocols[0], getNumProtocols());
}

void ObjCQualifiedIdType::Profile(llvm::FoldingSetNodeID &ID,
                                         ObjCProtocolDecl **protocols, 
                                         unsigned NumProtocols) {
  for (unsigned i = 0; i != NumProtocols; i++)
    ID.AddPointer(protocols[i]);
}

void ObjCQualifiedIdType::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, &Protocols[0], getNumProtocols());
}

/// LookThroughTypedefs - Return the ultimate type this typedef corresponds to
/// potentially looking through *all* consequtive typedefs.  This returns the
/// sum of the type qualifiers, so if you have:
///   typedef const int A;
///   typedef volatile A B;
/// looking through the typedefs for B will give you "const volatile A".
///
QualType TypedefType::LookThroughTypedefs() const {
  // Usually, there is only a single level of typedefs, be fast in that case.
  QualType FirstType = getDecl()->getUnderlyingType();
  if (!isa<TypedefType>(FirstType))
    return FirstType;
  
  // Otherwise, do the fully general loop.
  unsigned TypeQuals = 0;
  const TypedefType *TDT = this;
  while (1) {
    QualType CurType = TDT->getDecl()->getUnderlyingType();
    
    
    /// FIXME:
    /// FIXME: This is incorrect for ASQuals!
    /// FIXME:
    TypeQuals |= CurType.getCVRQualifiers();

    TDT = dyn_cast<TypedefType>(CurType);
    if (TDT == 0)
      return QualType(CurType.getTypePtr(), TypeQuals);
  }
}

bool RecordType::classof(const TagType *TT) {
  return isa<RecordDecl>(TT->getDecl());
}

bool EnumType::classof(const TagType *TT) {
  return isa<EnumDecl>(TT->getDecl());
}


//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

void QualType::dump(const char *msg) const {
  std::string R = "identifier";
  getAsStringInternal(R);
  if (msg)
    fprintf(stderr, "%s: %s\n", msg, R.c_str());
  else
    fprintf(stderr, "%s\n", R.c_str());
}

static void AppendTypeQualList(std::string &S, unsigned TypeQuals) {
  // Note: funkiness to ensure we get a space only between quals.
  bool NonePrinted = true;
  if (TypeQuals & QualType::Const)
    S += "const", NonePrinted = false;
  if (TypeQuals & QualType::Volatile)
    S += (NonePrinted+" volatile"), NonePrinted = false;
  if (TypeQuals & QualType::Restrict)
    S += (NonePrinted+" restrict"), NonePrinted = false;
}

void QualType::getAsStringInternal(std::string &S) const {
  if (isNull()) {
    S += "NULL TYPE\n";
    return;
  }
  
  // Print qualifiers as appropriate.
  if (unsigned Tq = getCVRQualifiers()) {
    std::string TQS;
    AppendTypeQualList(TQS, Tq);
    if (!S.empty())
      S = TQS + ' ' + S;
    else
      S = TQS;
  }

  getTypePtr()->getAsStringInternal(S);
}

void BuiltinType::getAsStringInternal(std::string &S) const {
  if (S.empty()) {
    S = getName();
  } else {
    // Prefix the basic type, e.g. 'int X'.
    S = ' ' + S;
    S = getName() + S;
  }
}

void ComplexType::getAsStringInternal(std::string &S) const {
  ElementType->getAsStringInternal(S);
  S = "_Complex " + S;
}

void ASQualType::getAsStringInternal(std::string &S) const {
  S = "__attribute__((address_space("+llvm::utostr_32(AddressSpace)+")))" + S;
  BaseType->getAsStringInternal(S);
}

void PointerType::getAsStringInternal(std::string &S) const {
  S = '*' + S;
  
  // Handle things like 'int (*A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(getPointeeType()))
    S = '(' + S + ')';
  
  getPointeeType().getAsStringInternal(S);
}

void ReferenceType::getAsStringInternal(std::string &S) const {
  S = '&' + S;
  
  // Handle things like 'int (&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(getPointeeType()))
    S = '(' + S + ')';
  
  getPointeeType().getAsStringInternal(S);
}

void ConstantArrayType::getAsStringInternal(std::string &S) const {
  S += '[';
  S += llvm::utostr(getSize().getZExtValue());
  S += ']';
  
  getElementType().getAsStringInternal(S);
}

void IncompleteArrayType::getAsStringInternal(std::string &S) const {
  S += "[]";

  getElementType().getAsStringInternal(S);
}

void VariableArrayType::getAsStringInternal(std::string &S) const {
  S += '[';
  
  if (getIndexTypeQualifier()) {
    AppendTypeQualList(S, getIndexTypeQualifier());
    S += ' ';
  }
  
  if (getSizeModifier() == Static)
    S += "static";
  else if (getSizeModifier() == Star)
    S += '*';
  
  if (getSizeExpr()) {
    std::ostringstream s;
    getSizeExpr()->printPretty(s);
    S += s.str();
  }
  S += ']';
  
  getElementType().getAsStringInternal(S);
}

void VectorType::getAsStringInternal(std::string &S) const {
  S += " __attribute__((__vector_size__(";
  // FIXME: should multiply by element size somehow.
  S += llvm::utostr_32(NumElements*4); // convert back to bytes.
  S += ")))";
  ElementType.getAsStringInternal(S);
}

void ExtVectorType::getAsStringInternal(std::string &S) const {
  S += " __attribute__((ext_vector_type(";
  S += llvm::utostr_32(NumElements);
  S += ")))";
  ElementType.getAsStringInternal(S);
}

void TypeOfExpr::getAsStringInternal(std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typeof(e) X'.
    InnerString = ' ' + InnerString;
  std::ostringstream s;
  getUnderlyingExpr()->printPretty(s);
  InnerString = "typeof(" + s.str() + ")" + InnerString;
}

void TypeOfType::getAsStringInternal(std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typeof(t) X'.
    InnerString = ' ' + InnerString;
  std::string Tmp;
  getUnderlyingType().getAsStringInternal(Tmp);
  InnerString = "typeof(" + Tmp + ")" + InnerString;
}

void FunctionTypeNoProto::getAsStringInternal(std::string &S) const {
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  if (!S.empty())
    S = "(" + S + ")";
  
  S += "()";
  getResultType().getAsStringInternal(S);
}

void FunctionTypeProto::getAsStringInternal(std::string &S) const {
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  if (!S.empty())
    S = "(" + S + ")";
  
  S += "(";
  std::string Tmp;
  for (unsigned i = 0, e = getNumArgs(); i != e; ++i) {
    if (i) S += ", ";
    getArgType(i).getAsStringInternal(Tmp);
    S += Tmp;
    Tmp.clear();
  }
  
  if (isVariadic()) {
    if (getNumArgs())
      S += ", ";
    S += "...";
  } else if (getNumArgs() == 0) {
    // Do not emit int() if we have a proto, emit 'int(void)'.
    S += "void";
  }
  
  S += ")";
  getResultType().getAsStringInternal(S);
}


void TypedefType::getAsStringInternal(std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  InnerString = getDecl()->getIdentifier()->getName() + InnerString;
}

void ObjCInterfaceType::getAsStringInternal(std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  InnerString = getDecl()->getIdentifier()->getName() + InnerString;
}

void ObjCQualifiedInterfaceType::getAsStringInternal(
                                  std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  std::string ObjCQIString = getDecl()->getName();
  ObjCQIString += '<';
  bool isFirst = true;
  for (qual_iterator I = qual_begin(), E = qual_end(); I != E; ++I) {
    if (isFirst)
      isFirst = false;
    else
      ObjCQIString += ',';
    ObjCQIString += (*I)->getName();
  }
  ObjCQIString += '>';
  InnerString = ObjCQIString + InnerString;
}

void ObjCQualifiedIdType::getAsStringInternal(
                                              std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  std::string ObjCQIString = "id";
  ObjCQIString += '<';
  int num = getNumProtocols();
  for (int i = 0; i < num; i++) {
    ObjCQIString += getProtocols(i)->getName();
    if (i < num-1)
      ObjCQIString += ',';
  }
  ObjCQIString += '>';
  InnerString = ObjCQIString + InnerString;
}

void TagType::getAsStringInternal(std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  
  const char *Kind = getDecl()->getKindName();
  const char *ID;
  if (const IdentifierInfo *II = getDecl()->getIdentifier())
    ID = II->getName();
  else
    ID = "<anonymous>";

  InnerString = std::string(Kind) + " " + ID + InnerString;
}
