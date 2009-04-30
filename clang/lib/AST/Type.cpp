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

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

bool QualType::isConstant(ASTContext &Ctx) const {
  if (isConstQualified())
    return true;

  if (getTypePtr()->isArrayType())
    return Ctx.getAsArrayType(*this)->getElementType().isConstant(Ctx);

  return false;
}

void Type::Destroy(ASTContext& C) {
  this->~Type();
  C.Deallocate(this);
}

void VariableArrayType::Destroy(ASTContext& C) {
  if (SizeExpr)
    SizeExpr->Destroy(C);
  this->~VariableArrayType();
  C.Deallocate(this);
}

void DependentSizedArrayType::Destroy(ASTContext& C) {
  SizeExpr->Destroy(C);
  this->~DependentSizedArrayType();
  C.Deallocate(this);
}

/// getArrayElementTypeNoTypeQual - If this is an array type, return the
/// element type of the array, potentially with type qualifiers missing.
/// This method should never be used when type qualifiers are meaningful.
const Type *Type::getArrayElementTypeNoTypeQual() const {
  // If this is directly an array type, return it.
  if (const ArrayType *ATy = dyn_cast<ArrayType>(this))
    return ATy->getElementType().getTypePtr();
    
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ArrayType>(CanonicalType)) {
    // Look through type qualifiers
    if (ArrayType *AT = dyn_cast<ArrayType>(CanonicalType.getUnqualifiedType()))
      return AT->getElementType().getTypePtr();
    return 0;
  }
  
  // If this is a typedef for an array type, strip the typedef off without
  // losing all typedef information.
  return cast<ArrayType>(getDesugaredType())->getElementType().getTypePtr();
}

/// getDesugaredType - Return the specified type with any "sugar" removed from
/// the type.  This takes off typedefs, typeof's etc.  If the outer level of
/// the type is already concrete, it returns it unmodified.  This is similar
/// to getting the canonical type, but it doesn't remove *all* typedefs.  For
/// example, it returns "T*" as "T*", (not as "int*"), because the pointer is
/// concrete.
///
/// \param ForDisplay When true, the desugaring is provided for
/// display purposes only. In this case, we apply more heuristics to
/// decide whether it is worth providing a desugared form of the type
/// or not.
QualType QualType::getDesugaredType(bool ForDisplay) const {
  return getTypePtr()->getDesugaredType(ForDisplay)
     .getWithAdditionalQualifiers(getCVRQualifiers());
}

/// getDesugaredType - Return the specified type with any "sugar" removed from
/// type type.  This takes off typedefs, typeof's etc.  If the outer level of
/// the type is already concrete, it returns it unmodified.  This is similar
/// to getting the canonical type, but it doesn't remove *all* typedefs.  For
/// example, it return "T*" as "T*", (not as "int*"), because the pointer is
/// concrete.
///
/// \param ForDisplay When true, the desugaring is provided for
/// display purposes only. In this case, we apply more heuristics to
/// decide whether it is worth providing a desugared form of the type
/// or not.
QualType Type::getDesugaredType(bool ForDisplay) const {
  if (const TypedefType *TDT = dyn_cast<TypedefType>(this))
    return TDT->LookThroughTypedefs().getDesugaredType();
  if (const TypeOfExprType *TOE = dyn_cast<TypeOfExprType>(this))
    return TOE->getUnderlyingExpr()->getType().getDesugaredType();
  if (const TypeOfType *TOT = dyn_cast<TypeOfType>(this))
    return TOT->getUnderlyingType().getDesugaredType();
  if (const TemplateSpecializationType *Spec 
        = dyn_cast<TemplateSpecializationType>(this)) {
    if (ForDisplay)
      return QualType(this, 0);

    QualType Canon = Spec->getCanonicalTypeInternal();
    if (Canon->getAsTemplateSpecializationType())
      return QualType(this, 0);
    return Canon->getDesugaredType();
  }
  if (const QualifiedNameType *QualName  = dyn_cast<QualifiedNameType>(this)) {
    if (ForDisplay) {
      // If desugaring the type that the qualified name is referring to
      // produces something interesting, that's our desugared type.
      QualType NamedType = QualName->getNamedType().getDesugaredType();
      if (NamedType != QualName->getNamedType())
        return NamedType;
    } else
      return QualName->getNamedType().getDesugaredType();
  }

  return QualType(this, 0);
}

/// isVoidType - Helper method to determine if this is the 'void' type.
bool Type::isVoidType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Void;
  if (const ExtQualType *AS = dyn_cast<ExtQualType>(CanonicalType))
    return AS->getBaseType()->isVoidType();
  return false;
}

bool Type::isObjectType() const {
  if (isa<FunctionType>(CanonicalType) || isa<ReferenceType>(CanonicalType) ||
      isa<IncompleteArrayType>(CanonicalType) || isVoidType())
    return false;
  if (const ExtQualType *AS = dyn_cast<ExtQualType>(CanonicalType))
    return AS->getBaseType()->isObjectType();
  return true;
}

bool Type::isDerivedType() const {
  switch (CanonicalType->getTypeClass()) {
  case ExtQual:
    return cast<ExtQualType>(CanonicalType)->getBaseType()->isDerivedType();
  case Pointer:
  case VariableArray:
  case ConstantArray:
  case IncompleteArray:
  case FunctionProto:
  case FunctionNoProto:
  case LValueReference:
  case RValueReference:
  case Record:
    return true;
  default:
    return false;
  }
}

bool Type::isClassType() const {
  if (const RecordType *RT = getAsRecordType())
    return RT->getDecl()->isClass();
  return false;
}
bool Type::isStructureType() const {
  if (const RecordType *RT = getAsRecordType())
    return RT->getDecl()->isStruct();
  return false;
}
bool Type::isUnionType() const {
  if (const RecordType *RT = getAsRecordType())
    return RT->getDecl()->isUnion();
  return false;
}

bool Type::isComplexType() const {
  if (const ComplexType *CT = dyn_cast<ComplexType>(CanonicalType))
    return CT->getElementType()->isFloatingType();
  if (const ExtQualType *AS = dyn_cast<ExtQualType>(CanonicalType))
    return AS->getBaseType()->isComplexType();
  return false;
}

bool Type::isComplexIntegerType() const {
  // Check for GCC complex integer extension.
  if (const ComplexType *CT = dyn_cast<ComplexType>(CanonicalType))
    return CT->getElementType()->isIntegerType();
  if (const ExtQualType *AS = dyn_cast<ExtQualType>(CanonicalType))
    return AS->getBaseType()->isComplexIntegerType();
  return false;
}

const ComplexType *Type::getAsComplexIntegerType() const {
  // Are we directly a complex type?
  if (const ComplexType *CTy = dyn_cast<ComplexType>(this)) {
    if (CTy->getElementType()->isIntegerType())
      return CTy;
    return 0;
  }
  
  // If the canonical form of this type isn't what we want, reject it.
  if (!isa<ComplexType>(CanonicalType)) {
    // Look through type qualifiers (e.g. ExtQualType's).
    if (isa<ComplexType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsComplexIntegerType();
    return 0;
  }
  
  // If this is a typedef for a complex type, strip the typedef off without
  // losing all typedef information.
  return cast<ComplexType>(getDesugaredType());
}

const BuiltinType *Type::getAsBuiltinType() const {
  // If this is directly a builtin type, return it.
  if (const BuiltinType *BTy = dyn_cast<BuiltinType>(this))
    return BTy;

  // If the canonical form of this type isn't a builtin type, reject it.
  if (!isa<BuiltinType>(CanonicalType)) {
    // Look through type qualifiers (e.g. ExtQualType's).
    if (isa<BuiltinType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsBuiltinType();
    return 0;
  }

  // If this is a typedef for a builtin type, strip the typedef off without
  // losing all typedef information.
  return cast<BuiltinType>(getDesugaredType());
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
  return cast<FunctionType>(getDesugaredType());
}

const FunctionNoProtoType *Type::getAsFunctionNoProtoType() const {
  return dyn_cast_or_null<FunctionNoProtoType>(getAsFunctionType());
}

const FunctionProtoType *Type::getAsFunctionProtoType() const {
  return dyn_cast_or_null<FunctionProtoType>(getAsFunctionType());
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
  return cast<PointerType>(getDesugaredType());
}

const BlockPointerType *Type::getAsBlockPointerType() const {
  // If this is directly a block pointer type, return it.
  if (const BlockPointerType *PTy = dyn_cast<BlockPointerType>(this))
    return PTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<BlockPointerType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<BlockPointerType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsBlockPointerType();
    return 0;
  }
  
  // If this is a typedef for a block pointer type, strip the typedef off 
  // without losing all typedef information.
  return cast<BlockPointerType>(getDesugaredType());
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
  return cast<ReferenceType>(getDesugaredType());
}

const LValueReferenceType *Type::getAsLValueReferenceType() const {
  // If this is directly an lvalue reference type, return it.
  if (const LValueReferenceType *RTy = dyn_cast<LValueReferenceType>(this))
    return RTy;

  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<LValueReferenceType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<LValueReferenceType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsLValueReferenceType();
    return 0;
  }

  // If this is a typedef for an lvalue reference type, strip the typedef off
  // without losing all typedef information.
  return cast<LValueReferenceType>(getDesugaredType());
}

const RValueReferenceType *Type::getAsRValueReferenceType() const {
  // If this is directly an rvalue reference type, return it.
  if (const RValueReferenceType *RTy = dyn_cast<RValueReferenceType>(this))
    return RTy;

  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<RValueReferenceType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<RValueReferenceType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsRValueReferenceType();
    return 0;
  }

  // If this is a typedef for an rvalue reference type, strip the typedef off
  // without losing all typedef information.
  return cast<RValueReferenceType>(getDesugaredType());
}

const MemberPointerType *Type::getAsMemberPointerType() const {
  // If this is directly a member pointer type, return it.
  if (const MemberPointerType *MTy = dyn_cast<MemberPointerType>(this))
    return MTy;

  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<MemberPointerType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<MemberPointerType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsMemberPointerType();
    return 0;
  }

  // If this is a typedef for a member pointer type, strip the typedef off
  // without losing all typedef information.
  return cast<MemberPointerType>(getDesugaredType());
}

/// isVariablyModifiedType (C99 6.7.5p3) - Return true for variable length
/// array types and types that contain variable array types in their
/// declarator
bool Type::isVariablyModifiedType() const {
  // A VLA is a variably modified type.
  if (isVariableArrayType())
    return true;

  // An array can contain a variably modified type
  if (const Type *T = getArrayElementTypeNoTypeQual())
    return T->isVariablyModifiedType();

  // A pointer can point to a variably modified type.
  // Also, C++ references and member pointers can point to a variably modified
  // type, where VLAs appear as an extension to C++, and should be treated
  // correctly.
  if (const PointerType *PT = getAsPointerType())
    return PT->getPointeeType()->isVariablyModifiedType();
  if (const ReferenceType *RT = getAsReferenceType())
    return RT->getPointeeType()->isVariablyModifiedType();
  if (const MemberPointerType *PT = getAsMemberPointerType())
    return PT->getPointeeType()->isVariablyModifiedType();

  // A function can return a variably modified type
  // This one isn't completely obvious, but it follows from the
  // definition in C99 6.7.5p3. Because of this rule, it's
  // illegal to declare a function returning a variably modified type.
  if (const FunctionType *FT = getAsFunctionType())
    return FT->getResultType()->isVariablyModifiedType();

  return false;
}

const RecordType *Type::getAsRecordType() const {
  // If this is directly a record type, return it.
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
  return cast<RecordType>(getDesugaredType());
}

const TagType *Type::getAsTagType() const {
  // If this is directly a tag type, return it.
  if (const TagType *TagTy = dyn_cast<TagType>(this))
    return TagTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<TagType>(CanonicalType)) {
    // Look through type qualifiers
    if (isa<TagType>(CanonicalType.getUnqualifiedType()))
      return CanonicalType.getUnqualifiedType()->getAsTagType();
    return 0;
  }

  // If this is a typedef for a tag type, strip the typedef off without
  // losing all typedef information.
  return cast<TagType>(getDesugaredType());
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
    return cast<RecordType>(getDesugaredType());
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
    return cast<RecordType>(getDesugaredType());
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
  return cast<ComplexType>(getDesugaredType());
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
  return cast<VectorType>(getDesugaredType());
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
  return cast<ExtVectorType>(getDesugaredType());
}

const ObjCInterfaceType *Type::getAsObjCInterfaceType() const {
  // There is no sugar for ObjCInterfaceType's, just return the canonical
  // type pointer if it is the right class.  There is no typedef information to
  // return and these cannot be Address-space qualified.
  return dyn_cast<ObjCInterfaceType>(CanonicalType.getUnqualifiedType());
}

const ObjCQualifiedInterfaceType *
Type::getAsObjCQualifiedInterfaceType() const {
  // There is no sugar for ObjCQualifiedInterfaceType's, just return the
  // canonical type pointer if it is the right class.
  return dyn_cast<ObjCQualifiedInterfaceType>(CanonicalType.getUnqualifiedType());
}

const ObjCQualifiedIdType *Type::getAsObjCQualifiedIdType() const {
  // There is no sugar for ObjCQualifiedIdType's, just return the canonical
  // type pointer if it is the right class.
  return dyn_cast<ObjCQualifiedIdType>(CanonicalType.getUnqualifiedType());
}

const TemplateTypeParmType *Type::getAsTemplateTypeParmType() const {
  // There is no sugar for template type parameters, so just return
  // the canonical type pointer if it is the right class.
  // FIXME: can these be address-space qualified?
  return dyn_cast<TemplateTypeParmType>(CanonicalType);
}

const TemplateSpecializationType *
Type::getAsTemplateSpecializationType() const {
  // There is no sugar for class template specialization types, so
  // just return the canonical type pointer if it is the right class.
  return dyn_cast<TemplateSpecializationType>(CanonicalType);
}

bool Type::isIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::Int128;
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    // Incomplete enum types are not treated as integer types.
    // FIXME: In C++, enum types are never integer types.
    if (TT->getDecl()->isEnum() && TT->getDecl()->isDefinition())
      return true;
  if (isa<FixedWidthIntType>(CanonicalType))
    return true;
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isIntegerType();
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isIntegerType();
  return false;
}

bool Type::isIntegralType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
    BT->getKind() <= BuiltinType::LongLong;
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    if (TT->getDecl()->isEnum() && TT->getDecl()->isDefinition())
      return true;  // Complete enum types are integral.
                    // FIXME: In C++, enum types are never integral.
  if (isa<FixedWidthIntType>(CanonicalType))
    return true;
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isIntegralType();
  return false;
}

bool Type::isEnumeralType() const {
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    return TT->getDecl()->isEnum();
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isEnumeralType();
  return false;
}

bool Type::isBooleanType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Bool;
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isBooleanType();
  return false;
}

bool Type::isCharType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Char_U ||
           BT->getKind() == BuiltinType::UChar ||
           BT->getKind() == BuiltinType::Char_S ||
           BT->getKind() == BuiltinType::SChar;
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isCharType();
  return false;
}

bool Type::isWideCharType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::WChar;
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isWideCharType();
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
  
  if (const FixedWidthIntType *FWIT =
          dyn_cast<FixedWidthIntType>(CanonicalType))
    return FWIT->isSigned();
  
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isSignedIntegerType();
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isSignedIntegerType();
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

  if (const FixedWidthIntType *FWIT =
          dyn_cast<FixedWidthIntType>(CanonicalType))
    return !FWIT->isSigned();

  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isUnsignedIntegerType();
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isUnsignedIntegerType();
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
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isFloatingType();
  return false;
}

bool Type::isRealFloatingType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Float &&
           BT->getKind() <= BuiltinType::LongDouble;
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isRealFloatingType();
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isRealFloatingType();
  return false;
}

bool Type::isRealType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::LongDouble;
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    return TT->getDecl()->isEnum() && TT->getDecl()->isDefinition();
  if (isa<FixedWidthIntType>(CanonicalType))
    return true;
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isRealType();
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isRealType();
  return false;
}

bool Type::isArithmeticType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::LongDouble;
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
    // GCC allows forward declaration of enum types (forbid by C99 6.7.2.3p2).
    // If a body isn't seen by the time we get here, return false.
    return ET->getDecl()->isDefinition();
  if (isa<FixedWidthIntType>(CanonicalType))
    return true;
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isArithmeticType();
  return isa<ComplexType>(CanonicalType) || isa<VectorType>(CanonicalType);
}

bool Type::isScalarType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() != BuiltinType::Void;
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType)) {
    // Enums are scalar types, but only if they are defined.  Incomplete enums
    // are not treated as scalar types.
    if (TT->getDecl()->isEnum() && TT->getDecl()->isDefinition())
      return true;
    return false;
  }
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isScalarType();
  if (isa<FixedWidthIntType>(CanonicalType))
    return true;
  return isa<PointerType>(CanonicalType) ||
         isa<BlockPointerType>(CanonicalType) ||
         isa<MemberPointerType>(CanonicalType) ||
         isa<ComplexType>(CanonicalType) ||
         isa<ObjCQualifiedIdType>(CanonicalType);
}

/// \brief Determines whether the type is a C++ aggregate type or C
/// aggregate or union type.
///
/// An aggregate type is an array or a class type (struct, union, or
/// class) that has no user-declared constructors, no private or
/// protected non-static data members, no base classes, and no virtual
/// functions (C++ [dcl.init.aggr]p1). The notion of an aggregate type
/// subsumes the notion of C aggregates (C99 6.2.5p21) because it also
/// includes union types.
bool Type::isAggregateType() const {
  if (const RecordType *Record = dyn_cast<RecordType>(CanonicalType)) {
    if (CXXRecordDecl *ClassDecl = dyn_cast<CXXRecordDecl>(Record->getDecl()))
      return ClassDecl->isAggregate();

    return true;
  }

  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isAggregateType();
  return isa<ArrayType>(CanonicalType);
}

/// isConstantSizeType - Return true if this is not a variable sized type,
/// according to the rules of C99 6.7.5p3.  It is not legal to call this on
/// incomplete types or dependent types.
bool Type::isConstantSizeType() const {
  if (const ExtQualType *EXTQT = dyn_cast<ExtQualType>(CanonicalType))
    return EXTQT->getBaseType()->isConstantSizeType();
  assert(!isIncompleteType() && "This doesn't make sense for incomplete types");
  assert(!isDependentType() && "This doesn't make sense for dependent types");
  // The VAT must have a size, as it is known to be complete.
  return !isa<VariableArrayType>(CanonicalType);
}

/// isIncompleteType - Return true if this is an incomplete type (C99 6.2.5p1)
/// - a type that can describe objects, but which lacks information needed to
/// determine its size.
bool Type::isIncompleteType() const { 
  switch (CanonicalType->getTypeClass()) { 
  default: return false;
  case ExtQual:
    return cast<ExtQualType>(CanonicalType)->getBaseType()->isIncompleteType();
  case Builtin:
    // Void is the only incomplete builtin type.  Per C99 6.2.5p19, it can never
    // be completed.
    return isVoidType();
  case Record:
  case Enum:
    // A tagged type (struct/union/enum/class) is incomplete if the decl is a
    // forward declaration, but not a full definition (C99 6.2.5p22).
    return !cast<TagType>(CanonicalType)->getDecl()->isDefinition();
  case IncompleteArray:
    // An array of unknown size is an incomplete type (C99 6.2.5p22).
    return true;
  case ObjCInterface:
  case ObjCQualifiedInterface:
    // ObjC interfaces are incomplete if they are @class, not @interface.
    return cast<ObjCInterfaceType>(this)->getDecl()->isForwardDecl();
  }
}

/// isPODType - Return true if this is a plain-old-data type (C++ 3.9p10)
bool Type::isPODType() const {
  // The compiler shouldn't query this for incomplete types, but the user might.
  // We return false for that case.
  if (isIncompleteType())
    return false;

  switch (CanonicalType->getTypeClass()) {
    // Everything not explicitly mentioned is not POD.
  default: return false;
  case ExtQual:
    return cast<ExtQualType>(CanonicalType)->getBaseType()->isPODType();
  case VariableArray:
  case ConstantArray:
    // IncompleteArray is caught by isIncompleteType() above.
    return cast<ArrayType>(CanonicalType)->getElementType()->isPODType();

  case Builtin:
  case Complex:
  case Pointer:
  case MemberPointer:
  case Vector:
  case ExtVector:
  case ObjCQualifiedId:
    return true;

  case Enum:
    return true;

  case Record:
    if (CXXRecordDecl *ClassDecl 
          = dyn_cast<CXXRecordDecl>(cast<RecordType>(CanonicalType)->getDecl()))
      return ClassDecl->isPOD();

    // C struct/union is POD.
    return true;
  }
}

bool Type::isPromotableIntegerType() const {
  if (const BuiltinType *BT = getAsBuiltinType())
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
  return false;
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
  case Int128:            return "__int128_t";
  case UChar:             return "unsigned char";
  case UShort:            return "unsigned short";
  case UInt:              return "unsigned int";
  case ULong:             return "unsigned long";
  case ULongLong:         return "unsigned long long";
  case UInt128:           return "__uint128_t";
  case Float:             return "float";
  case Double:            return "double";
  case LongDouble:        return "long double";
  case WChar:             return "wchar_t";
  case Overload:          return "<overloaded function type>";
  case Dependent:         return "<dependent type>";
  }
}

void FunctionProtoType::Profile(llvm::FoldingSetNodeID &ID, QualType Result,
                                arg_type_iterator ArgTys,
                                unsigned NumArgs, bool isVariadic,
                                unsigned TypeQuals, bool hasExceptionSpec,
                                bool anyExceptionSpec, unsigned NumExceptions,
                                exception_iterator Exs) {
  ID.AddPointer(Result.getAsOpaquePtr());
  for (unsigned i = 0; i != NumArgs; ++i)
    ID.AddPointer(ArgTys[i].getAsOpaquePtr());
  ID.AddInteger(isVariadic);
  ID.AddInteger(TypeQuals);
  ID.AddInteger(hasExceptionSpec);
  if (hasExceptionSpec) {
    ID.AddInteger(anyExceptionSpec);
    for(unsigned i = 0; i != NumExceptions; ++i)
      ID.AddPointer(Exs[i].getAsOpaquePtr());
  }
}

void FunctionProtoType::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getResultType(), arg_type_begin(), NumArgs, isVariadic(),
          getTypeQuals(), hasExceptionSpec(), hasAnyExceptionSpec(),
          getNumExceptions(), exception_begin());
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
    /// FIXME: This is incorrect for ExtQuals!
    /// FIXME:
    TypeQuals |= CurType.getCVRQualifiers();

    TDT = dyn_cast<TypedefType>(CurType);
    if (TDT == 0)
      return QualType(CurType.getTypePtr(), TypeQuals);
  }
}

TypeOfExprType::TypeOfExprType(Expr *E, QualType can)
  : Type(TypeOfExpr, can, E->isTypeDependent()), TOExpr(E) {
  assert(!isa<TypedefType>(can) && "Invalid canonical type");
}

bool RecordType::classof(const TagType *TT) {
  return isa<RecordDecl>(TT->getDecl());
}

bool EnumType::classof(const TagType *TT) {
  return isa<EnumDecl>(TT->getDecl());
}

bool 
TemplateSpecializationType::
anyDependentTemplateArguments(const TemplateArgument *Args, unsigned NumArgs) {
  for (unsigned Idx = 0; Idx < NumArgs; ++Idx) {
    switch (Args[Idx].getKind()) {
    case TemplateArgument::Type:
      if (Args[Idx].getAsType()->isDependentType())
        return true;
      break;
      
    case TemplateArgument::Declaration:
    case TemplateArgument::Integral:
      // Never dependent
      break;

    case TemplateArgument::Expression:
      if (Args[Idx].getAsExpr()->isTypeDependent() ||
          Args[Idx].getAsExpr()->isValueDependent())
        return true;
      break;
    }
  }

  return false;
}

TemplateSpecializationType::
TemplateSpecializationType(TemplateName T, const TemplateArgument *Args,
                           unsigned NumArgs, QualType Canon)
  : Type(TemplateSpecialization, 
         Canon.isNull()? QualType(this, 0) : Canon,
         T.isDependent() || anyDependentTemplateArguments(Args, NumArgs)),
    Template(T), NumArgs(NumArgs)
{
  assert((!Canon.isNull() || 
          T.isDependent() || anyDependentTemplateArguments(Args, NumArgs)) &&
         "No canonical type for non-dependent class template specialization");

  TemplateArgument *TemplateArgs 
    = reinterpret_cast<TemplateArgument *>(this + 1);
  for (unsigned Arg = 0; Arg < NumArgs; ++Arg)
    new (&TemplateArgs[Arg]) TemplateArgument(Args[Arg]);
}

void TemplateSpecializationType::Destroy(ASTContext& C) {
  for (unsigned Arg = 0; Arg < NumArgs; ++Arg) {
    // FIXME: Not all expressions get cloned, so we can't yet perform
    // this destruction.
    //    if (Expr *E = getArg(Arg).getAsExpr())
    //      E->Destroy(C);
  }
}

TemplateSpecializationType::iterator
TemplateSpecializationType::end() const {
  return begin() + getNumArgs();
}

const TemplateArgument &
TemplateSpecializationType::getArg(unsigned Idx) const {
  assert(Idx < getNumArgs() && "Template argument out of range");
  return getArgs()[Idx];
}

void 
TemplateSpecializationType::Profile(llvm::FoldingSetNodeID &ID, 
                                    TemplateName T, 
                                    const TemplateArgument *Args, 
                                    unsigned NumArgs) {
  T.Profile(ID);
  for (unsigned Idx = 0; Idx < NumArgs; ++Idx)
    Args[Idx].Profile(ID);
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
void QualType::dump() const {
  dump("");
}

void Type::dump() const {
  std::string S = "identifier";
  getAsStringInternal(S);
  fprintf(stderr, "%s\n", S.c_str());
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
    S += "NULL TYPE";
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

void FixedWidthIntType::getAsStringInternal(std::string &S) const {
  // FIXME: Once we get bitwidth attribute, write as
  // "int __attribute__((bitwidth(x)))".
  std::string prefix = "__clang_fixedwidth";
  prefix += llvm::utostr_32(Width);
  prefix += (char)(Signed ? 'S' : 'U');
  if (S.empty()) {
    S = prefix;
  } else {
    // Prefix the basic type, e.g. 'int X'.
    S = prefix + S;
  }
}


void ComplexType::getAsStringInternal(std::string &S) const {
  ElementType->getAsStringInternal(S);
  S = "_Complex " + S;
}

void ExtQualType::getAsStringInternal(std::string &S) const {
  bool NeedsSpace = false;
  if (AddressSpace) {
    S = "__attribute__((address_space("+llvm::utostr_32(AddressSpace)+")))" + S;
    NeedsSpace = true;
  }
  if (GCAttrType != QualType::GCNone) {
    if (NeedsSpace)
      S += ' ';
    S += "__attribute__((objc_gc(";
    if (GCAttrType == QualType::Weak)
      S += "weak";
    else
      S += "strong";
    S += ")))";
  }
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

void BlockPointerType::getAsStringInternal(std::string &S) const {
  S = '^' + S;
  PointeeType.getAsStringInternal(S);
}

void LValueReferenceType::getAsStringInternal(std::string &S) const {
  S = '&' + S;

  // Handle things like 'int (&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(getPointeeType()))
    S = '(' + S + ')';

  getPointeeType().getAsStringInternal(S);
}

void RValueReferenceType::getAsStringInternal(std::string &S) const {
  S = "&&" + S;

  // Handle things like 'int (&&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(getPointeeType()))
    S = '(' + S + ')';

  getPointeeType().getAsStringInternal(S);
}

void MemberPointerType::getAsStringInternal(std::string &S) const {
  std::string C;
  Class->getAsStringInternal(C);
  C += "::*";
  S = C + S;

  // Handle things like 'int (Cls::*A)[4];' correctly.
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
    std::string SStr;
    llvm::raw_string_ostream s(SStr);
    getSizeExpr()->printPretty(s);
    S += s.str();
  }
  S += ']';
  
  getElementType().getAsStringInternal(S);
}

void DependentSizedArrayType::getAsStringInternal(std::string &S) const {
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
    std::string SStr;
    llvm::raw_string_ostream s(SStr);
    getSizeExpr()->printPretty(s);
    S += s.str();
  }
  S += ']';
  
  getElementType().getAsStringInternal(S);
}

void VectorType::getAsStringInternal(std::string &S) const {
  // FIXME: We prefer to print the size directly here, but have no way
  // to get the size of the type.
  S += " __attribute__((__vector_size__(";
  S += llvm::utostr_32(NumElements); // convert back to bytes.
  S += " * sizeof(" + ElementType.getAsString() + "))))";
  ElementType.getAsStringInternal(S);
}

void ExtVectorType::getAsStringInternal(std::string &S) const {
  S += " __attribute__((ext_vector_type(";
  S += llvm::utostr_32(NumElements);
  S += ")))";
  ElementType.getAsStringInternal(S);
}

void TypeOfExprType::getAsStringInternal(std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typeof(e) X'.
    InnerString = ' ' + InnerString;
  std::string Str;
  llvm::raw_string_ostream s(Str);
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

void FunctionNoProtoType::getAsStringInternal(std::string &S) const {
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  if (!S.empty())
    S = "(" + S + ")";
  
  S += "()";
  getResultType().getAsStringInternal(S);
}

void FunctionProtoType::getAsStringInternal(std::string &S) const {
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

void TemplateTypeParmType::getAsStringInternal(std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'parmname X'.
    InnerString = ' ' + InnerString;

  if (!Name)
    InnerString = "type-parameter-" + llvm::utostr_32(Depth) + '-' + 
      llvm::utostr_32(Index) + InnerString;
  else
    InnerString = Name->getName() + InnerString;
}

std::string TemplateSpecializationType::PrintTemplateArgumentList(
                                              const TemplateArgument *Args,
                                              unsigned NumArgs) {
  std::string SpecString;
  SpecString += '<';
  for (unsigned Arg = 0; Arg < NumArgs; ++Arg) {
    if (Arg)
      SpecString += ", ";
    
    // Print the argument into a string.
    std::string ArgString;
    switch (Args[Arg].getKind()) {
    case TemplateArgument::Type:
      Args[Arg].getAsType().getAsStringInternal(ArgString);
      break;

    case TemplateArgument::Declaration:
      ArgString = cast<NamedDecl>(Args[Arg].getAsDecl())->getNameAsString();
      break;

    case TemplateArgument::Integral:
      ArgString = Args[Arg].getAsIntegral()->toString(10, true);
      break;

    case TemplateArgument::Expression: {
      llvm::raw_string_ostream s(ArgString);
      Args[Arg].getAsExpr()->printPretty(s);
      break;
    }
    }

    // If this is the first argument and its string representation
    // begins with the global scope specifier ('::foo'), add a space
    // to avoid printing the diagraph '<:'.
    if (!Arg && !ArgString.empty() && ArgString[0] == ':')
      SpecString += ' ';

    SpecString += ArgString;
  }

  // If the last character of our string is '>', add another space to
  // keep the two '>''s separate tokens. We don't *have* to do this in
  // C++0x, but it's still good hygiene.
  if (SpecString[SpecString.size() - 1] == '>')
    SpecString += ' ';

  SpecString += '>';

  return SpecString;
}

void 
TemplateSpecializationType::
getAsStringInternal(std::string &InnerString) const {
  std::string SpecString;

  {
    llvm::raw_string_ostream OS(SpecString);
    Template.print(OS);
  }

  SpecString += PrintTemplateArgumentList(getArgs(), getNumArgs());
  if (InnerString.empty())
    InnerString.swap(SpecString);
  else
    InnerString = SpecString + ' ' + InnerString;
}

void QualifiedNameType::getAsStringInternal(std::string &InnerString) const {
  std::string MyString;

  {
    llvm::raw_string_ostream OS(MyString);
    NNS->print(OS);
  }
  
  std::string TypeStr;
  if (const TagType *TagT = dyn_cast<TagType>(NamedType.getTypePtr())) {
    // Suppress printing of 'enum', 'struct', 'union', or 'class'.
    TagT->getAsStringInternal(TypeStr, true);
  } else
    NamedType.getAsStringInternal(TypeStr);

  MyString += TypeStr;
  if (InnerString.empty())
    InnerString.swap(MyString);
  else
    InnerString = MyString + ' ' + InnerString;
}

void TypenameType::getAsStringInternal(std::string &InnerString) const {
  std::string MyString;

  {
    llvm::raw_string_ostream OS(MyString);
    OS << "typename ";
    NNS->print(OS);

    if (const IdentifierInfo *Ident = getIdentifier())
      OS << Ident->getName();
    else if (const TemplateSpecializationType *Spec = getTemplateId()) {
      Spec->getTemplateName().print(OS, true);
      OS << TemplateSpecializationType::PrintTemplateArgumentList(
                                         Spec->getArgs(), Spec->getNumArgs());
    }
  }
  
  if (InnerString.empty())
    InnerString.swap(MyString);
  else
    InnerString = MyString + ' ' + InnerString;
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
  std::string ObjCQIString = getDecl()->getNameAsString();
  ObjCQIString += '<';
  bool isFirst = true;
  for (qual_iterator I = qual_begin(), E = qual_end(); I != E; ++I) {
    if (isFirst)
      isFirst = false;
    else
      ObjCQIString += ',';
    ObjCQIString += (*I)->getNameAsString();
  }
  ObjCQIString += '>';
  InnerString = ObjCQIString + InnerString;
}

void ObjCQualifiedIdType::getAsStringInternal(std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  std::string ObjCQIString = "id";
  ObjCQIString += '<';
  int num = getNumProtocols();
  for (int i = 0; i < num; i++) {
    ObjCQIString += getProtocols(i)->getNameAsString();
    if (i < num-1)
      ObjCQIString += ',';
  }
  ObjCQIString += '>';
  InnerString = ObjCQIString + InnerString;
}

void TagType::getAsStringInternal(std::string &InnerString) const {
  getAsStringInternal(InnerString, false);
}

void TagType::getAsStringInternal(std::string &InnerString,
                                  bool SuppressTagKind) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  
  const char *Kind = SuppressTagKind? 0 : getDecl()->getKindName();
  const char *ID;
  if (const IdentifierInfo *II = getDecl()->getIdentifier())
    ID = II->getName();
  else if (TypedefDecl *Typedef = getDecl()->getTypedefForAnonDecl()) {
    Kind = 0;
    assert(Typedef->getIdentifier() && "Typedef without identifier?");
    ID = Typedef->getIdentifier()->getName();
  } else
    ID = "<anonymous>";

  // If this is a class template specialization, print the template
  // arguments.
  if (ClassTemplateSpecializationDecl *Spec 
        = dyn_cast<ClassTemplateSpecializationDecl>(getDecl())) {
    std::string TemplateArgs 
      = TemplateSpecializationType::PrintTemplateArgumentList(
                                                  Spec->getTemplateArgs(),
                                                  Spec->getNumTemplateArgs());
    InnerString = TemplateArgs + InnerString;
  }

  if (Kind) {
    // Compute the full nested-name-specifier for this type. In C,
    // this will always be empty.
    std::string ContextStr;
    for (DeclContext *DC = getDecl()->getDeclContext(); 
         !DC->isTranslationUnit(); DC = DC->getParent()) {
      std::string MyPart;
      if (NamespaceDecl *NS = dyn_cast<NamespaceDecl>(DC)) {
        if (NS->getIdentifier())
          MyPart = NS->getNameAsString();
      } else if (ClassTemplateSpecializationDecl *Spec 
                   = dyn_cast<ClassTemplateSpecializationDecl>(DC)) {
        std::string TemplateArgs 
          = TemplateSpecializationType::PrintTemplateArgumentList(
                                                  Spec->getTemplateArgs(),
                                                  Spec->getNumTemplateArgs());
        MyPart = Spec->getIdentifier()->getName() + TemplateArgs;
      } else if (TagDecl *Tag = dyn_cast<TagDecl>(DC)) {
        if (TypedefDecl *Typedef = Tag->getTypedefForAnonDecl())
          MyPart = Typedef->getIdentifier()->getName();
        else if (Tag->getIdentifier())
          MyPart = Tag->getIdentifier()->getName();
      }

      if (!MyPart.empty())
        ContextStr = MyPart + "::" + ContextStr;
    }

    InnerString = std::string(Kind) + " " + ContextStr + ID + InnerString;
  } else
    InnerString = ID + InnerString;
}
