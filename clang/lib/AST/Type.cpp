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
#include "clang/AST/PrettyPrinter.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

bool QualType::isConstant(QualType T, ASTContext &Ctx) {
  if (T.isConstQualified())
    return true;

  if (const ArrayType *AT = Ctx.getAsArrayType(T))
    return AT->getElementType().isConstant(Ctx);

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
  // FIXME: Resource contention like in ConstantArrayWithExprType ?
  // May crash, depending on platform or a particular build.
  // SizeExpr->Destroy(C);
  this->~DependentSizedArrayType();
  C.Deallocate(this);
}

void DependentSizedArrayType::Profile(llvm::FoldingSetNodeID &ID,
                                      ASTContext &Context,
                                      QualType ET,
                                      ArraySizeModifier SizeMod,
                                      unsigned TypeQuals,
                                      Expr *E) {
  ID.AddPointer(ET.getAsOpaquePtr());
  ID.AddInteger(SizeMod);
  ID.AddInteger(TypeQuals);
  E->Profile(ID, Context, true);
}

void
DependentSizedExtVectorType::Profile(llvm::FoldingSetNodeID &ID,
                                     ASTContext &Context,
                                     QualType ElementType, Expr *SizeExpr) {
  ID.AddPointer(ElementType.getAsOpaquePtr());
  SizeExpr->Profile(ID, Context, true);
}

void DependentSizedExtVectorType::Destroy(ASTContext& C) {
  // FIXME: Deallocate size expression, once we're cloning properly.
//  if (SizeExpr)
//    SizeExpr->Destroy(C);
  this->~DependentSizedExtVectorType();
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
  if (!isa<ArrayType>(CanonicalType))
    return 0;

  // If this is a typedef for an array type, strip the typedef off without
  // losing all typedef information.
  return cast<ArrayType>(getUnqualifiedDesugaredType())
    ->getElementType().getTypePtr();
}

/// getDesugaredType - Return the specified type with any "sugar" removed from
/// the type.  This takes off typedefs, typeof's etc.  If the outer level of
/// the type is already concrete, it returns it unmodified.  This is similar
/// to getting the canonical type, but it doesn't remove *all* typedefs.  For
/// example, it returns "T*" as "T*", (not as "int*"), because the pointer is
/// concrete.
QualType QualType::getDesugaredType(QualType T) {
  QualifierCollector Qs;

  QualType Cur = T;
  while (true) {
    const Type *CurTy = Qs.strip(Cur);
    switch (CurTy->getTypeClass()) {
#define ABSTRACT_TYPE(Class, Parent)
#define TYPE(Class, Parent) \
    case Type::Class: { \
      const Class##Type *Ty = cast<Class##Type>(CurTy); \
      if (!Ty->isSugared()) \
        return Qs.apply(Cur); \
      Cur = Ty->desugar(); \
      break; \
    }
#include "clang/AST/TypeNodes.def"
    }
  }
}

/// getUnqualifiedDesugaredType - Pull any qualifiers and syntactic
/// sugar off the given type.  This should produce an object of the
/// same dynamic type as the canonical type.
const Type *Type::getUnqualifiedDesugaredType() const {
  const Type *Cur = this;

  while (true) {
    switch (Cur->getTypeClass()) {
#define ABSTRACT_TYPE(Class, Parent)
#define TYPE(Class, Parent) \
    case Class: { \
      const Class##Type *Ty = cast<Class##Type>(Cur); \
      if (!Ty->isSugared()) return Cur; \
      Cur = Ty->desugar().getTypePtr(); \
      break; \
    }
#include "clang/AST/TypeNodes.def"
    }
  }
}

/// isVoidType - Helper method to determine if this is the 'void' type.
bool Type::isVoidType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Void;
  return false;
}

bool Type::isObjectType() const {
  if (isa<FunctionType>(CanonicalType) || isa<ReferenceType>(CanonicalType) ||
      isa<IncompleteArrayType>(CanonicalType) || isVoidType())
    return false;
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
  case LValueReference:
  case RValueReference:
  case Record:
    return true;
  default:
    return false;
  }
}

bool Type::isClassType() const {
  if (const RecordType *RT = getAs<RecordType>())
    return RT->getDecl()->isClass();
  return false;
}
bool Type::isStructureType() const {
  if (const RecordType *RT = getAs<RecordType>())
    return RT->getDecl()->isStruct();
  return false;
}
bool Type::isVoidPointerType() const {
  if (const PointerType *PT = getAs<PointerType>())
    return PT->getPointeeType()->isVoidType();
  return false;
}

bool Type::isUnionType() const {
  if (const RecordType *RT = getAs<RecordType>())
    return RT->getDecl()->isUnion();
  return false;
}

bool Type::isComplexType() const {
  if (const ComplexType *CT = dyn_cast<ComplexType>(CanonicalType))
    return CT->getElementType()->isFloatingType();
  return false;
}

bool Type::isComplexIntegerType() const {
  // Check for GCC complex integer extension.
  return getAsComplexIntegerType();
}

const ComplexType *Type::getAsComplexIntegerType() const {
  if (const ComplexType *Complex = getAs<ComplexType>())
    if (Complex->getElementType()->isIntegerType())
      return Complex;
  return 0;
}

QualType Type::getPointeeType() const {
  if (const PointerType *PT = getAs<PointerType>())
    return PT->getPointeeType();
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>())
    return OPT->getPointeeType();
  if (const BlockPointerType *BPT = getAs<BlockPointerType>())
    return BPT->getPointeeType();
  return QualType();
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
  if (const PointerType *PT = getAs<PointerType>())
    return PT->getPointeeType()->isVariablyModifiedType();
  if (const ReferenceType *RT = getAs<ReferenceType>())
    return RT->getPointeeType()->isVariablyModifiedType();
  if (const MemberPointerType *PT = getAs<MemberPointerType>())
    return PT->getPointeeType()->isVariablyModifiedType();

  // A function can return a variably modified type
  // This one isn't completely obvious, but it follows from the
  // definition in C99 6.7.5p3. Because of this rule, it's
  // illegal to declare a function returning a variably modified type.
  if (const FunctionType *FT = getAs<FunctionType>())
    return FT->getResultType()->isVariablyModifiedType();

  return false;
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
    return cast<RecordType>(getUnqualifiedDesugaredType());
  }
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
    return cast<RecordType>(getUnqualifiedDesugaredType());
  }

  return 0;
}

const ObjCInterfaceType *Type::getAsObjCQualifiedInterfaceType() const {
  // There is no sugar for ObjCInterfaceType's, just return the canonical
  // type pointer if it is the right class.  There is no typedef information to
  // return and these cannot be Address-space qualified.
  if (const ObjCInterfaceType *OIT = getAs<ObjCInterfaceType>())
    if (OIT->getNumProtocols())
      return OIT;
  return 0;
}

bool Type::isObjCQualifiedInterfaceType() const {
  return getAsObjCQualifiedInterfaceType() != 0;
}

const ObjCObjectPointerType *Type::getAsObjCQualifiedIdType() const {
  // There is no sugar for ObjCQualifiedIdType's, just return the canonical
  // type pointer if it is the right class.
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>()) {
    if (OPT->isObjCQualifiedIdType())
      return OPT;
  }
  return 0;
}

const ObjCObjectPointerType *Type::getAsObjCInterfacePointerType() const {
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>()) {
    if (OPT->getInterfaceType())
      return OPT;
  }
  return 0;
}

const CXXRecordDecl *Type::getCXXRecordDeclForPointerType() const {
  if (const PointerType *PT = getAs<PointerType>())
    if (const RecordType *RT = PT->getPointeeType()->getAs<RecordType>())
      return dyn_cast<CXXRecordDecl>(RT->getDecl());
  return 0;
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
  return false;
}

bool Type::isEnumeralType() const {
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    return TT->getDecl()->isEnum();
  return false;
}

bool Type::isBooleanType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Bool;
  return false;
}

bool Type::isCharType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Char_U ||
           BT->getKind() == BuiltinType::UChar ||
           BT->getKind() == BuiltinType::Char_S ||
           BT->getKind() == BuiltinType::SChar;
  return false;
}

bool Type::isWideCharType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::WChar;
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
  return false;
}

bool Type::isRealFloatingType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Float &&
           BT->getKind() <= BuiltinType::LongDouble;
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isRealFloatingType();
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
  if (isa<FixedWidthIntType>(CanonicalType))
    return true;
  return isa<PointerType>(CanonicalType) ||
         isa<BlockPointerType>(CanonicalType) ||
         isa<MemberPointerType>(CanonicalType) ||
         isa<ComplexType>(CanonicalType) ||
         isa<ObjCObjectPointerType>(CanonicalType);
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

  return isa<ArrayType>(CanonicalType);
}

/// isConstantSizeType - Return true if this is not a variable sized type,
/// according to the rules of C99 6.7.5p3.  It is not legal to call this on
/// incomplete types or dependent types.
bool Type::isConstantSizeType() const {
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
  case ObjCObjectPointer:
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
  if (const BuiltinType *BT = getAs<BuiltinType>())
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

bool Type::isNullPtrType() const {
  if (const BuiltinType *BT = getAs<BuiltinType>())
    return BT->getKind() == BuiltinType::NullPtr;
  return false;
}

bool Type::isSpecifierType() const {
  // Note that this intentionally does not use the canonical type.
  switch (getTypeClass()) {
  case Builtin:
  case Record:
  case Enum:
  case Typedef:
  case Complex:
  case TypeOfExpr:
  case TypeOf:
  case TemplateTypeParm:
  case SubstTemplateTypeParm:
  case TemplateSpecialization:
  case QualifiedName:
  case Typename:
  case ObjCInterface:
  case ObjCObjectPointer:
    return true;
  default:
    return false;
  }
}

const char *Type::getTypeClassName() const {
  switch (TC) {
  default: assert(0 && "Type class not in TypeNodes.def!");
#define ABSTRACT_TYPE(Derived, Base)
#define TYPE(Derived, Base) case Derived: return #Derived;
#include "clang/AST/TypeNodes.def"
  }
}

const char *BuiltinType::getName(const LangOptions &LO) const {
  switch (getKind()) {
  default: assert(0 && "Unknown builtin type!");
  case Void:              return "void";
  case Bool:              return LO.Bool ? "bool" : "_Bool";
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
  case Char16:            return "char16_t";
  case Char32:            return "char32_t";
  case NullPtr:           return "nullptr_t";
  case Overload:          return "<overloaded function type>";
  case Dependent:         return "<dependent type>";
  case UndeducedAuto:     return "auto";
  case ObjCId:            return "id";
  case ObjCClass:         return "Class";
  }
}

void FunctionProtoType::Profile(llvm::FoldingSetNodeID &ID, QualType Result,
                                arg_type_iterator ArgTys,
                                unsigned NumArgs, bool isVariadic,
                                unsigned TypeQuals, bool hasExceptionSpec,
                                bool anyExceptionSpec, unsigned NumExceptions,
                                exception_iterator Exs, bool NoReturn) {
  ID.AddPointer(Result.getAsOpaquePtr());
  for (unsigned i = 0; i != NumArgs; ++i)
    ID.AddPointer(ArgTys[i].getAsOpaquePtr());
  ID.AddInteger(isVariadic);
  ID.AddInteger(TypeQuals);
  ID.AddInteger(hasExceptionSpec);
  if (hasExceptionSpec) {
    ID.AddInteger(anyExceptionSpec);
    for (unsigned i = 0; i != NumExceptions; ++i)
      ID.AddPointer(Exs[i].getAsOpaquePtr());
  }
  ID.AddInteger(NoReturn);
}

void FunctionProtoType::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getResultType(), arg_type_begin(), NumArgs, isVariadic(),
          getTypeQuals(), hasExceptionSpec(), hasAnyExceptionSpec(),
          getNumExceptions(), exception_begin(), getNoReturnAttr());
}

void ObjCObjectPointerType::Profile(llvm::FoldingSetNodeID &ID,
                                    QualType OIT, ObjCProtocolDecl **protocols,
                                    unsigned NumProtocols) {
  ID.AddPointer(OIT.getAsOpaquePtr());
  for (unsigned i = 0; i != NumProtocols; i++)
    ID.AddPointer(protocols[i]);
}

void ObjCObjectPointerType::Profile(llvm::FoldingSetNodeID &ID) {
  if (getNumProtocols())
    Profile(ID, getPointeeType(), &Protocols[0], getNumProtocols());
  else
    Profile(ID, getPointeeType(), 0, 0);
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
  QualifierCollector Qs;

  QualType CurType;
  const TypedefType *TDT = this;
  do {
    CurType = TDT->getDecl()->getUnderlyingType();
    TDT = dyn_cast<TypedefType>(Qs.strip(CurType));
  } while (TDT);

  return Qs.apply(CurType);
}

QualType TypedefType::desugar() const {
  return getDecl()->getUnderlyingType();
}

TypeOfExprType::TypeOfExprType(Expr *E, QualType can)
  : Type(TypeOfExpr, can, E->isTypeDependent()), TOExpr(E) {
}

QualType TypeOfExprType::desugar() const {
  return getUnderlyingExpr()->getType();
}

void DependentTypeOfExprType::Profile(llvm::FoldingSetNodeID &ID,
                                      ASTContext &Context, Expr *E) {
  E->Profile(ID, Context, true);
}

DecltypeType::DecltypeType(Expr *E, QualType underlyingType, QualType can)
  : Type(Decltype, can, E->isTypeDependent()), E(E),
  UnderlyingType(underlyingType) {
}

DependentDecltypeType::DependentDecltypeType(ASTContext &Context, Expr *E)
  : DecltypeType(E, Context.DependentTy), Context(Context) { }

void DependentDecltypeType::Profile(llvm::FoldingSetNodeID &ID,
                                    ASTContext &Context, Expr *E) {
  E->Profile(ID, Context, true);
}

TagType::TagType(TypeClass TC, TagDecl *D, QualType can)
  : Type(TC, can, D->isDependentType()), decl(D, 0) {}

bool RecordType::classof(const TagType *TT) {
  return isa<RecordDecl>(TT->getDecl());
}

bool EnumType::classof(const TagType *TT) {
  return isa<EnumDecl>(TT->getDecl());
}

static bool isDependent(const TemplateArgument &Arg) {
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    assert(false && "Should not have a NULL template argument");
    return false;

  case TemplateArgument::Type:
    return Arg.getAsType()->isDependentType();

  case TemplateArgument::Declaration:
  case TemplateArgument::Integral:
    // Never dependent
    return false;

  case TemplateArgument::Expression:
    return (Arg.getAsExpr()->isTypeDependent() ||
            Arg.getAsExpr()->isValueDependent());

  case TemplateArgument::Pack:
    assert(0 && "FIXME: Implement!");
    return false;
  }

  return false;
}

bool TemplateSpecializationType::
anyDependentTemplateArguments(const TemplateArgumentLoc *Args, unsigned N) {
  for (unsigned i = 0; i != N; ++i)
    if (isDependent(Args[i].getArgument()))
      return true;
  return false;
}

bool TemplateSpecializationType::
anyDependentTemplateArguments(const TemplateArgument *Args, unsigned N) {
  for (unsigned i = 0; i != N; ++i)
    if (isDependent(Args[i]))
      return true;
  return false;
}

TemplateSpecializationType::
TemplateSpecializationType(ASTContext &Context, TemplateName T,
                           const TemplateArgument *Args,
                           unsigned NumArgs, QualType Canon)
  : Type(TemplateSpecialization,
         Canon.isNull()? QualType(this, 0) : Canon,
         T.isDependent() || anyDependentTemplateArguments(Args, NumArgs)),
    Context(Context),
    Template(T), NumArgs(NumArgs) {
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
                                    unsigned NumArgs,
                                    ASTContext &Context) {
  T.Profile(ID);
  for (unsigned Idx = 0; Idx < NumArgs; ++Idx)
    Args[Idx].Profile(ID, Context);
}

QualType QualifierCollector::apply(QualType QT) const {
  if (!hasNonFastQualifiers())
    return QT.withFastQualifiers(getFastQualifiers());

  assert(Context && "extended qualifiers but no context!");
  return Context->getQualifiedType(QT, *this);
}

QualType QualifierCollector::apply(const Type *T) const {
  if (!hasNonFastQualifiers())
    return QualType(T, getFastQualifiers());

  assert(Context && "extended qualifiers but no context!");
  return Context->getQualifiedType(T, *this);
}


//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

void QualType::dump(const char *msg) const {
  std::string R = "identifier";
  LangOptions LO;
  getAsStringInternal(R, PrintingPolicy(LO));
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
  LangOptions LO;
  getAsStringInternal(S, PrintingPolicy(LO));
  fprintf(stderr, "%s\n", S.c_str());
}



static void AppendTypeQualList(std::string &S, unsigned TypeQuals) {
  if (TypeQuals & Qualifiers::Const) {
    if (!S.empty()) S += ' ';
    S += "const";
  }
  if (TypeQuals & Qualifiers::Volatile) {
    if (!S.empty()) S += ' ';
    S += "volatile";
  }
  if (TypeQuals & Qualifiers::Restrict) {
    if (!S.empty()) S += ' ';
    S += "restrict";
  }
}

std::string Qualifiers::getAsString() const {
  LangOptions LO;
  return getAsString(PrintingPolicy(LO));
}

// Appends qualifiers to the given string, separated by spaces.  Will
// prefix a space if the string is non-empty.  Will not append a final
// space.
void Qualifiers::getAsStringInternal(std::string &S,
                                     const PrintingPolicy&) const {
  AppendTypeQualList(S, getCVRQualifiers());
  if (unsigned AddressSpace = getAddressSpace()) {
    if (!S.empty()) S += ' ';
    S += "__attribute__((address_space(";
    S += llvm::utostr_32(AddressSpace);
    S += ")))";
  }
  if (Qualifiers::GC GCAttrType = getObjCGCAttr()) {
    if (!S.empty()) S += ' ';
    S += "__attribute__((objc_gc(";
    if (GCAttrType == Qualifiers::Weak)
      S += "weak";
    else
      S += "strong";
    S += ")))";
  }
}

std::string QualType::getAsString() const {
  std::string S;
  LangOptions LO;
  getAsStringInternal(S, PrintingPolicy(LO));
  return S;
}

void
QualType::getAsStringInternal(std::string &S,
                              const PrintingPolicy &Policy) const {
  if (isNull()) {
    S += "NULL TYPE";
    return;
  }

  if (Policy.SuppressSpecifiers && getTypePtr()->isSpecifierType())
    return;

  // Print qualifiers as appropriate.
  Qualifiers Quals = getQualifiers();
  if (!Quals.empty()) {
    std::string TQS;
    Quals.getAsStringInternal(TQS, Policy);

    if (!S.empty()) {
      TQS += ' ';
      TQS += S;
    }
    std::swap(S, TQS);
  }

  getTypePtr()->getAsStringInternal(S, Policy);
}

void BuiltinType::getAsStringInternal(std::string &S,
                                      const PrintingPolicy &Policy) const {
  if (S.empty()) {
    S = getName(Policy.LangOpts);
  } else {
    // Prefix the basic type, e.g. 'int X'.
    S = ' ' + S;
    S = getName(Policy.LangOpts) + S;
  }
}

void FixedWidthIntType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
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


void ComplexType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  ElementType->getAsStringInternal(S, Policy);
  S = "_Complex " + S;
}

void PointerType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  S = '*' + S;

  // Handle things like 'int (*A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(getPointeeType()))
    S = '(' + S + ')';

  getPointeeType().getAsStringInternal(S, Policy);
}

void BlockPointerType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  S = '^' + S;
  PointeeType.getAsStringInternal(S, Policy);
}

void LValueReferenceType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  S = '&' + S;

  // Handle things like 'int (&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(getPointeeTypeAsWritten()))
    S = '(' + S + ')';

  getPointeeTypeAsWritten().getAsStringInternal(S, Policy);
}

void RValueReferenceType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  S = "&&" + S;

  // Handle things like 'int (&&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(getPointeeTypeAsWritten()))
    S = '(' + S + ')';

  getPointeeTypeAsWritten().getAsStringInternal(S, Policy);
}

void MemberPointerType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  std::string C;
  Class->getAsStringInternal(C, Policy);
  C += "::*";
  S = C + S;

  // Handle things like 'int (Cls::*A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(getPointeeType()))
    S = '(' + S + ')';

  getPointeeType().getAsStringInternal(S, Policy);
}

void ConstantArrayType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  S += '[';
  S += llvm::utostr(getSize().getZExtValue());
  S += ']';

  getElementType().getAsStringInternal(S, Policy);
}

void IncompleteArrayType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  S += "[]";

  getElementType().getAsStringInternal(S, Policy);
}

void VariableArrayType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  S += '[';

  if (getIndexTypeQualifiers().hasQualifiers()) {
    AppendTypeQualList(S, getIndexTypeCVRQualifiers());
    S += ' ';
  }

  if (getSizeModifier() == Static)
    S += "static";
  else if (getSizeModifier() == Star)
    S += '*';

  if (getSizeExpr()) {
    std::string SStr;
    llvm::raw_string_ostream s(SStr);
    getSizeExpr()->printPretty(s, 0, Policy);
    S += s.str();
  }
  S += ']';

  getElementType().getAsStringInternal(S, Policy);
}

void DependentSizedArrayType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  S += '[';

  if (getIndexTypeQualifiers().hasQualifiers()) {
    AppendTypeQualList(S, getIndexTypeCVRQualifiers());
    S += ' ';
  }

  if (getSizeModifier() == Static)
    S += "static";
  else if (getSizeModifier() == Star)
    S += '*';

  if (getSizeExpr()) {
    std::string SStr;
    llvm::raw_string_ostream s(SStr);
    getSizeExpr()->printPretty(s, 0, Policy);
    S += s.str();
  }
  S += ']';

  getElementType().getAsStringInternal(S, Policy);
}

void DependentSizedExtVectorType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  getElementType().getAsStringInternal(S, Policy);

  S += " __attribute__((ext_vector_type(";
  if (getSizeExpr()) {
    std::string SStr;
    llvm::raw_string_ostream s(SStr);
    getSizeExpr()->printPretty(s, 0, Policy);
    S += s.str();
  }
  S += ")))";
}

void VectorType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  // FIXME: We prefer to print the size directly here, but have no way
  // to get the size of the type.
  S += " __attribute__((__vector_size__(";
  S += llvm::utostr_32(NumElements); // convert back to bytes.
  S += " * sizeof(" + ElementType.getAsString() + "))))";
  ElementType.getAsStringInternal(S, Policy);
}

void ExtVectorType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  S += " __attribute__((ext_vector_type(";
  S += llvm::utostr_32(NumElements);
  S += ")))";
  ElementType.getAsStringInternal(S, Policy);
}

void TypeOfExprType::getAsStringInternal(std::string &InnerString, const PrintingPolicy &Policy) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typeof(e) X'.
    InnerString = ' ' + InnerString;
  std::string Str;
  llvm::raw_string_ostream s(Str);
  getUnderlyingExpr()->printPretty(s, 0, Policy);
  InnerString = "typeof " + s.str() + InnerString;
}

void TypeOfType::getAsStringInternal(std::string &InnerString, const PrintingPolicy &Policy) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typeof(t) X'.
    InnerString = ' ' + InnerString;
  std::string Tmp;
  getUnderlyingType().getAsStringInternal(Tmp, Policy);
  InnerString = "typeof(" + Tmp + ")" + InnerString;
}

void DecltypeType::getAsStringInternal(std::string &InnerString,
                                       const PrintingPolicy &Policy) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'decltype(t) X'.
    InnerString = ' ' + InnerString;
  std::string Str;
  llvm::raw_string_ostream s(Str);
  getUnderlyingExpr()->printPretty(s, 0, Policy);
  InnerString = "decltype(" + s.str() + ")" + InnerString;
}

void FunctionNoProtoType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  if (!S.empty())
    S = "(" + S + ")";

  S += "()";
  if (getNoReturnAttr())
    S += " __attribute__((noreturn))";
  getResultType().getAsStringInternal(S, Policy);
}

void FunctionProtoType::getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const {
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  if (!S.empty())
    S = "(" + S + ")";

  S += "(";
  std::string Tmp;
  PrintingPolicy ParamPolicy(Policy);
  ParamPolicy.SuppressSpecifiers = false;
  for (unsigned i = 0, e = getNumArgs(); i != e; ++i) {
    if (i) S += ", ";
    getArgType(i).getAsStringInternal(Tmp, ParamPolicy);
    S += Tmp;
    Tmp.clear();
  }

  if (isVariadic()) {
    if (getNumArgs())
      S += ", ";
    S += "...";
  } else if (getNumArgs() == 0 && !Policy.LangOpts.CPlusPlus) {
    // Do not emit int() if we have a proto, emit 'int(void)'.
    S += "void";
  }

  S += ")";
  if (getNoReturnAttr())
    S += " __attribute__((noreturn))";
  getResultType().getAsStringInternal(S, Policy);
}


void TypedefType::getAsStringInternal(std::string &InnerString, const PrintingPolicy &Policy) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  InnerString = getDecl()->getIdentifier()->getName().str() + InnerString;
}

void TemplateTypeParmType::getAsStringInternal(std::string &InnerString, const PrintingPolicy &Policy) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'parmname X'.
    InnerString = ' ' + InnerString;

  if (!Name)
    InnerString = "type-parameter-" + llvm::utostr_32(Depth) + '-' +
      llvm::utostr_32(Index) + InnerString;
  else
    InnerString = Name->getName().str() + InnerString;
}

void SubstTemplateTypeParmType::getAsStringInternal(std::string &InnerString, const PrintingPolicy &Policy) const {
  getReplacementType().getAsStringInternal(InnerString, Policy);
}

static void PrintTemplateArgument(std::string &Buffer,
                                  const TemplateArgument &Arg,
                                  const PrintingPolicy &Policy) {
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    assert(false && "Null template argument");
    break;
    
  case TemplateArgument::Type:
    Arg.getAsType().getAsStringInternal(Buffer, Policy);
    break;

  case TemplateArgument::Declaration:
    Buffer = cast<NamedDecl>(Arg.getAsDecl())->getNameAsString();
    break;

  case TemplateArgument::Integral:
    Buffer = Arg.getAsIntegral()->toString(10, true);
    break;

  case TemplateArgument::Expression: {
    llvm::raw_string_ostream s(Buffer);
    Arg.getAsExpr()->printPretty(s, 0, Policy);
    break;
  }

  case TemplateArgument::Pack:
    assert(0 && "FIXME: Implement!");
    break;
  }
}

std::string
TemplateSpecializationType::PrintTemplateArgumentList(
                                                  const TemplateArgument *Args,
                                                  unsigned NumArgs,
                                                  const PrintingPolicy &Policy) {
  std::string SpecString;
  SpecString += '<';
  for (unsigned Arg = 0; Arg < NumArgs; ++Arg) {
    if (Arg)
      SpecString += ", ";

    // Print the argument into a string.
    std::string ArgString;
    PrintTemplateArgument(ArgString, Args[Arg], Policy);

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

// Sadly, repeat all that with TemplateArgLoc.
std::string TemplateSpecializationType::
PrintTemplateArgumentList(const TemplateArgumentLoc *Args, unsigned NumArgs,
                          const PrintingPolicy &Policy) {
  std::string SpecString;
  SpecString += '<';
  for (unsigned Arg = 0; Arg < NumArgs; ++Arg) {
    if (Arg)
      SpecString += ", ";

    // Print the argument into a string.
    std::string ArgString;
    PrintTemplateArgument(ArgString, Args[Arg].getArgument(), Policy);

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
getAsStringInternal(std::string &InnerString, const PrintingPolicy &Policy) const {
  std::string SpecString;

  {
    llvm::raw_string_ostream OS(SpecString);
    Template.print(OS, Policy);
  }

  SpecString += PrintTemplateArgumentList(getArgs(), getNumArgs(), Policy);
  if (InnerString.empty())
    InnerString.swap(SpecString);
  else
    InnerString = SpecString + ' ' + InnerString;
}

void QualifiedNameType::getAsStringInternal(std::string &InnerString, const PrintingPolicy &Policy) const {
  std::string MyString;

  {
    llvm::raw_string_ostream OS(MyString);
    NNS->print(OS, Policy);
  }

  std::string TypeStr;
  PrintingPolicy InnerPolicy(Policy);
  InnerPolicy.SuppressTagKind = true;
  InnerPolicy.SuppressScope = true;
  NamedType.getAsStringInternal(TypeStr, InnerPolicy);

  MyString += TypeStr;
  if (InnerString.empty())
    InnerString.swap(MyString);
  else
    InnerString = MyString + ' ' + InnerString;
}

void TypenameType::getAsStringInternal(std::string &InnerString, const PrintingPolicy &Policy) const {
  std::string MyString;

  {
    llvm::raw_string_ostream OS(MyString);
    OS << "typename ";
    NNS->print(OS, Policy);

    if (const IdentifierInfo *Ident = getIdentifier())
      OS << Ident->getName();
    else if (const TemplateSpecializationType *Spec = getTemplateId()) {
      Spec->getTemplateName().print(OS, Policy, true);
      OS << TemplateSpecializationType::PrintTemplateArgumentList(
                                                               Spec->getArgs(),
                                                            Spec->getNumArgs(),
                                                               Policy);
    }
  }

  if (InnerString.empty())
    InnerString.swap(MyString);
  else
    InnerString = MyString + ' ' + InnerString;
}

void ObjCInterfaceType::Profile(llvm::FoldingSetNodeID &ID,
                                         const ObjCInterfaceDecl *Decl,
                                         ObjCProtocolDecl **protocols,
                                         unsigned NumProtocols) {
  ID.AddPointer(Decl);
  for (unsigned i = 0; i != NumProtocols; i++)
    ID.AddPointer(protocols[i]);
}

void ObjCInterfaceType::Profile(llvm::FoldingSetNodeID &ID) {
  if (getNumProtocols())
    Profile(ID, getDecl(), &Protocols[0], getNumProtocols());
  else
    Profile(ID, getDecl(), 0, 0);
}

void ObjCInterfaceType::getAsStringInternal(std::string &InnerString,
                                           const PrintingPolicy &Policy) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;

  std::string ObjCQIString = getDecl()->getNameAsString();
  if (getNumProtocols()) {
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
  }
  InnerString = ObjCQIString + InnerString;
}

void ObjCObjectPointerType::getAsStringInternal(std::string &InnerString,
                                                const PrintingPolicy &Policy) const {
  std::string ObjCQIString;

  if (isObjCIdType() || isObjCQualifiedIdType())
    ObjCQIString = "id";
  else if (isObjCClassType() || isObjCQualifiedClassType())
    ObjCQIString = "Class";
  else
    ObjCQIString = getInterfaceDecl()->getNameAsString();

  if (!qual_empty()) {
    ObjCQIString += '<';
    for (qual_iterator I = qual_begin(), E = qual_end(); I != E; ++I) {
      ObjCQIString += (*I)->getNameAsString();
      if (I+1 != E)
        ObjCQIString += ',';
    }
    ObjCQIString += '>';
  }

  PointeeType.getQualifiers().getAsStringInternal(ObjCQIString, Policy);

  if (!isObjCIdType() && !isObjCQualifiedIdType())
    ObjCQIString += " *"; // Don't forget the implicit pointer.
  else if (!InnerString.empty()) // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;

  InnerString = ObjCQIString + InnerString;
}

void ElaboratedType::getAsStringInternal(std::string &InnerString,
                                         const PrintingPolicy &Policy) const {
  std::string TypeStr;
  PrintingPolicy InnerPolicy(Policy);
  InnerPolicy.SuppressTagKind = true;
  UnderlyingType.getAsStringInternal(InnerString, InnerPolicy);

  InnerString = std::string(getNameForTagKind(getTagKind())) + ' ' + InnerString;
}

void TagType::getAsStringInternal(std::string &InnerString, const PrintingPolicy &Policy) const {
  if (Policy.SuppressTag)
    return;

  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;

  const char *Kind = Policy.SuppressTagKind? 0 : getDecl()->getKindName();
  const char *ID;
  if (const IdentifierInfo *II = getDecl()->getIdentifier())
    ID = II->getNameStart();
  else if (TypedefDecl *Typedef = getDecl()->getTypedefForAnonDecl()) {
    Kind = 0;
    assert(Typedef->getIdentifier() && "Typedef without identifier?");
    ID = Typedef->getIdentifier()->getNameStart();
  } else
    ID = "<anonymous>";

  // If this is a class template specialization, print the template
  // arguments.
  if (ClassTemplateSpecializationDecl *Spec
        = dyn_cast<ClassTemplateSpecializationDecl>(getDecl())) {
    const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
    std::string TemplateArgsStr
      = TemplateSpecializationType::PrintTemplateArgumentList(
                                            TemplateArgs.getFlatArgumentList(),
                                            TemplateArgs.flat_size(),
                                                              Policy);
    InnerString = TemplateArgsStr + InnerString;
  }

  if (!Policy.SuppressScope) {
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
        const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
        std::string TemplateArgsStr
          = TemplateSpecializationType::PrintTemplateArgumentList(
                                           TemplateArgs.getFlatArgumentList(),
                                           TemplateArgs.flat_size(),
                                           Policy);
        MyPart = Spec->getIdentifier()->getName().str() + TemplateArgsStr;
      } else if (TagDecl *Tag = dyn_cast<TagDecl>(DC)) {
        if (TypedefDecl *Typedef = Tag->getTypedefForAnonDecl())
          MyPart = Typedef->getIdentifier()->getName();
        else if (Tag->getIdentifier())
          MyPart = Tag->getIdentifier()->getName();
      }

      if (!MyPart.empty())
        ContextStr = MyPart + "::" + ContextStr;
    }

    if (Kind)
      InnerString = std::string(Kind) + ' ' + ContextStr + ID + InnerString;
    else
      InnerString = ContextStr + ID + InnerString;
  } else
    InnerString = ID + InnerString;
}
