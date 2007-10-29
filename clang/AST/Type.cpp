//===--- Type.cpp - Type representation and manipulation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

Type::~Type() {}

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
  case FunctionProto:
  case FunctionNoProto:
  case Reference:
    return true;
  case Tagged: {
    const TagType *TT = cast<TagType>(CanonicalType);
    const Decl::Kind Kind = TT->getDecl()->getKind();
    return Kind == Decl::Struct || Kind == Decl::Union;
  }
  default:
    return false;
  }
}

bool Type::isStructureType() const {
  if (const RecordType *RT = dyn_cast<RecordType>(this))
    if (RT->getDecl()->getKind() == Decl::Struct)
      return true;
  return false;
}
bool Type::isUnionType() const {
  if (const RecordType *RT = dyn_cast<RecordType>(this))
    if (RT->getDecl()->getKind() == Decl::Union)
      return true;
  return false;
}

bool Type::isComplexType() const {
  return isa<ComplexType>(CanonicalType);
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
  if (!isa<BuiltinType>(CanonicalType))
    return 0;

  // If this is a typedef for a builtin type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsBuiltinType();
}

const FunctionType *Type::getAsFunctionType() const {
  // If this is directly a function type, return it.
  if (const FunctionType *FTy = dyn_cast<FunctionType>(this))
    return FTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<FunctionType>(CanonicalType))
    return 0;
  
  // If this is a typedef for a function type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsFunctionType();
}

const PointerType *Type::getAsPointerType() const {
  // If this is directly a pointer type, return it.
  if (const PointerType *PTy = dyn_cast<PointerType>(this))
    return PTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<PointerType>(CanonicalType))
    return 0;

  // If this is a typedef for a pointer type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsPointerType();
}

const ReferenceType *Type::getAsReferenceType() const {
  // If this is directly a reference type, return it.
  if (const ReferenceType *RTy = dyn_cast<ReferenceType>(this))
    return RTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ReferenceType>(CanonicalType))
    return 0;

  // If this is a typedef for a reference type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsReferenceType();
}

const ArrayType *Type::getAsArrayType() const {
  // If this is directly an array type, return it.
  if (const ArrayType *ATy = dyn_cast<ArrayType>(this))
    return ATy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ArrayType>(CanonicalType))
    return 0;
  
  // If this is a typedef for an array type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsArrayType();
}

const ConstantArrayType *Type::getAsConstantArrayType() const {
  // If this is directly a constant array type, return it.
  if (const ConstantArrayType *ATy = dyn_cast<ConstantArrayType>(this))
    return ATy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ConstantArrayType>(CanonicalType))
    return 0;
  
  // If this is a typedef for a constant array type, strip the typedef off
  // without losing all typedef information.
  return getDesugaredType()->getAsConstantArrayType();
}

const VariableArrayType *Type::getAsVariableArrayType() const {
  // If this is directly a variable array type, return it.
  if (const VariableArrayType *ATy = dyn_cast<VariableArrayType>(this))
    return ATy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<VariableArrayType>(CanonicalType))
    return 0;

  // If this is a typedef for a variable array type, strip the typedef off
  // without losing all typedef information.
  return getDesugaredType()->getAsVariableArrayType();
}

/// isVariablyModifiedType (C99 6.7.5.2p2) - Return true for variable array
/// types that have a non-constant expression. This does not include "[]".
bool Type::isVariablyModifiedType() const {
  if (const VariableArrayType *VAT = getAsVariableArrayType()) {
    if (VAT->getSizeExpr())
      return true;
  }
  return false;
}

const VariableArrayType *Type::getAsVariablyModifiedType() const {
  if (const VariableArrayType *VAT = getAsVariableArrayType()) {
    if (VAT->getSizeExpr())
      return VAT;
  }
  return 0;
}

const RecordType *Type::getAsRecordType() const {
  // If this is directly a reference type, return it.
  if (const RecordType *RTy = dyn_cast<RecordType>(this))
    return RTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<RecordType>(CanonicalType))
    return 0;

  // If this is a typedef for a record type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsRecordType();
}

const RecordType *Type::getAsStructureType() const {
  // If this is directly a structure type, return it.
  if (const RecordType *RT = dyn_cast<RecordType>(this)) {
    if (RT->getDecl()->getKind() == Decl::Struct)
      return RT;
  }

  // If the canonical form of this type isn't the right kind, reject it.
  if (const RecordType *RT = dyn_cast<RecordType>(CanonicalType)) {
    if (RT->getDecl()->getKind() != Decl::Struct)
      return 0;
    
    // If this is a typedef for a structure type, strip the typedef off without
    // losing all typedef information.
    return getDesugaredType()->getAsStructureType();
  }
  return 0;
}

const RecordType *Type::getAsUnionType() const { 
  // If this is directly a union type, return it.
  if (const RecordType *RT = dyn_cast<RecordType>(this)) {
    if (RT->getDecl()->getKind() == Decl::Union)
      return RT;
  }
  // If the canonical form of this type isn't the right kind, reject it.
  if (const RecordType *RT = dyn_cast<RecordType>(CanonicalType)) {
    if (RT->getDecl()->getKind() != Decl::Union)
      return 0;

    // If this is a typedef for a union type, strip the typedef off without
    // losing all typedef information.
    return getDesugaredType()->getAsUnionType();
  }
  return 0;
}

const ComplexType *Type::getAsComplexType() const {
  // Are we directly a complex type?
  if (const ComplexType *CTy = dyn_cast<ComplexType>(this))
    return CTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ComplexType>(CanonicalType))
    return 0;

  // If this is a typedef for a complex type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsComplexType();
}

const VectorType *Type::getAsVectorType() const {
  // Are we directly a vector type?
  if (const VectorType *VTy = dyn_cast<VectorType>(this))
    return VTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<VectorType>(CanonicalType))
    return 0;

  // If this is a typedef for a vector type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsVectorType();
}

const OCUVectorType *Type::getAsOCUVectorType() const {
  // Are we directly an OpenCU vector type?
  if (const OCUVectorType *VTy = dyn_cast<OCUVectorType>(this))
    return VTy;
  
  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<OCUVectorType>(CanonicalType))
    return 0;

  // If this is a typedef for an ocuvector type, strip the typedef off without
  // losing all typedef information.
  return getDesugaredType()->getAsOCUVectorType();
}

bool Type::isIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::LongLong;
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    if (TT->getDecl()->getKind() == Decl::Enum)
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
    if (TT->getDecl()->getKind() == Decl::Enum)
      return true;
  return false;
}

bool Type::isEnumeralType() const {
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    return TT->getDecl()->getKind() == Decl::Enum;
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

/// isSignedIntegerType - Return true if this is an integer type that is
/// signed, according to C99 6.2.5p4 [char, signed char, short, int, long..],
/// an enum decl which has a signed representation, or a vector of signed
/// integer element type.
bool Type::isSignedIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType)) {
    return BT->getKind() >= BuiltinType::Char_S &&
           BT->getKind() <= BuiltinType::LongLong;
  }
  
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    if (const EnumDecl *ED = dyn_cast<EnumDecl>(TT->getDecl()))
      return ED->getIntegerType()->isSignedIntegerType();
  
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

  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    if (const EnumDecl *ED = dyn_cast<EnumDecl>(TT->getDecl()))
      return ED->getIntegerType()->isUnsignedIntegerType();

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
    return TT->getDecl()->getKind() == Decl::Enum;
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isRealType();
  return false;
}

bool Type::isArithmeticType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() != BuiltinType::Void;
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType))
    if (TT->getDecl()->getKind() == Decl::Enum)
      return true;
  return isa<ComplexType>(CanonicalType) || isa<VectorType>(CanonicalType);
}

bool Type::isScalarType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() != BuiltinType::Void;
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType)) {
    if (TT->getDecl()->getKind() == Decl::Enum)
      return true;
    return false;
  }
  return isa<PointerType>(CanonicalType) || isa<ComplexType>(CanonicalType) ||
         isa<VectorType>(CanonicalType);
}

bool Type::isAggregateType() const {
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType)) {
    if (TT->getDecl()->getKind() == Decl::Struct)
      return true;
    return false;
  }
  return CanonicalType->getTypeClass() == ConstantArray ||
         CanonicalType->getTypeClass() == VariableArray;
}

// The only variable size types are auto arrays within a function. Structures 
// cannot contain a VLA member. They can have a flexible array member, however
// the structure is still constant size (C99 6.7.2.1p16).
bool Type::isConstantSizeType(ASTContext &Ctx, SourceLocation *loc) const {
  if (isa<VariableArrayType>(CanonicalType))
    return false;
  return true;
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
  case Tagged:
    // A tagged type (struct/union/enum/class) is incomplete if the decl is a
    // forward declaration, but not a full definition (C99 6.2.5p22).
    return !cast<TagType>(CanonicalType)->getDecl()->isDefinition();
  case VariableArray:
    // An array of unknown size is an incomplete type (C99 6.2.5p22).
    return cast<VariableArrayType>(CanonicalType)->getSizeExpr() == 0;
  }
}

bool Type::isPromotableIntegerType() const {
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

void ObjcQualifiedInterfaceType::Profile(llvm::FoldingSetNodeID &ID,
                                         ObjcInterfaceType *interfaceType, 
                                         ObjcProtocolDecl **protocols, 
                                         unsigned NumProtocols) {
  ID.AddPointer(interfaceType);
  for (unsigned i = 0; i != NumProtocols; i++)
    ID.AddPointer(protocols[i]);
}

void ObjcQualifiedInterfaceType::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getInterfaceType(), &Protocols[0], getNumProtocols());
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
    TypeQuals |= CurType.getQualifiers();

    TDT = dyn_cast<TypedefType>(CurType);
    if (TDT == 0)
      return QualType(CurType.getTypePtr(), TypeQuals);
  }
}

bool RecordType::classof(const Type *T) {
  if (const TagType *TT = dyn_cast<TagType>(T))
    return isa<RecordDecl>(TT->getDecl());
  return false;
}


//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

void QualType::dump(const char *msg) const {
  std::string R = "foo";
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
  unsigned TQ = getQualifiers();
  if (TQ) {
    std::string TQS;
    AppendTypeQualList(TQS, TQ);
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

void PointerType::getAsStringInternal(std::string &S) const {
  S = '*' + S;
  
  // Handle things like 'int (*A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(PointeeType.getTypePtr()))
    S = '(' + S + ')';
  
  PointeeType.getAsStringInternal(S);
}

void ReferenceType::getAsStringInternal(std::string &S) const {
  S = '&' + S;
  
  // Handle things like 'int (&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(ReferenceeType.getTypePtr()))
    S = '(' + S + ')';
  
  ReferenceeType.getAsStringInternal(S);
}

void ConstantArrayType::getAsStringInternal(std::string &S) const {
  S += '[';
  S += llvm::utostr(getSize().getZExtValue());
  S += ']';
  
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
  S += " __attribute__((vector_size(";
  // FIXME: should multiply by element size somehow.
  S += llvm::utostr_32(NumElements*4); // convert back to bytes.
  S += ")))";
  ElementType.getAsStringInternal(S);
}

void OCUVectorType::getAsStringInternal(std::string &S) const {
  S += " __attribute__((ocu_vector_type(";
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

void ObjcInterfaceType::getAsStringInternal(std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  InnerString = getDecl()->getIdentifier()->getName() + InnerString;
}

void ObjcQualifiedInterfaceType::getAsStringInternal(
                                  std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  std::string ObjcQIString = getInterfaceType()->getDecl()->getName();
  ObjcQIString += '<';
  int num = getNumProtocols();
  for (int i = 0; i < num; i++) {
    ObjcQIString += getProtocols(i)->getName();
    if (i < num-1)
      ObjcQIString += ',';
  }
  ObjcQIString += '>';
  InnerString = ObjcQIString + InnerString;
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
