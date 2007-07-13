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

#include "clang/Lex/IdentifierTable.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;

Type::~Type() {}

/// getSize - the number of bits to represent the type.
unsigned Type::getSize() const
{
  switch (CanonicalType->getTypeClass()) {
  case Builtin: {
    // FIXME: need to use TargetInfo to derive the target specific sizes. This
    // implementation will suffice for play with vector support.
    switch (cast<BuiltinType>(this)->getKind()) {
    case BuiltinType::Void:       return 0;
    case BuiltinType::Bool:       
    case BuiltinType::Char_S:     
    case BuiltinType::Char_U:     return sizeof(char) * 8;
    case BuiltinType::SChar:      return sizeof(signed char) * 8;
    case BuiltinType::Short:      return sizeof(short) * 8;
    case BuiltinType::Int:        return sizeof(int) * 8;
    case BuiltinType::Long:       return sizeof(long) * 8;
    case BuiltinType::LongLong:   return sizeof(long long) * 8;
    case BuiltinType::UChar:      return sizeof(unsigned char) * 8;
    case BuiltinType::UShort:     return sizeof(unsigned short) * 8;
    case BuiltinType::UInt:       return sizeof(unsigned int) * 8;
    case BuiltinType::ULong:      return sizeof(unsigned long) * 8;
    case BuiltinType::ULongLong:  return sizeof(unsigned long long) * 8;
    case BuiltinType::Float:      return sizeof(float) * 8;
    case BuiltinType::Double:     return sizeof(double) * 8;
    case BuiltinType::LongDouble: return sizeof(long double) * 8;
    }
    assert(0 && "Can't get here");
  }
  case Pointer:
    // FIXME: need to use TargetInfo again
    return sizeof(void *) * 8;
  case Reference:
    // seems that sizeof(T&) == sizeof(T) -- spec reference?
    return (cast<ReferenceType>(this)->getReferenceeType()->getSize());
  case Complex:
  case Array:
  case Vector:
  case FunctionNoProto:
  case FunctionProto:
  case TypeName:
  case Tagged:
    assert(0 && "Type sizes are not yet known, in general");
  }
  assert(0 && "Can't get here");
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
  case Array:
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

bool Type::isFunctionType() const {
  return isa<FunctionType>(CanonicalType);
}

bool Type::isPointerType() const {
  return isa<PointerType>(CanonicalType);
}

bool Type::isReferenceType() const {
  return isa<ReferenceType>(CanonicalType);
}

bool Type::isArrayType() const {
  return isa<ArrayType>(CanonicalType);
}

bool Type::isStructureType() const {
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType)) {
    if (TT->getDecl()->getKind() == Decl::Struct)
      return true;
  }
  return false;
}

bool Type::isUnionType() const { 
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType)) {
    if (TT->getDecl()->getKind() == Decl::Union)
      return true;
  }
  return false;
}

// C99 6.2.7p1: If both are complete types, then the following additional
// requirements apply...FIXME (handle compatibility across source files).
bool Type::tagTypesAreCompatible(QualType lhs, QualType rhs) {
  TagDecl *ldecl = cast<TagType>(lhs.getCanonicalType())->getDecl();
  TagDecl *rdecl = cast<TagType>(rhs.getCanonicalType())->getDecl();
  
  if (ldecl->getKind() == Decl::Struct && rdecl->getKind() == Decl::Struct) {
    if (ldecl->getIdentifier() == rdecl->getIdentifier())
      return true;
  }
  if (ldecl->getKind() == Decl::Union && rdecl->getKind() == Decl::Union) {
    if (ldecl->getIdentifier() == rdecl->getIdentifier())
      return true;
  }
  return false;
}

bool Type::pointerTypesAreCompatible(QualType lhs, QualType rhs) {
  // C99 6.7.5.1p2: For two pointer types to be compatible, both shall be 
  // identically qualified and both shall be pointers to compatible types.
  if (lhs.getQualifiers() != rhs.getQualifiers())
    return false;
    
  QualType ltype = cast<PointerType>(lhs.getCanonicalType())->getPointeeType();
  QualType rtype = cast<PointerType>(rhs.getCanonicalType())->getPointeeType();
  
  return typesAreCompatible(ltype, rtype);
}

// C++ 5.17p6: When the left opperand of an assignment operator denotes a
// reference to T, the operation assigns to the object of type T denoted by the
// reference.
bool Type::referenceTypesAreCompatible(QualType lhs, QualType rhs) {
  QualType ltype = lhs;

  if (lhs->isReferenceType())
    ltype = cast<ReferenceType>(lhs.getCanonicalType())->getReferenceeType();

  QualType rtype = rhs;

  if (rhs->isReferenceType())
    rtype = cast<ReferenceType>(rhs.getCanonicalType())->getReferenceeType();

  return typesAreCompatible(ltype, rtype);
}

bool Type::functionTypesAreCompatible(QualType lhs, QualType rhs) {
  const FunctionType *lbase = cast<FunctionType>(lhs.getCanonicalType());
  const FunctionType *rbase = cast<FunctionType>(rhs.getCanonicalType());
  const FunctionTypeProto *lproto = dyn_cast<FunctionTypeProto>(lbase);
  const FunctionTypeProto *rproto = dyn_cast<FunctionTypeProto>(rbase);

  // first check the return types (common between C99 and K&R).
  if (!typesAreCompatible(lbase->getResultType(), rbase->getResultType()))
    return false;

  if (lproto && rproto) { // two C99 style function prototypes
    unsigned lproto_nargs = lproto->getNumArgs();
    unsigned rproto_nargs = rproto->getNumArgs();
    
    if (lproto_nargs != rproto_nargs)
      return false;
      
    // both prototypes have the same number of arguments.
    if ((lproto->isVariadic() && !rproto->isVariadic()) ||
        (rproto->isVariadic() && !lproto->isVariadic()))
      return false;
      
    // The use of ellipsis agree...now check the argument types.
    for (unsigned i = 0; i < lproto_nargs; i++)
      if (!typesAreCompatible(lproto->getArgType(i), rproto->getArgType(i)))
        return false;
    return true;
  }
  if (!lproto && !rproto) // two K&R style function decls, nothing to do.
    return true;

  // we have a mixture of K&R style with C99 prototypes
  const FunctionTypeProto *proto = lproto ? lproto : rproto;
  
  if (proto->isVariadic())
    return false;
    
  // FIXME: Each parameter type T in the prototype must be compatible with the
  // type resulting from applying the usual argument conversions to T.
  return true;
}

bool Type::arrayTypesAreCompatible(QualType lhs, QualType rhs) {
  QualType ltype = cast<ArrayType>(lhs.getCanonicalType())->getElementType();
  QualType rtype = cast<ArrayType>(rhs.getCanonicalType())->getElementType();
  
  if (!typesAreCompatible(ltype, rtype))
    return false;
    
  // FIXME: If both types specify constant sizes, then the sizes must also be 
  // the same. Even if the sizes are the same, GCC produces an error.
  return true;
}

/// typesAreCompatible - C99 6.7.3p9: For two qualified types to be compatible, 
/// both shall have the identically qualified version of a compatible type.
/// C99 6.2.7p1: Two types have compatible types if their types are the 
/// same. See 6.7.[2,3,5] for additional rules.
bool Type::typesAreCompatible(QualType lhs, QualType rhs) {
  QualType lcanon = lhs.getCanonicalType();
  QualType rcanon = rhs.getCanonicalType();

  // If two types are identical, they are are compatible
  if (lcanon == rcanon)
    return true;
  
  // If the canonical type classes don't match, they can't be compatible
  if (lcanon->getTypeClass() != rcanon->getTypeClass())
    return false;

  switch (lcanon->getTypeClass()) {
    case Type::Pointer:
      return pointerTypesAreCompatible(lcanon, rcanon);
    case Type::Reference:
      return referenceTypesAreCompatible(lcanon, rcanon);
    case Type::Array:
      return arrayTypesAreCompatible(lcanon, rcanon);
    case Type::FunctionNoProto:
    case Type::FunctionProto:
      return functionTypesAreCompatible(lcanon, rcanon);
    case Type::Tagged: // handle structures, unions
      return tagTypesAreCompatible(lcanon, rcanon);
    case Type::Builtin:
      return false; 
    default:
      assert(0 && "unexpected type");
  }
  return true; // should never get here...
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

bool Type::isSignedIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType)) {
    return BT->getKind() >= BuiltinType::Char_S &&
           BT->getKind() <= BuiltinType::LongLong;
  }
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isSignedIntegerType();
  return false;
}

bool Type::isUnsignedIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType)) {
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::ULongLong;
  }
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isUnsignedIntegerType();
  return false;
}

bool Type::isFloatingType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Float &&
           BT->getKind() <= BuiltinType::LongDouble;
  if (const ComplexType *CT = dyn_cast<ComplexType>(CanonicalType))
    return CT->isFloatingType();
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

bool Type::isComplexType() const {
  return isa<ComplexType>(CanonicalType);
}

bool Type::isVectorType() const {
  return isa<VectorType>(CanonicalType);
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
  return CanonicalType->getTypeClass() == Array;
}

// The only variable size types are auto arrays within a function. Structures 
// cannot contain a VLA member. They can have a flexible array member, however
// the structure is still constant size (C99 6.7.2.1p16).
bool Type::isConstantSizeType(SourceLocation *loc) const {
  if (const ArrayType *Ary = dyn_cast<ArrayType>(CanonicalType)) {
    assert(Ary->getSizeExpr() && "Incomplete types don't have a size at all!");
    // Variable Length Array?
    return Ary->getSizeExpr()->isIntegerConstantExpr(loc);
  }
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
  case Array:
    // An array of unknown size is an incomplete type (C99 6.2.5p22).
    return cast<ArrayType>(CanonicalType)->getSizeExpr() == 0;
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
                                QualType* ArgTys,
                                unsigned NumArgs, bool isVariadic) {
  ID.AddPointer(Result.getAsOpaquePtr());
  for (unsigned i = 0; i != NumArgs; ++i)
    ID.AddPointer(ArgTys[i].getAsOpaquePtr());
  ID.AddInteger(isVariadic);
}

void FunctionTypeProto::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getResultType(), ArgInfo, NumArgs, isVariadic());
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

void ArrayType::getAsStringInternal(std::string &S) const {
  S += '[';
  
  if (IndexTypeQuals) {
    AppendTypeQualList(S, IndexTypeQuals);
    S += ' ';
  }
  
  if (SizeModifier == Static)
    S += "static";
  else if (SizeModifier == Star)
    S += '*';
  
  S += ']';
  
  ElementType.getAsStringInternal(S);
}

void VectorType::getAsStringInternal(std::string &S) const {
  S += " __attribute__(( vector_size(";
  // FIXME: handle types that are != 32 bits.
  S += llvm::utostr_32(NumElements*4); // convert back to bytes.
  S += ") ))";
  ElementType.getAsStringInternal(S);
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
