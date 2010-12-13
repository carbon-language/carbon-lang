//===--- TypeLoc.cpp - Type Source Info Wrapper -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TypeLoc subclasses implementations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// TypeLoc Implementation
//===----------------------------------------------------------------------===//

namespace {
  class TypeLocRanger : public TypeLocVisitor<TypeLocRanger, SourceRange> {
  public:
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
    SourceRange Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc) { \
      return TyLoc.getLocalSourceRange(); \
    }
#include "clang/AST/TypeLocNodes.def"
  };
}

SourceRange TypeLoc::getLocalSourceRangeImpl(TypeLoc TL) {
  if (TL.isNull()) return SourceRange();
  return TypeLocRanger().Visit(TL);
}

namespace {
  class TypeSizer : public TypeLocVisitor<TypeSizer, unsigned> {
  public:
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
    unsigned Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc) { \
      return TyLoc.getFullDataSize(); \
    }
#include "clang/AST/TypeLocNodes.def"
  };
}

/// \brief Returns the size of the type source info data block.
unsigned TypeLoc::getFullDataSizeForType(QualType Ty) {
  if (Ty.isNull()) return 0;
  return TypeSizer().Visit(TypeLoc(Ty, 0));
}

namespace {
  class NextLoc : public TypeLocVisitor<NextLoc, TypeLoc> {
  public:
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
    TypeLoc Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc) { \
      return TyLoc.getNextTypeLoc(); \
    }
#include "clang/AST/TypeLocNodes.def"
  };
}

/// \brief Get the next TypeLoc pointed by this TypeLoc, e.g for "int*" the
/// TypeLoc is a PointerLoc and next TypeLoc is for "int".
TypeLoc TypeLoc::getNextTypeLocImpl(TypeLoc TL) {
  return NextLoc().Visit(TL);
}

/// \brief Initializes a type location, and all of its children
/// recursively, as if the entire tree had been written in the
/// given location.
void TypeLoc::initializeImpl(TypeLoc TL, SourceLocation Loc) {
  while (true) {
    switch (TL.getTypeLocClass()) {
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT)        \
    case CLASS: {                     \
      CLASS##TypeLoc TLCasted = cast<CLASS##TypeLoc>(TL); \
      TLCasted.initializeLocal(Loc);  \
      TL = TLCasted.getNextTypeLoc(); \
      if (!TL) return;                \
      continue;                       \
    }
#include "clang/AST/TypeLocNodes.def"
    }
  }
}

/// \brief Initializes a type location by copying all its data from
/// another type location of the same type.
void TypeLoc::initializeFullCopyImpl(TypeLoc TL, TypeLoc Other) {
  assert(TL.getType() == Other.getType() && "Must copy from same type");
  memcpy(TL.getOpaqueData(), Other.getOpaqueData(), TL.getFullDataSize());
}

SourceLocation TypeLoc::getBeginLoc() const {
  TypeLoc Cur = *this;
  while (true) {
    switch (Cur.getTypeLocClass()) {
    // FIXME: Currently QualifiedTypeLoc does not have a source range
    // case Qualified:
    case Elaborated:
      break;
    default:
      TypeLoc Next = Cur.getNextTypeLoc();
      if (Next.isNull()) break;
      Cur = Next;
      continue;
    }
    break;
  }
  return Cur.getLocalSourceRange().getBegin();
}

SourceLocation TypeLoc::getEndLoc() const {
  TypeLoc Cur = *this;
  while (true) {
    switch (Cur.getTypeLocClass()) {
    default:
      break;
    case Qualified:
    case Elaborated:
      Cur = Cur.getNextTypeLoc();
      continue;
    }
    break;
  }
  return Cur.getLocalSourceRange().getEnd();
}


namespace {
  struct TSTChecker : public TypeLocVisitor<TSTChecker, bool> {
    // Overload resolution does the real work for us.
    static bool isTypeSpec(TypeSpecTypeLoc _) { return true; }
    static bool isTypeSpec(TypeLoc _) { return false; }

#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
    bool Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc) { \
      return isTypeSpec(TyLoc); \
    }
#include "clang/AST/TypeLocNodes.def"
  };
}


/// \brief Determines if the given type loc corresponds to a
/// TypeSpecTypeLoc.  Since there is not actually a TypeSpecType in
/// the type hierarchy, this is made somewhat complicated.
///
/// There are a lot of types that currently use TypeSpecTypeLoc
/// because it's a convenient base class.  Ideally we would not accept
/// those here, but ideally we would have better implementations for
/// them.
bool TypeSpecTypeLoc::classof(const TypeLoc *TL) {
  if (TL->getType().hasLocalQualifiers()) return false;
  return TSTChecker().Visit(*TL);
}

// Reimplemented to account for GNU/C++ extension
//     typeof unary-expression
// where there are no parentheses.
SourceRange TypeOfExprTypeLoc::getLocalSourceRange() const {
  if (getRParenLoc().isValid())
    return SourceRange(getTypeofLoc(), getRParenLoc());
  else
    return SourceRange(getTypeofLoc(),
                       getUnderlyingExpr()->getSourceRange().getEnd());
}


TypeSpecifierType BuiltinTypeLoc::getWrittenTypeSpec() const {
  if (needsExtraLocalData())
    return static_cast<TypeSpecifierType>(getWrittenBuiltinSpecs().Type);
  else {
    switch (getTypePtr()->getKind()) {
    case BuiltinType::Void:
      return TST_void;
    case BuiltinType::Bool:
      return TST_bool;
    case BuiltinType::Char_U:
    case BuiltinType::Char_S:
      return TST_char;
    case BuiltinType::Char16:
      return TST_char16;        
    case BuiltinType::Char32:
      return TST_char32;
    case BuiltinType::WChar:
      return TST_wchar;
    case BuiltinType::UndeducedAuto:
      return TST_auto;
        
    case BuiltinType::UChar:
    case BuiltinType::UShort:
    case BuiltinType::UInt:
    case BuiltinType::ULong:
    case BuiltinType::ULongLong:
    case BuiltinType::UInt128:
    case BuiltinType::SChar:
    case BuiltinType::Short:
    case BuiltinType::Int:
    case BuiltinType::Long:
    case BuiltinType::LongLong:
    case BuiltinType::Int128:
    case BuiltinType::Float:
    case BuiltinType::Double:
    case BuiltinType::LongDouble:
      llvm_unreachable("Builtin type needs extra local data!");
      // Fall through, if the impossible happens.
        
    case BuiltinType::NullPtr:
    case BuiltinType::Overload:
    case BuiltinType::Dependent:
    case BuiltinType::ObjCId:
    case BuiltinType::ObjCClass:
    case BuiltinType::ObjCSel:
      return TST_unspecified;
    }
  }
  
  return TST_unspecified;
}

TypeLoc TypeLoc::IgnoreParens() const {
  TypeLoc TL = *this;
  while (ParenTypeLoc* PTL = dyn_cast<ParenTypeLoc>(&TL))
    TL = PTL->getInnerLoc();
  return TL;
}

