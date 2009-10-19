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
      return TyLoc.getSourceRange(); \
    }
#include "clang/AST/TypeLocNodes.def"
  };
}

SourceRange TypeLoc::getSourceRangeImpl(TypeLoc TL) {
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

namespace {
  struct TypeLocInitializer : public TypeLocVisitor<TypeLocInitializer> {
    SourceLocation Loc;
    TypeLocInitializer(SourceLocation Loc) : Loc(Loc) {}
  
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
    void Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc) { \
      TyLoc.initializeLocal(Loc); \
    }
#include "clang/AST/TypeLocNodes.def"
  };
}

/// \brief Initializes a type location, and all of its children
/// recursively, as if the entire tree had been written in the
/// given location.
void TypeLoc::initializeImpl(TypeLoc TL, SourceLocation Loc) {
  do {
    TypeLocInitializer(Loc).Visit(TL);
  } while ((TL = TL.getNextTypeLoc()));
}
