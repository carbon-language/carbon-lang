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

/// \brief Return the source range for the visited TypeSpecLoc.
class TypeLocRanger : public TypeLocVisitor<TypeLocRanger, SourceRange> {
public:
#define ABSTRACT_TYPELOC(CLASS)
#define TYPELOC(CLASS, PARENT) \
    SourceRange Visit##CLASS(CLASS TyLoc) { return TyLoc.getSourceRange(); }
#include "clang/AST/TypeLocNodes.def"

  SourceRange VisitTypeLoc(TypeLoc TyLoc) {
    assert(0 && "A typeloc wrapper was not handled!");
    return SourceRange();
  }
};

}

SourceRange TypeLoc::getSourceRange() const {
  if (isNull())
    return SourceRange();
  return TypeLocRanger().Visit(*this);
}

/// \brief Find the TypeSpecLoc that is part of this TypeLoc.
TypeSpecLoc TypeLoc::getTypeSpecLoc() const {
  if (isNull())
    return TypeSpecLoc();
  UnqualTypeLoc Cur = getUnqualifiedLoc();
  if (const DeclaratorLoc *DL = dyn_cast<DeclaratorLoc>(&Cur))
    return DL->getTypeSpecLoc();
  return cast<TypeSpecLoc>(Cur);
}

namespace {

/// \brief Report the full source info data size for the visited TypeLoc.
class TypeSizer : public TypeLocVisitor<TypeSizer, unsigned> {
public:
#define ABSTRACT_TYPELOC(CLASS)
#define TYPELOC(CLASS, PARENT) \
    unsigned Visit##CLASS(CLASS TyLoc) { return TyLoc.getFullDataSize(); }
#include "clang/AST/TypeLocNodes.def"

  unsigned VisitTypeLoc(TypeLoc TyLoc) {
    assert(0 && "A type loc wrapper was not handled!");
    return 0;
  }
};

}

/// \brief Returns the size of the type source info data block.
unsigned TypeLoc::getFullDataSizeForType(QualType Ty) {
  if (Ty.isNull()) return 0;
  return TypeSizer().Visit(TypeLoc(Ty, 0));
}

namespace {

/// \brief Return the "next" TypeLoc for the visited TypeLoc, e.g for "int*" the
/// TypeLoc is a PointerLoc and next TypeLoc is for "int".
class NextLoc : public TypeLocVisitor<NextLoc, TypeLoc> {
public:
#define TYPELOC(CLASS, PARENT)
#define DECLARATOR_TYPELOC(CLASS, TYPE) \
  TypeLoc Visit##CLASS(CLASS TyLoc);
#include "clang/AST/TypeLocNodes.def"

  TypeLoc VisitTypeSpecLoc(TypeLoc TyLoc) { return TypeLoc(); }
  TypeLoc VisitObjCProtocolListLoc(ObjCProtocolListLoc TL);
  TypeLoc VisitQualifiedLoc(QualifiedLoc TyLoc) {
    return TyLoc.getNextTypeLoc();
  }

  TypeLoc VisitTypeLoc(TypeLoc TyLoc) {
    assert(0 && "A declarator loc wrapper was not handled!");
    return TypeLoc();
  }
};

}

TypeLoc NextLoc::VisitObjCProtocolListLoc(ObjCProtocolListLoc TL) {
  return TL.getNextTypeLoc();
}

TypeLoc NextLoc::VisitPointerLoc(PointerLoc TL) {
  return TL.getNextTypeLoc();
}
TypeLoc NextLoc::VisitMemberPointerLoc(MemberPointerLoc TL) {
  return TL.getNextTypeLoc();
}
TypeLoc NextLoc::VisitBlockPointerLoc(BlockPointerLoc TL) {
  return TL.getNextTypeLoc();
}
TypeLoc NextLoc::VisitReferenceLoc(ReferenceLoc TL) {
  return TL.getNextTypeLoc();
}
TypeLoc NextLoc::VisitFunctionLoc(FunctionLoc TL) {
  return TL.getNextTypeLoc();
}
TypeLoc NextLoc::VisitArrayLoc(ArrayLoc TL) {
  return TL.getNextTypeLoc();
}

/// \brief Get the next TypeLoc pointed by this TypeLoc, e.g for "int*" the
/// TypeLoc is a PointerLoc and next TypeLoc is for "int".
TypeLoc TypeLoc::getNextTypeLoc() const {
  return NextLoc().Visit(*this);
}

namespace {
struct TypeLocInitializer : public TypeLocVisitor<TypeLocInitializer> {
  SourceLocation Loc;
  TypeLocInitializer(SourceLocation Loc) : Loc(Loc) {}
  
#define ABSTRACT_TYPELOC(CLASS)
#define TYPELOC(CLASS, PARENT) \
  void Visit##CLASS(CLASS TyLoc) { TyLoc.initializeLocal(Loc); }
#include "clang/AST/TypeLocNodes.def"
};
}

void TypeLoc::initializeImpl(TypeLoc TL, SourceLocation Loc) {
  do {
    TypeLocInitializer(Loc).Visit(TL);
  } while (TL = TL.getNextTypeLoc());
}

//===----------------------------------------------------------------------===//
// TypeSpecLoc Implementation
//===----------------------------------------------------------------------===//

namespace {
class TypeSpecChecker : public TypeLocVisitor<TypeSpecChecker, bool> {
public:
  bool VisitTypeSpecLoc(TypeSpecLoc TyLoc) { return true; }
};

}

bool TypeSpecLoc::classof(const UnqualTypeLoc *TL) {
  return TypeSpecChecker().Visit(*TL);
}

//===----------------------------------------------------------------------===//
// DeclaratorLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

/// \brief Return the TypeSpecLoc for the visited DeclaratorLoc.
class TypeSpecGetter : public TypeLocVisitor<TypeSpecGetter, TypeSpecLoc> {
public:
#define TYPELOC(CLASS, PARENT)
#define DECLARATOR_TYPELOC(CLASS, TYPE) \
    TypeSpecLoc Visit##CLASS(CLASS TyLoc) { return TyLoc.getTypeSpecLoc(); }
#include "clang/AST/TypeLocNodes.def"

  TypeSpecLoc VisitTypeLoc(TypeLoc TyLoc) {
    assert(0 && "A declarator loc wrapper was not handled!");
    return TypeSpecLoc();
  }

  TypeSpecLoc VisitQualifiedLoc(QualifiedLoc TyLoc) {
    return Visit(TyLoc.getUnqualifiedLoc());
  }
};

}

/// \brief Find the TypeSpecLoc that is part of this DeclaratorLoc.
TypeSpecLoc DeclaratorLoc::getTypeSpecLoc() const {
  return TypeSpecGetter().Visit(*this);
}

namespace {

class DeclaratorLocChecker : public TypeLocVisitor<DeclaratorLocChecker, bool> {
public:
  bool VisitDeclaratorLoc(DeclaratorLoc TyLoc) { return true; }
};

}

bool DeclaratorLoc::classof(const UnqualTypeLoc *TL) {
  return DeclaratorLocChecker().Visit(*TL);
}

//===----------------------------------------------------------------------===//
// DefaultTypeSpecLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

class DefaultTypeSpecLocChecker :
                        public TypeLocVisitor<DefaultTypeSpecLocChecker, bool> {
public:
  bool VisitDefaultTypeSpecLoc(DefaultTypeSpecLoc TyLoc) { return true; }
};

}

bool DefaultTypeSpecLoc::classofType(const Type *Ty) {
  return
    DefaultTypeSpecLocChecker().Visit(UnqualTypeLoc(const_cast<Type*>(Ty), 0));
}
 
