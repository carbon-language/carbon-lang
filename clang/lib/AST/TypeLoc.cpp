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

#include "clang/AST/TypeLoc.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// TypeLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

/// \brief Return the source range for the visited TypeSpecLoc.
class TypeLocRanger : public TypeLocVisitor<TypeLocRanger, SourceRange> {
public:
#define ABSTRACT_TYPELOC(CLASS)
#define TYPELOC(CLASS, PARENT, TYPE) \
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

/// \brief Returns the size of type source info data block for the given type.
unsigned TypeLoc::getFullDataSizeForType(QualType Ty) {
  return TypeLoc(Ty, 0).getFullDataSize();
}

/// \brief Find the TypeSpecLoc that is part of this TypeLoc.
TypeSpecLoc TypeLoc::getTypeSpecLoc() const {
  if (isNull())
    return TypeSpecLoc();

  if (const DeclaratorLoc *DL = dyn_cast<DeclaratorLoc>(this))
    return DL->getTypeSpecLoc();
  return cast<TypeSpecLoc>(*this);
}

/// \brief Find the TypeSpecLoc that is part of this TypeLoc and return its
/// SourceRange.
SourceRange TypeLoc::getTypeSpecRange() const {
  return getTypeSpecLoc().getSourceRange();
}

namespace {

/// \brief Report the full source info data size for the visited TypeLoc.
class TypeSizer : public TypeLocVisitor<TypeSizer, unsigned> {
public:
#define ABSTRACT_TYPELOC(CLASS)
#define TYPELOC(CLASS, PARENT, TYPE) \
    unsigned Visit##CLASS(CLASS TyLoc) { return TyLoc.getFullDataSize(); }
#include "clang/AST/TypeLocNodes.def"

  unsigned VisitTypeLoc(TypeLoc TyLoc) {
    assert(0 && "A type loc wrapper was not handled!");
    return 0;
  }
};

}

/// \brief Returns the size of the type source info data block.
unsigned TypeLoc::getFullDataSize() const {
  if (isNull()) return 0;
  return TypeSizer().Visit(*this);
}

namespace {

/// \brief Return the "next" TypeLoc for the visited TypeLoc, e.g for "int*" the
/// TypeLoc is a PointerLoc and next TypeLoc is for "int".
class NextLoc : public TypeLocVisitor<NextLoc, TypeLoc> {
public:
#define TYPELOC(CLASS, PARENT, TYPE)
#define DECLARATOR_TYPELOC(CLASS, TYPE) \
    TypeLoc Visit##CLASS(CLASS TyLoc);
#include "clang/AST/TypeLocNodes.def"

  TypeLoc VisitTypeSpecLoc(TypeLoc TyLoc) { return TypeLoc(); }
  TypeLoc VisitObjCProtocolListLoc(ObjCProtocolListLoc TL);

  TypeLoc VisitTypeLoc(TypeLoc TyLoc) {
    assert(0 && "A declarator loc wrapper was not handled!");
    return TypeLoc();
  }
};

}

TypeLoc NextLoc::VisitObjCProtocolListLoc(ObjCProtocolListLoc TL) {
  return TL.getBaseTypeLoc();
}

TypeLoc NextLoc::VisitPointerLoc(PointerLoc TL) {
  return TL.getPointeeLoc();
}
TypeLoc NextLoc::VisitMemberPointerLoc(MemberPointerLoc TL) {
  return TL.getPointeeLoc();
}
TypeLoc NextLoc::VisitBlockPointerLoc(BlockPointerLoc TL) {
  return TL.getPointeeLoc();
}
TypeLoc NextLoc::VisitReferenceLoc(ReferenceLoc TL) {
  return TL.getPointeeLoc();
}
TypeLoc NextLoc::VisitFunctionLoc(FunctionLoc TL) {
  return TL.getResultLoc();
}
TypeLoc NextLoc::VisitArrayLoc(ArrayLoc TL) {
  return TL.getElementLoc();
}

/// \brief Get the next TypeLoc pointed by this TypeLoc, e.g for "int*" the
/// TypeLoc is a PointerLoc and next TypeLoc is for "int".
TypeLoc TypeLoc::getNextTypeLoc() const {
  return NextLoc().Visit(*this);
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

bool TypeSpecLoc::classof(const TypeLoc *TL) {
  return TypeSpecChecker().Visit(*TL);
}

//===----------------------------------------------------------------------===//
// DeclaratorLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

/// \brief Return the TypeSpecLoc for the visited DeclaratorLoc.
class TypeSpecGetter : public TypeLocVisitor<TypeSpecGetter, TypeSpecLoc> {
public:
#define TYPELOC(CLASS, PARENT, TYPE)
#define DECLARATOR_TYPELOC(CLASS, TYPE) \
    TypeSpecLoc Visit##CLASS(CLASS TyLoc) { return TyLoc.getTypeSpecLoc(); }
#include "clang/AST/TypeLocNodes.def"

  TypeSpecLoc VisitTypeLoc(TypeLoc TyLoc) {
    assert(0 && "A declarator loc wrapper was not handled!");
    return TypeSpecLoc();
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

bool DeclaratorLoc::classof(const TypeLoc *TL) {
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

bool DefaultTypeSpecLoc::classof(const TypeLoc *TL) {
  return DefaultTypeSpecLocChecker().Visit(*TL);
}

//===----------------------------------------------------------------------===//
// TypedefLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

class TypedefLocChecker : public TypeLocVisitor<TypedefLocChecker, bool> {
public:
  bool VisitTypedefLoc(TypedefLoc TyLoc) { return true; }
};

}

bool TypedefLoc::classof(const TypeLoc *TL) {
  return TypedefLocChecker().Visit(*TL);
}

//===----------------------------------------------------------------------===//
// ObjCProtocolListLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

class ObjCProtocolListLocChecker :
  public TypeLocVisitor<ObjCProtocolListLocChecker, bool> {
public:
  bool VisitObjCProtocolListLoc(ObjCProtocolListLoc TyLoc) { return true; }
};

}

bool ObjCProtocolListLoc::classof(const TypeLoc *TL) {
  return ObjCProtocolListLocChecker().Visit(*TL);
}

//===----------------------------------------------------------------------===//
// PointerLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

class PointerLocChecker : public TypeLocVisitor<PointerLocChecker, bool> {
public:
  bool VisitPointerLoc(PointerLoc TyLoc) { return true; }
};

}

bool PointerLoc::classof(const TypeLoc *TL) {
  return PointerLocChecker().Visit(*TL);
}

//===----------------------------------------------------------------------===//
// BlockPointerLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

class BlockPointerLocChecker :
           public TypeLocVisitor<BlockPointerLocChecker, bool> {
public:
  bool VisitBlockPointerLoc(BlockPointerLoc TyLoc) { return true; }
};

}

bool BlockPointerLoc::classof(const TypeLoc *TL) {
  return BlockPointerLocChecker().Visit(*TL);
}

//===----------------------------------------------------------------------===//
// MemberPointerLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

class MemberPointerLocChecker :
           public TypeLocVisitor<MemberPointerLocChecker, bool> {
public:
  bool VisitMemberPointerLoc(MemberPointerLoc TyLoc) { return true; }
};

}

bool MemberPointerLoc::classof(const TypeLoc *TL) {
  return MemberPointerLocChecker().Visit(*TL);
}

//===----------------------------------------------------------------------===//
// ReferenceLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

class ReferenceLocChecker : public TypeLocVisitor<ReferenceLocChecker, bool> {
public:
  bool VisitReferenceLoc(ReferenceLoc TyLoc) { return true; }
};

}

bool ReferenceLoc::classof(const TypeLoc *TL) {
  return ReferenceLocChecker().Visit(*TL);
}

//===----------------------------------------------------------------------===//
// FunctionLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

class FunctionLocChecker : public TypeLocVisitor<FunctionLocChecker, bool> {
public:
  bool VisitFunctionLoc(FunctionLoc TyLoc) { return true; }
};

}

bool FunctionLoc::classof(const TypeLoc *TL) {
  return FunctionLocChecker().Visit(*TL);
}

//===----------------------------------------------------------------------===//
// ArrayLoc Implementation
//===----------------------------------------------------------------------===//

namespace {

class ArrayLocChecker : public TypeLocVisitor<ArrayLocChecker, bool> {
public:
  bool VisitArrayLoc(ArrayLoc TyLoc) { return true; }
};

}

bool ArrayLoc::classof(const TypeLoc *TL) {
  return ArrayLocChecker().Visit(*TL);
}
