//===--- Attr.h - Classes for representing attributes ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Attr interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ATTR_H
#define LLVM_CLANG_AST_ATTR_H

#include "clang/AST/AttrIterator.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/VersionTuple.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstring>

namespace clang {
  class ASTContext;
  class IdentifierInfo;
  class ObjCInterfaceDecl;
  class Expr;
  class QualType;
  class FunctionDecl;
  class TypeSourceInfo;

/// Attr - This represents one attribute.
class Attr {
private:
  SourceRange Range;
  unsigned AttrKind : 16;

protected:
  bool Inherited : 1;

  virtual ~Attr();
  
  void* operator new(size_t bytes) throw() {
    llvm_unreachable("Attrs cannot be allocated with regular 'new'.");
  }
  void operator delete(void* data) throw() {
    llvm_unreachable("Attrs cannot be released with regular 'delete'.");
  }

public:
  // Forward so that the regular new and delete do not hide global ones.
  void* operator new(size_t Bytes, ASTContext &C,
                     size_t Alignment = 16) throw() {
    return ::operator new(Bytes, C, Alignment);
  }
  void operator delete(void *Ptr, ASTContext &C,
                       size_t Alignment) throw() {
    return ::operator delete(Ptr, C, Alignment);
  }

protected:
  Attr(attr::Kind AK, SourceRange R)
    : Range(R), AttrKind(AK), Inherited(false) {}

public:

  attr::Kind getKind() const {
    return static_cast<attr::Kind>(AttrKind);
  }

  SourceLocation getLocation() const { return Range.getBegin(); }
  SourceRange getRange() const { return Range; }
  void setRange(SourceRange R) { Range = R; }

  bool isInherited() const { return Inherited; }

  // Clone this attribute.
  virtual Attr* clone(ASTContext &C) const = 0;

  virtual bool isLateParsed() const { return false; }

  // Pretty print this attribute.
  virtual void printPretty(llvm::raw_ostream &OS,
                           const PrintingPolicy &Policy) const = 0;
};

class InheritableAttr : public Attr {
  virtual void anchor();
protected:
  InheritableAttr(attr::Kind AK, SourceRange R)
    : Attr(AK, R) {}

public:
  void setInherited(bool I) { Inherited = I; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() <= attr::LAST_INHERITABLE;
  }
};

class InheritableParamAttr : public InheritableAttr {
  virtual void anchor();
protected:
  InheritableParamAttr(attr::Kind AK, SourceRange R)
    : InheritableAttr(AK, R) {}

public:
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() <= attr::LAST_INHERITABLE_PARAM;
  }
};

#include "clang/AST/Attrs.inc"

}  // end namespace clang

#endif
