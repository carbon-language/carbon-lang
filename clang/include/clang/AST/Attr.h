//===--- Attr.h - Classes for representing expressions ----------*- C++ -*-===//
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

#include "llvm/Support/Casting.h"
#include "llvm/ADT/StringRef.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/AST/Type.h"
#include <cassert>
#include <cstring>
#include <algorithm>
using llvm::dyn_cast;

namespace clang {
  class ASTContext;
  class IdentifierInfo;
  class ObjCInterfaceDecl;
  class Expr;
  class QualType;
}

// Defined in ASTContext.h
void *operator new(size_t Bytes, clang::ASTContext &C,
                   size_t Alignment = 16) throw ();

// It is good practice to pair new/delete operators.  Also, MSVC gives many
// warnings if a matching delete overload is not declared, even though the
// throw() spec guarantees it will not be implicitly called.
void operator delete(void *Ptr, clang::ASTContext &C, size_t)
              throw ();

namespace clang {

/// Attr - This represents one attribute.
class Attr {
private:
  Attr *Next;
  attr::Kind AttrKind;
  bool Inherited : 1;

protected:
  virtual ~Attr();
  
  void* operator new(size_t bytes) throw() {
    assert(0 && "Attrs cannot be allocated with regular 'new'.");
    return 0;
  }
  void operator delete(void* data) throw() {
    assert(0 && "Attrs cannot be released with regular 'delete'.");
  }

protected:
  Attr(attr::Kind AK) : Next(0), AttrKind(AK), Inherited(false) {}
  
public:

  /// \brief Whether this attribute should be merged to new
  /// declarations.
  virtual bool isMerged() const { return true; }

  attr::Kind getKind() const { return AttrKind; }

  Attr *getNext() { return Next; }
  const Attr *getNext() const { return Next; }
  void setNext(Attr *next) { Next = next; }

  template<typename T> const T *getNext() const {
    for (const Attr *attr = getNext(); attr; attr = attr->getNext())
      if (const T *V = dyn_cast<T>(attr))
        return V;
    return 0;
  }

  bool isInherited() const { return Inherited; }
  void setInherited(bool value) { Inherited = value; }

  void addAttr(Attr *attr) {
    assert((attr != 0) && "addAttr(): attr is null");

    // FIXME: This doesn't preserve the order in any way.
    attr->Next = Next;
    Next = attr;
  }

  // Clone this attribute.
  virtual Attr* clone(ASTContext &C) const = 0;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *) { return true; }
};

#include "clang/AST/Attrs.inc"
  
class AttrWithString : public Attr {
private:
  const char *Str;
  unsigned StrLen;
protected:
  AttrWithString(attr::Kind AK, ASTContext &C, llvm::StringRef s);
  llvm::StringRef getString() const { return llvm::StringRef(Str, StrLen); }
  void ReplaceString(ASTContext &C, llvm::StringRef newS);
};

#define DEF_SIMPLE_ATTR(ATTR)                                           \
class ATTR##Attr : public Attr {                                        \
public:                                                                 \
  ATTR##Attr() : Attr(attr::ATTR) {}                                          \
  virtual Attr *clone(ASTContext &C) const;                             \
  static bool classof(const Attr *A) { return A->getKind() == attr::ATTR; }   \
  static bool classof(const ATTR##Attr *A) { return true; }             \
}

DEF_SIMPLE_ATTR(Packed);

/// \brief Attribute for specifying a maximum field alignment; this is only
/// valid on record decls.
class MaxFieldAlignmentAttr : public Attr {
  unsigned Alignment;

public:
  MaxFieldAlignmentAttr(unsigned alignment)
    : Attr(attr::MaxFieldAlignment), Alignment(alignment) {}

  /// getAlignment - The specified alignment in bits.
  unsigned getAlignment() const { return Alignment; }

  virtual Attr* clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == attr::MaxFieldAlignment;
  }
  static bool classof(const MaxFieldAlignmentAttr *A) { return true; }
};

DEF_SIMPLE_ATTR(AlignMac68k);

/// \brief Atribute for specifying the alignment of a variable or type.
///
/// This node will either contain the precise Alignment (in bits, not bytes!)
/// or will contain the expression for the alignment attribute in the case of
/// a dependent expression within a class or function template. At template
/// instantiation time these are transformed into concrete attributes.
class AlignedAttr : public Attr {
  unsigned Alignment;
  Expr *AlignmentExpr;
public:
  AlignedAttr(unsigned alignment)
    : Attr(attr::Aligned), Alignment(alignment), AlignmentExpr(0) {}
  AlignedAttr(Expr *E)
    : Attr(attr::Aligned), Alignment(0), AlignmentExpr(E) {}

  /// getAlignmentExpr - Get a dependent alignment expression if one is present.
  Expr *getAlignmentExpr() const {
    return AlignmentExpr;
  }

  /// isDependent - Is the alignment a dependent expression
  bool isDependent() const {
    return getAlignmentExpr();
  }

  /// getAlignment - The specified alignment in bits. Requires !isDependent().
  unsigned getAlignment() const {
    assert(!isDependent() && "Cannot get a value dependent alignment");
    return Alignment;
  }

  /// getMaxAlignment - Get the maximum alignment of attributes on this list.
  unsigned getMaxAlignment() const {
    const AlignedAttr *Next = getNext<AlignedAttr>();
    if (Next)
      return std::max(Next->getMaxAlignment(), getAlignment());
    else
      return getAlignment();
  }

  virtual Attr* clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == attr::Aligned;
  }
  static bool classof(const AlignedAttr *A) { return true; }
};

class AnnotateAttr : public AttrWithString {
public:
  AnnotateAttr(ASTContext &C, llvm::StringRef ann)
    : AttrWithString(attr::Annotate, C, ann) {}

  llvm::StringRef getAnnotation() const { return getString(); }

  virtual Attr* clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == attr::Annotate;
  }
  static bool classof(const AnnotateAttr *A) { return true; }
};

class AsmLabelAttr : public AttrWithString {
public:
  AsmLabelAttr(ASTContext &C, llvm::StringRef L)
    : AttrWithString(attr::AsmLabel, C, L) {}

  llvm::StringRef getLabel() const { return getString(); }

  virtual Attr* clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == attr::AsmLabel;
  }
  static bool classof(const AsmLabelAttr *A) { return true; }
};

DEF_SIMPLE_ATTR(AlwaysInline);

class AliasAttr : public AttrWithString {
public:
  AliasAttr(ASTContext &C, llvm::StringRef aliasee)
    : AttrWithString(attr::Alias, C, aliasee) {}

  llvm::StringRef getAliasee() const { return getString(); }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == attr::Alias; }
  static bool classof(const AliasAttr *A) { return true; }
};

class ConstructorAttr : public Attr {
  int priority;
public:
  ConstructorAttr(int p) : Attr(attr::Constructor), priority(p) {}

  int getPriority() const { return priority; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A)
    { return A->getKind() == attr::Constructor; }
  static bool classof(const ConstructorAttr *A) { return true; }
};

class DestructorAttr : public Attr {
  int priority;
public:
  DestructorAttr(int p) : Attr(attr::Destructor), priority(p) {}

  int getPriority() const { return priority; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A)
    { return A->getKind() == attr::Destructor; }
  static bool classof(const DestructorAttr *A) { return true; }
};

class IBOutletAttr : public Attr {
public:
  IBOutletAttr() : Attr(attr::IBOutlet) {}

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == attr::IBOutlet;
  }
  static bool classof(const IBOutletAttr *A) { return true; }
};

class IBOutletCollectionAttr : public Attr {
  QualType QT;
public:
  IBOutletCollectionAttr(QualType qt = QualType())
    : Attr(attr::IBOutletCollection), QT(qt) {}

  QualType getType() const { return QT; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == attr::IBOutletCollection;
  }
  static bool classof(const IBOutletCollectionAttr *A) { return true; }
};

class IBActionAttr : public Attr {
public:
  IBActionAttr() : Attr(attr::IBAction) {}

  virtual Attr *clone(ASTContext &C) const;

    // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == attr::IBAction;
  }
  static bool classof(const IBActionAttr *A) { return true; }
};

DEF_SIMPLE_ATTR(AnalyzerNoReturn);
DEF_SIMPLE_ATTR(Deprecated);
DEF_SIMPLE_ATTR(GNUInline);
DEF_SIMPLE_ATTR(Malloc);
DEF_SIMPLE_ATTR(NoReturn);
DEF_SIMPLE_ATTR(NoInstrumentFunction);

class SectionAttr : public AttrWithString {
public:
  SectionAttr(ASTContext &C, llvm::StringRef N)
    : AttrWithString(attr::Section, C, N) {}

  llvm::StringRef getName() const { return getString(); }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == attr::Section;
  }
  static bool classof(const SectionAttr *A) { return true; }
};

DEF_SIMPLE_ATTR(Unavailable);
DEF_SIMPLE_ATTR(Unused);
DEF_SIMPLE_ATTR(Used);
DEF_SIMPLE_ATTR(Weak);
DEF_SIMPLE_ATTR(WeakImport);
DEF_SIMPLE_ATTR(WeakRef);
DEF_SIMPLE_ATTR(NoThrow);
DEF_SIMPLE_ATTR(Const);
DEF_SIMPLE_ATTR(Pure);

class NonNullAttr : public Attr {
  unsigned* ArgNums;
  unsigned Size;
public:
  NonNullAttr(ASTContext &C, unsigned* arg_nums = 0, unsigned size = 0);

  typedef const unsigned *iterator;
  iterator begin() const { return ArgNums; }
  iterator end() const { return ArgNums + Size; }
  unsigned size() const { return Size; }

  bool isNonNull(unsigned arg) const {
    return ArgNums ? std::binary_search(ArgNums, ArgNums+Size, arg) : true;
  }

  virtual Attr *clone(ASTContext &C) const;

  static bool classof(const Attr *A) { return A->getKind() == attr::NonNull; }
  static bool classof(const NonNullAttr *A) { return true; }
};

/// OwnershipAttr
/// Ownership attributes are used to annotate pointers that own a resource
/// in order for the analyzer to check correct allocation and deallocation.
/// There are three attributes, ownership_returns, ownership_holds and
/// ownership_takes, represented by subclasses of OwnershipAttr
class OwnershipAttr: public AttrWithString {
 protected:
  unsigned* ArgNums;
  unsigned Size;
public:
  attr::Kind AKind;
public:
  OwnershipAttr(attr::Kind AK, ASTContext &C, unsigned* arg_nums, unsigned size,
                llvm::StringRef module);


  virtual void Destroy(ASTContext &C);

  /// Ownership attributes have a 'module', which is the name of a kind of
  /// resource that can be checked.
  /// The Malloc checker uses the module 'malloc'.
  llvm::StringRef getModule() const {
    return getString();
  }
  void setModule(ASTContext &C, llvm::StringRef module) {
    ReplaceString(C, module);
  }
  bool isModule(const char *m) const {
    return getModule().equals(m);
  }

  typedef const unsigned *iterator;
  iterator begin() const {
    return ArgNums;
  }
  iterator end() const {
    return ArgNums + Size;
  }
  unsigned size() const {
    return Size;
  }

  virtual Attr *clone(ASTContext &C) const;

  static bool classof(const Attr *A) {
    switch (A->getKind()) {
    case attr::OwnershipTakes:
    case attr::OwnershipHolds:
    case attr::OwnershipReturns:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const OwnershipAttr *A) {
    return true;
  }
};

class OwnershipTakesAttr: public OwnershipAttr {
public:
  OwnershipTakesAttr(ASTContext &C, unsigned* arg_nums, unsigned size,
                     llvm::StringRef module);

  virtual Attr *clone(ASTContext &C) const;

  static bool classof(const Attr *A) {
    return A->getKind() == attr::OwnershipTakes;
  }
  static bool classof(const OwnershipTakesAttr *A) {
    return true;
  }
};

class OwnershipHoldsAttr: public OwnershipAttr {
public:
  OwnershipHoldsAttr(ASTContext &C, unsigned* arg_nums, unsigned size,
                     llvm::StringRef module);

  virtual Attr *clone(ASTContext &C) const;

  static bool classof(const Attr *A) {
    return A->getKind() == attr::OwnershipHolds;
  }
  static bool classof(const OwnershipHoldsAttr *A) {
    return true;
  }
};

class OwnershipReturnsAttr: public OwnershipAttr {
public:
  OwnershipReturnsAttr(ASTContext &C, unsigned* arg_nums, unsigned size,
                     llvm::StringRef module);

  virtual Attr *clone(ASTContext &C) const;

  static bool classof(const Attr *A) {
    return A->getKind() == attr::OwnershipReturns;
  }
  static bool classof(const OwnershipReturnsAttr *A) {
    return true;
  }
};

class FormatAttr : public AttrWithString {
  int formatIdx, firstArg;
public:
  FormatAttr(ASTContext &C, llvm::StringRef type, int idx, int first)
    : AttrWithString(attr::Format, C, type), formatIdx(idx), firstArg(first) {}

  llvm::StringRef getType() const { return getString(); }
  void setType(ASTContext &C, llvm::StringRef type);
  int getFormatIdx() const { return formatIdx; }
  int getFirstArg() const { return firstArg; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == attr::Format; }
  static bool classof(const FormatAttr *A) { return true; }
};

class FormatArgAttr : public Attr {
  int formatIdx;
public:
  FormatArgAttr(int idx) : Attr(attr::FormatArg), formatIdx(idx) {}
  int getFormatIdx() const { return formatIdx; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == attr::FormatArg; }
  static bool classof(const FormatArgAttr *A) { return true; }
};

class SentinelAttr : public Attr {
  int sentinel, NullPos;
public:
  SentinelAttr(int sentinel_val, int nullPos) : Attr(attr::Sentinel),
               sentinel(sentinel_val), NullPos(nullPos) {}
  int getSentinel() const { return sentinel; }
  int getNullPos() const { return NullPos; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == attr::Sentinel; }
  static bool classof(const SentinelAttr *A) { return true; }
};

class VisibilityAttr : public Attr {
public:
  /// @brief An enumeration for the kinds of visibility of symbols.
  enum VisibilityTypes {
    DefaultVisibility = 0,
    HiddenVisibility,
    ProtectedVisibility
  };
private:
  VisibilityTypes VisibilityType;
  bool FromPragma;
public:
  VisibilityAttr(VisibilityTypes v, bool fp) : Attr(attr::Visibility),
                 VisibilityType(v), FromPragma(fp) {}

  VisibilityTypes getVisibility() const { return VisibilityType; }

  bool isFromPragma() const { return FromPragma; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A)
    { return A->getKind() == attr::Visibility; }
  static bool classof(const VisibilityAttr *A) { return true; }
};

DEF_SIMPLE_ATTR(FastCall);
DEF_SIMPLE_ATTR(StdCall);
DEF_SIMPLE_ATTR(ThisCall);
DEF_SIMPLE_ATTR(CDecl);
DEF_SIMPLE_ATTR(TransparentUnion);
DEF_SIMPLE_ATTR(ObjCNSObject);
DEF_SIMPLE_ATTR(ObjCException);

class OverloadableAttr : public Attr {
public:
  OverloadableAttr() : Attr(attr::Overloadable) { }

  virtual bool isMerged() const { return false; }

  virtual Attr *clone(ASTContext &C) const;

  static bool classof(const Attr *A)
    { return A->getKind() == attr::Overloadable; }
  static bool classof(const OverloadableAttr *) { return true; }
};

class BlocksAttr : public Attr {
public:
  enum BlocksAttrTypes {
    ByRef = 0
  };
private:
  BlocksAttrTypes BlocksAttrType;
public:
  BlocksAttr(BlocksAttrTypes t) : Attr(attr::Blocks), BlocksAttrType(t) {}

  BlocksAttrTypes getType() const { return BlocksAttrType; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == attr::Blocks; }
  static bool classof(const BlocksAttr *A) { return true; }
};

class FunctionDecl;

class CleanupAttr : public Attr {
  FunctionDecl *FD;

public:
  CleanupAttr(FunctionDecl *fd) : Attr(attr::Cleanup), FD(fd) {}

  const FunctionDecl *getFunctionDecl() const { return FD; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == attr::Cleanup; }
  static bool classof(const CleanupAttr *A) { return true; }
};

DEF_SIMPLE_ATTR(NoDebug);
DEF_SIMPLE_ATTR(WarnUnusedResult);
DEF_SIMPLE_ATTR(NoInline);

class RegparmAttr : public Attr {
  unsigned NumParams;

public:
  RegparmAttr(unsigned np) : Attr(attr::Regparm), NumParams(np) {}

  unsigned getNumParams() const { return NumParams; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == attr::Regparm; }
  static bool classof(const RegparmAttr *A) { return true; }
};

class ReqdWorkGroupSizeAttr : public Attr {
  unsigned X, Y, Z;
public:
  ReqdWorkGroupSizeAttr(unsigned X, unsigned Y, unsigned Z)
  : Attr(attr::ReqdWorkGroupSize), X(X), Y(Y), Z(Z) {}

  unsigned getXDim() const { return X; }
  unsigned getYDim() const { return Y; }
  unsigned getZDim() const { return Z; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == attr::ReqdWorkGroupSize;
  }
  static bool classof(const ReqdWorkGroupSizeAttr *A) { return true; }
};

class InitPriorityAttr : public Attr {
  unsigned Priority;
public:
  InitPriorityAttr(unsigned priority) 
    : Attr(attr::InitPriority),  Priority(priority) {}
    
  unsigned getPriority() const { return Priority; }
    
  virtual Attr *clone(ASTContext &C) const;
    
  static bool classof(const Attr *A) 
    { return A->getKind() == attr::InitPriority; }
  static bool classof(const InitPriorityAttr *A) { return true; }
};
  
// Checker-specific attributes.
DEF_SIMPLE_ATTR(CFReturnsNotRetained);
DEF_SIMPLE_ATTR(CFReturnsRetained);
DEF_SIMPLE_ATTR(NSReturnsNotRetained);
DEF_SIMPLE_ATTR(NSReturnsRetained);

// Target-specific attributes
DEF_SIMPLE_ATTR(DLLImport);
DEF_SIMPLE_ATTR(DLLExport);

class MSP430InterruptAttr : public Attr {
  unsigned Number;

public:
  MSP430InterruptAttr(unsigned n) : Attr(attr::MSP430Interrupt), Number(n) {}

  unsigned getNumber() const { return Number; }

  virtual Attr *clone(ASTContext &C) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A)
    { return A->getKind() == attr::MSP430Interrupt; }
  static bool classof(const MSP430InterruptAttr *A) { return true; }
};

DEF_SIMPLE_ATTR(X86ForceAlignArgPointer);

#undef DEF_SIMPLE_ATTR

}  // end namespace clang

#endif
