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

#include <cassert>
#include <cstring>
#include <string>
#include <algorithm>

namespace clang {
  class ASTContext;

/// Attr - This represents one attribute.
class Attr {
public:
  enum Kind {
    Alias,
    Aligned,
    AlwaysInline,
    Annotate,
    AsmLabel, // Represent GCC asm label extension.
    Blocks,
    Cleanup,
    Const,
    Constructor,
    DLLExport,
    DLLImport,
    Deprecated,
    Destructor,
    FastCall,    
    Format,
    IBOutletKind, // Clang-specific.  Use "Kind" suffix to not conflict with
    NoReturn,
    NoThrow,
    Nodebug,
    Noinline,
    NonNull,
    ObjCException,
    ObjCNSObject,
    Overloadable, // Clang-specific
    Packed,
    Pure,
    Section,
    StdCall,
    TransparentUnion,
    Unavailable,
    Unused,    
    Used,
    Visibility,
    WarnUnusedResult,
    Weak,
    WeakImport
  };
    
private:
  Attr *Next;
  Kind AttrKind;
  bool Inherited : 1;

protected:
  void* operator new(size_t bytes) throw() {
    assert(0 && "Attrs cannot be allocated with regular 'new'.");
    return 0;
  }
  void operator delete(void* data) throw() {
    assert(0 && "Attrs cannot be released with regular 'delete'.");
  }
  
protected:
  Attr(Kind AK) : Next(0), AttrKind(AK), Inherited(false) {}
  virtual ~Attr() {
    assert(Next == 0 && "Destroy didn't work");
  }
public:
  
  void Destroy(ASTContext &C);
  
  /// \brief Whether this attribute should be merged to new
  /// declarations.
  virtual bool isMerged() const { return true; }

  Kind getKind() const { return AttrKind; }

  Attr *getNext() { return Next; }
  const Attr *getNext() const { return Next; }
  void setNext(Attr *next) { Next = next; }

  bool isInherited() const { return Inherited; }
  void setInherited(bool value) { Inherited = value; }

  void addAttr(Attr *attr) {
    assert((attr != 0) && "addAttr(): attr is null");
    
    // FIXME: This doesn't preserve the order in any way.
    attr->Next = Next;
    Next = attr;
  }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *) { return true; }
};

class PackedAttr : public Attr {
  unsigned Alignment;

public:
  PackedAttr(unsigned alignment) : Attr(Packed), Alignment(alignment) {}

  /// getAlignment - The specified alignment in bits.
  unsigned getAlignment() const { return Alignment; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == Packed;
  }
  static bool classof(const PackedAttr *A) { return true; }
};
  
class AlignedAttr : public Attr {
  unsigned Alignment;
public:
  AlignedAttr(unsigned alignment) : Attr(Aligned), Alignment(alignment) {}

  /// getAlignment - The specified alignment in bits.
  unsigned getAlignment() const { return Alignment; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == Aligned;
  }
  static bool classof(const AlignedAttr *A) { return true; }
};

class AnnotateAttr : public Attr {
  std::string Annotation;
public:
  AnnotateAttr(const std::string &ann) : Attr(Annotate), Annotation(ann) {}
  
  const std::string& getAnnotation() const { return Annotation; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == Annotate;
  }
  static bool classof(const AnnotateAttr *A) { return true; }
};

class AsmLabelAttr : public Attr {
  std::string Label;
public:
  AsmLabelAttr(const std::string &L) : Attr(AsmLabel), Label(L) {}
  
  const std::string& getLabel() const { return Label; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == AsmLabel;
  }
  static bool classof(const AsmLabelAttr *A) { return true; }
};

class AlwaysInlineAttr : public Attr {
public:
  AlwaysInlineAttr() : Attr(AlwaysInline) {}

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == AlwaysInline; }
  static bool classof(const AlwaysInlineAttr *A) { return true; }
};

class AliasAttr : public Attr {
  std::string Aliasee;
public:
  AliasAttr(const std::string &aliasee) : Attr(Alias), Aliasee(aliasee) {}

  const std::string& getAliasee() const { return Aliasee; }

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == Alias; }
  static bool classof(const AliasAttr *A) { return true; }
};

class ConstructorAttr : public Attr {
  int priority;
public:
  ConstructorAttr(int p) : Attr(Constructor), priority(p) {}

  int getPriority() const { return priority; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == Constructor; }  
  static bool classof(const ConstructorAttr *A) { return true; }
};  
  
class DestructorAttr : public Attr {
  int priority;
public:
  DestructorAttr(int p) : Attr(Destructor), priority(p) {}

  int getPriority() const { return priority; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == Destructor; }  
  static bool classof(const DestructorAttr *A) { return true; }
};  
    
class IBOutletAttr : public Attr {
public:
  IBOutletAttr() : Attr(IBOutletKind) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == IBOutletKind;
  }
  static bool classof(const IBOutletAttr *A) { return true; }
};

class NoReturnAttr : public Attr {
public:
  NoReturnAttr() : Attr(NoReturn) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == NoReturn; }  
  static bool classof(const NoReturnAttr *A) { return true; }
};

class DeprecatedAttr : public Attr {
public:
  DeprecatedAttr() : Attr(Deprecated) {}

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == Deprecated; }
  static bool classof(const DeprecatedAttr *A) { return true; }
};

class SectionAttr : public Attr {
  std::string Name;
public:
  SectionAttr(const std::string &N) : Attr(Section), Name(N) {}
  
  const std::string& getName() const { return Name; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == Section;
  }
  static bool classof(const SectionAttr *A) { return true; }
};

class UnavailableAttr : public Attr {
public:
  UnavailableAttr() : Attr(Unavailable) {}

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == Unavailable; }
  static bool classof(const UnavailableAttr *A) { return true; }
};

class UnusedAttr : public Attr {
public:
  UnusedAttr() : Attr(Unused) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == Unused; }  
  static bool classof(const UnusedAttr *A) { return true; }
};  
  
class UsedAttr : public Attr {
public:
  UsedAttr() : Attr(Used) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == Used; }  
  static bool classof(const UsedAttr *A) { return true; }
};  
  
class WeakAttr : public Attr {
public:
  WeakAttr() : Attr(Weak) {}

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == Weak; }
  static bool classof(const WeakAttr *A) { return true; }
};

class WeakImportAttr : public Attr {
public:
  WeakImportAttr() : Attr(WeakImport) {}

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == WeakImport; }
  static bool classof(const WeakImportAttr *A) { return true; }
};

class NoThrowAttr : public Attr {
public:
  NoThrowAttr() : Attr(NoThrow) {}

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == NoThrow; }
  static bool classof(const NoThrowAttr *A) { return true; }
};

class ConstAttr : public Attr {
public:
  ConstAttr() : Attr(Const) {}

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == Const; }
  static bool classof(const ConstAttr *A) { return true; }
};

class PureAttr : public Attr {
public:
  PureAttr() : Attr(Pure) {}

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == Pure; }
  static bool classof(const PureAttr *A) { return true; }
};

class NonNullAttr : public Attr {
  unsigned* ArgNums;
  unsigned Size;
public:
  NonNullAttr(unsigned* arg_nums = 0, unsigned size = 0) : Attr(NonNull),
    ArgNums(0), Size(0) {
  
    if (size == 0) return;
    assert(arg_nums);
    ArgNums = new unsigned[size];
    Size = size;
    memcpy(ArgNums, arg_nums, sizeof(*ArgNums)*size);
  }
  
  virtual ~NonNullAttr() {
    delete [] ArgNums;
  }
  
  bool isNonNull(unsigned arg) const {
    return ArgNums ? std::binary_search(ArgNums, ArgNums+Size, arg) : true;
  }  
  
  static bool classof(const Attr *A) { return A->getKind() == NonNull; }
  static bool classof(const NonNullAttr *A) { return true; }
};

class FormatAttr : public Attr {
  std::string Type;
  int formatIdx, firstArg;
public:
  FormatAttr(const std::string &type, int idx, int first) : Attr(Format),
             Type(type), formatIdx(idx), firstArg(first) {}

  const std::string& getType() const { return Type; }
  void setType(const std::string &type) { Type = type; }
  int getFormatIdx() const { return formatIdx; }
  int getFirstArg() const { return firstArg; }

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == Format; }
  static bool classof(const FormatAttr *A) { return true; }
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
public:
  VisibilityAttr(VisibilityTypes v) : Attr(Visibility),
                 VisibilityType(v) {}

  VisibilityTypes getVisibility() const { return VisibilityType; }

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == Visibility; }
  static bool classof(const VisibilityAttr *A) { return true; }
};

class DLLImportAttr : public Attr {
public:
  DLLImportAttr() : Attr(DLLImport) {}

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == DLLImport; }
  static bool classof(const DLLImportAttr *A) { return true; }
};

class DLLExportAttr : public Attr {
public:
  DLLExportAttr() : Attr(DLLExport) {}

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == DLLExport; }
  static bool classof(const DLLExportAttr *A) { return true; }
};

class FastCallAttr : public Attr {
public:
  FastCallAttr() : Attr(FastCall) {}

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == FastCall; }
  static bool classof(const FastCallAttr *A) { return true; }
};

class StdCallAttr : public Attr {
public:
  StdCallAttr() : Attr(StdCall) {}

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == StdCall; }
  static bool classof(const StdCallAttr *A) { return true; }
};

class TransparentUnionAttr : public Attr {
public:
  TransparentUnionAttr() : Attr(TransparentUnion) {}

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == TransparentUnion; }
  static bool classof(const TransparentUnionAttr *A) { return true; }
};

class ObjCNSObjectAttr : public Attr {
// Implement isa/cast/dyncast/etc.
public:
  ObjCNSObjectAttr() : Attr(ObjCNSObject) {}
  
static bool classof(const Attr *A) { return A->getKind() == ObjCNSObject; }
static bool classof(const ObjCNSObjectAttr *A) { return true; }
};
  
  
class ObjCExceptionAttr : public Attr {
public:
  ObjCExceptionAttr() : Attr(ObjCException) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == ObjCException; }
  static bool classof(const ObjCExceptionAttr *A) { return true; }
};
  
  
class OverloadableAttr : public Attr {
public:
  OverloadableAttr() : Attr(Overloadable) { }

  virtual bool isMerged() const { return false; }

  static bool classof(const Attr *A) { return A->getKind() == Overloadable; }
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
  BlocksAttr(BlocksAttrTypes t) : Attr(Blocks), BlocksAttrType(t) {}

  BlocksAttrTypes getType() const { return BlocksAttrType; }

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == Blocks; }
  static bool classof(const BlocksAttr *A) { return true; }
};

class FunctionDecl;
  
class CleanupAttr : public Attr {
  FunctionDecl *FD;
  
public:
  CleanupAttr(FunctionDecl *fd) : Attr(Cleanup), FD(fd) {}

  const FunctionDecl *getFunctionDecl() const { return FD; }
  
  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == Cleanup; }
  static bool classof(const CleanupAttr *A) { return true; }
};

class NodebugAttr : public Attr {
public:
  NodebugAttr() : Attr(Nodebug) {}
    
  // Implement isa/cast/dyncast/etc.
    
  static bool classof(const Attr *A) { return A->getKind() == Nodebug; }
  static bool classof(const NodebugAttr *A) { return true; }
};
  
class WarnUnusedResultAttr : public Attr {
public:
  WarnUnusedResultAttr() : Attr(WarnUnusedResult) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == WarnUnusedResult;}
  static bool classof(const WarnUnusedResultAttr *A) { return true; }
};

class NoinlineAttr : public Attr {
public:
  NoinlineAttr() : Attr(Noinline) {}
    
  // Implement isa/cast/dyncast/etc.
    
  static bool classof(const Attr *A) { return A->getKind() == Noinline; }
  static bool classof(const NoinlineAttr *A) { return true; }
};

}  // end namespace clang

#endif
