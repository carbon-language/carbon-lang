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

/// Attr - This represents one attribute.
class Attr {
public:
  enum Kind {
    Alias,
    Aligned,
    Annotate,
    AsmLabel, // Represent GCC asm label extension.
    Constructor,
    Deprecated,
    Destructor,
    DLLImport,
    DLLExport,
    FastCall,    
    Format,
    IBOutletKind, // Clang-specific.  Use "Kind" suffix to not conflict with
    NonNull,
    NoReturn,
    NoThrow,
    ObjCGC,
    Packed,
    StdCall,
    TransparentUnion,
    Unused,    
    Visibility,
    Weak,
    Blocks,
    Const,
    Pure
  };
    
private:
  Attr *Next;
  Kind AttrKind;
  
protected:
  Attr(Kind AK) : Next(0), AttrKind(AK) {}
public:
  virtual ~Attr() {
    delete Next;
  }

  Kind getKind() const { return AttrKind; }

  Attr *getNext() { return Next; }
  const Attr *getNext() const { return Next; }
  void setNext(Attr *next) { Next = next; }
  
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
public:
  PackedAttr() : Attr(Packed) {}
  
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

class UnusedAttr : public Attr {
public:
  UnusedAttr() : Attr(Unused) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) { return A->getKind() == Unused; }  
  static bool classof(const UnusedAttr *A) { return true; }
};  
  
class WeakAttr : public Attr {
public:
  WeakAttr() : Attr(Weak) {}

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == Weak; }
  static bool classof(const WeakAttr *A) { return true; }
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
  
    if (size) {
      assert (arg_nums);
      ArgNums = new unsigned[size];
      Size = size;
      memcpy(ArgNums, arg_nums, sizeof(*ArgNums)*size);
    }
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

class ObjCGCAttr : public Attr {
public:
  enum GCAttrTypes {
    Weak = 0,
    Strong
  };
private:
  GCAttrTypes GCAttrType;
public:
  ObjCGCAttr(GCAttrTypes t) : Attr(ObjCGC), GCAttrType(t) {}

  GCAttrTypes getType() const { return GCAttrType; }

  // Implement isa/cast/dyncast/etc.

  static bool classof(const Attr *A) { return A->getKind() == ObjCGC; }
  static bool classof(const ObjCGCAttr *A) { return true; }
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
  static bool classof(const ObjCGCAttr *A) { return true; }
};

}  // end namespace clang

#endif
