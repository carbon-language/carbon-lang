//===--- Type.h - C Language Family Type Representation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Type interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TYPE_H
#define LLVM_CLANG_AST_TYPE_H

#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {
namespace clang {
  class TypeDecl;
  class Type;
  
/// TypeRef - For efficiency, we don't store CVR-qualified types as nodes on
/// their own: instead each reference to a type stores the qualifiers.  This
/// greatly reduces the number of nodes we need to allocate for types (for
/// example we only need one for 'int', 'const int', 'volatile int',
/// 'const volatile int', etc).
///
/// As an added efficiency bonus, instead of making this a pair, we just store
/// the three bits we care about in the low bits of the pointer.  To handle the
/// packing/unpacking, we make TypeRef be a simple wrapper class that acts like
/// a smart pointer.
class TypeRef {
  uintptr_t ThePtr;
public:
  enum {
    Const    = 0x1,
    Volatile = 0x2,
    Restrict = 0x4,
    CVRFlags = Const|Volatile|Restrict
  };
  
  TypeRef() : ThePtr(0) {}
  
  TypeRef(Type *Ptr, unsigned Quals = 0) {
    assert((Quals & ~CVRFlags) == 0 && "Invalid type qualifiers!");
    ThePtr = reinterpret_cast<uintptr_t>(Ptr);
    assert((ThePtr & CVRFlags) == 0 && "Type pointer not 8-byte aligned?");
    ThePtr |= Quals;
  }
  
  Type &operator*() const {
    return *reinterpret_cast<Type*>(ThePtr & ~CVRFlags);
  }

  Type *operator->() const {
    return reinterpret_cast<Type*>(ThePtr & ~CVRFlags);
  }
  
  /// isNull - Return true if this TypeRef doesn't point to a type yet.
  bool isNull() const {
    return ThePtr == 0;
  }

  bool isConstQualified() const {
    return ThePtr & Const;
  }
  bool isVolatileQualified() const {
    return ThePtr & Volatile;
  }
  bool isRestrictQualified() const {
    return ThePtr & Restrict;
  }
  unsigned getQualifiers() const {
    return ThePtr & CVRFlags;
  }
  
  void dump() const;
};


/// Type - This is the base class of the type hierarchy.  A central concept
/// with types is that each type always has a canonical type.  A canonical type
/// is the type with any typedef names stripped out of it or the types it
/// references.  For example, consider:
///
///  typedef int  foo;
///  typedef foo* bar;
///    'int *'    'foo *'    'bar'
///
/// There will be a Type object created for 'int'.  Since int is canonical, its
/// canonicaltype pointer points to itself.  There is also a Type for 'foo' (a
/// TypedefType).  Its CanonicalType pointer points to the 'int' Type.  Next
/// there is a PointerType that represents 'int*', which, like 'int', is
/// canonical.  Finally, there is a PointerType type for 'foo*' whose canonical
/// type is 'int*', and there is a TypedefType for 'bar', whose canonical type
/// is also 'int*'.
///
/// Non-canonical types are useful for emitting diagnostics, without losing
/// information about typedefs being used.  Canonical types are useful for type
/// comparisons (they allow by-pointer equality tests) and useful for reasoning
/// about whether something has a particular form (e.g. is a function type),
/// because they implicitly, recursively, strip all typedefs out of a type.
///
/// Types, once created, are immutable.
///
class Type {
  Type *CanonicalType;
public:
  virtual ~Type();
  
  bool isCanonical() const { return CanonicalType == this; }
  Type *getCanonicalType() const { return CanonicalType; }
  
  virtual void dump() const = 0;
};

class PointerType : public Type {
  TypeRef PointeeType;
public:
  
};

class TypedefType : public Type {
  // Decl * here.
public:
  
};


/// ...

// TODO: When we support C++, we should have types for uses of template with
// default parameters.  We should be able to distinguish source use of
// 'std::vector<int>' from 'std::vector<int, std::allocator<int> >'. Though they
// specify the same type, we want to print the default argument only if
// specified in the source code.

  
}  // end namespace clang
}  // end namespace llvm

#endif
