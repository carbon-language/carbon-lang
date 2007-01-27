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

#include "llvm/Support/Casting.h"
#include "llvm/ADT/FoldingSet.h"

namespace llvm {
namespace clang {
  class ASTContext;
  class Type;
  class TypedefDecl;
  class TagDecl;
  class RecordDecl;
  class EnumDecl;
  
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
  enum TQ {   // NOTE: These flags must be kept in sync with DeclSpec::TQ.
    Const    = 0x1,
    Restrict = 0x2,
    Volatile = 0x4,
    CVRFlags = Const|Restrict|Volatile
  };
  
  TypeRef() : ThePtr(0) {}
  
  TypeRef(Type *Ptr, unsigned Quals = 0) {
    assert((Quals & ~CVRFlags) == 0 && "Invalid type qualifiers!");
    ThePtr = reinterpret_cast<uintptr_t>(Ptr);
    assert((ThePtr & CVRFlags) == 0 && "Type pointer not 8-byte aligned?");
    ThePtr |= Quals;
  }

  static TypeRef getFromOpaquePtr(void *Ptr) {
    TypeRef T;
    T.ThePtr = reinterpret_cast<uintptr_t>(Ptr);
    return T;
  }
  
  unsigned getQualifiers() const {
    return ThePtr & CVRFlags;
  }
  Type *getTypePtr() const {
    return reinterpret_cast<Type*>(ThePtr & ~CVRFlags);
  }
  
  void *getAsOpaquePtr() const {
    return reinterpret_cast<void*>(ThePtr);
  }
  
  Type &operator*() const {
    return *getTypePtr();
  }

  Type *operator->() const {
    return getTypePtr();
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
  
  TypeRef getQualifiedType(unsigned TQs) const {
    return TypeRef(getTypePtr(), TQs);
  }
  
  TypeRef getUnqualifiedType() const {
    return TypeRef(getTypePtr());
  }
  
  /// operator==/!= - Indicate whether the specified types and qualifiers are
  /// identical.
  bool operator==(const TypeRef &RHS) const {
    return ThePtr == RHS.ThePtr;
  }
  bool operator!=(const TypeRef &RHS) const {
    return ThePtr != RHS.ThePtr;
  }
  
  /// getCanonicalType - Return the canonical version of this type, with the
  /// appropriate type qualifiers on it.
  inline TypeRef getCanonicalType() const;
  
  void getAsString(std::string &S) const;
  void dump() const;
};

} // end clang.

/// Implement simplify_type for TypeRef, so that we can dyn_cast from TypeRef to
/// a specific Type class.
template<> struct simplify_type<const clang::TypeRef> {
  typedef clang::Type* SimpleType;
  static SimpleType getSimplifiedValue(const clang::TypeRef &Val) {
    return Val.getTypePtr();
  }
};
template<> struct simplify_type<clang::TypeRef>
  : public simplify_type<const clang::TypeRef> {};

namespace clang {

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
/// TypeNameType).  Its CanonicalType pointer points to the 'int' Type.  Next
/// there is a PointerType that represents 'int*', which, like 'int', is
/// canonical.  Finally, there is a PointerType type for 'foo*' whose canonical
/// type is 'int*', and there is a TypeNameType for 'bar', whose canonical type
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
public:
  enum TypeClass {
    Builtin, Pointer, Array, FunctionNoProto, FunctionProto, TypeName, Tagged
  };
private:
  Type *CanonicalType;

  /// TypeClass bitfield - Enum that specifies what subclass this belongs to.
  /// Note that this should stay at the end of the ivars for Type so that
  /// subclasses can pack their bitfields into the same word.
  TypeClass TC : 3;
public:
  Type(TypeClass tc, Type *Canonical)
    : CanonicalType(Canonical ? Canonical : this), TC(tc) {}
  virtual ~Type();
  
  TypeClass getTypeClass() const { return TC; }
  
  bool isCanonical() const { return CanonicalType == this; }
  Type *getCanonicalType() const { return CanonicalType; }
  
  /// isVoidType - Helper method to determine if this is the 'void' type.
  bool isVoidType() const;
  
  /// isIncompleteType - Return true if this is an incomplete type (C99 6.2.5p1)
  /// - a type that can describe objects, but which lacks information needed to
  /// determine its size (e.g. void, or a fwd declared struct).
  bool isIncompleteType() const;
  
  virtual void getAsString(std::string &InnerString) const = 0;
  
  static bool classof(const Type *) { return true; }
};

/// BuiltinType - This class is used for builtin types like 'int'.  Builtin
/// types are always canonical and have a literal name field.
class BuiltinType : public Type {
public:
  enum Kind {
    Void, Bool, Char,
    SChar,  Short,  Int,  Long,  LongLong,
    UChar, UShort, UInt, ULong, ULongLong,
    Float, Double, LongDouble,
    FloatComplex, DoubleComplex, LongDoubleComplex
  };
private:
  Kind TypeKind;
public:
  BuiltinType(Kind K) : Type(Builtin, 0), TypeKind(K) {}
  
  Kind getKind() const { return TypeKind; }
  const char *getName() const;
  
  virtual void getAsString(std::string &InnerString) const;
  
  static bool classof(const Type *T) { return T->getTypeClass() == Builtin; }
  static bool classof(const BuiltinType *) { return true; }
};

/// PointerType - C99 6.7.5.1 - Pointer Declarators.
///
class PointerType : public Type, public FoldingSetNode {
  TypeRef PointeeType;
  PointerType(TypeRef Pointee, Type *CanonicalPtr) :
    Type(Pointer, CanonicalPtr), PointeeType(Pointee) {
  }
  friend class ASTContext;  // ASTContext creates these.
public:
    
  TypeRef getPointeeType() const { return PointeeType; }
  
  virtual void getAsString(std::string &InnerString) const;
  
  
  void Profile(FoldingSetNodeID &ID) {
    Profile(ID, getPointeeType());
  }
  static void Profile(FoldingSetNodeID &ID, TypeRef Pointee);
  
  static bool classof(const Type *T) { return T->getTypeClass() == Pointer; }
  static bool classof(const PointerType *) { return true; }
};

/// PointerType - C99 6.7.5.2 - Array Declarators.
///
class ArrayType : public Type {
public:
  /// ArraySizeModifier - Capture whether this is a normal array (e.g. int X[4])
  /// an array with a static size (e.g. int X[static 4]), or with a star size
  /// (e.g. int X[*]).
  enum ArraySizeModifier {
    Normal, Static, Star
  };
private:
  /// NOTE: These fields are packed into the bitfields space in the Type class.
  ArraySizeModifier SizeModifier : 2;
  
  /// IndexTypeQuals - Capture qualifiers in declarations like:
  /// 'int X[static restrict 4]'.
  unsigned IndexTypeQuals : 3;
  
  /// ElementType - The element type of the array.
  TypeRef ElementType;
  
  /// FIXME: Capture size for VLA or constant size.
  /// Use this to implement Type::isIncompleteType.

  ArrayType(TypeRef et, ArraySizeModifier sm, unsigned tq, Type *can)
    : Type(Array, can), SizeModifier(sm), IndexTypeQuals(tq), ElementType(et) {}
  friend class ASTContext;  // ASTContext creates these.
public:
    
  TypeRef getElementType() const { return ElementType; }
  ArraySizeModifier getSizeModifier() const { return SizeModifier; }
  unsigned getIndexTypeQualifier() const { return IndexTypeQuals; }
  
  virtual void getAsString(std::string &InnerString) const;
  
  static bool classof(const Type *T) { return T->getTypeClass() == Array; }
  static bool classof(const ArrayType *) { return true; }
};

/// FunctionType - C99 6.7.5.3 - Array Declarators.  This is the common base
/// class of FunctionTypeNoProto and FunctionTypeProto.
///
class FunctionType : public Type {
  /// SubClassData - This field is owned by the subclass, put here to pack
  /// tightly with the ivars in Type.
  bool SubClassData : 1;
  
  // The type returned by the function.
  TypeRef ResultType;
protected:
  FunctionType(TypeClass tc, TypeRef res, bool SubclassInfo, Type *Canonical)
    : Type(tc, Canonical), SubClassData(SubclassInfo), ResultType(res) {}
  bool getSubClassData() const { return SubClassData; }
public:
  
  TypeRef getResultType() const { return ResultType; }

  
  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionNoProto ||
           T->getTypeClass() == FunctionProto;
  }
  static bool classof(const FunctionType *) { return true; }
};

/// FunctionTypeNoProto - Represents a K&R-style 'int foo()' function, which has
/// no information available about its arguments.
class FunctionTypeNoProto : public FunctionType {
  FunctionTypeNoProto(TypeRef Result, Type *Canonical)
    : FunctionType(FunctionNoProto, Result, false, Canonical) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  // No additional state past what FunctionType provides.
  
  virtual void getAsString(std::string &InnerString) const;

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionNoProto;
  }
  static bool classof(const FunctionTypeNoProto *) { return true; }
};

/// FunctionTypeProto - Represents a prototype with argument type info, e.g.
/// 'int foo(int)' or 'int foo(void)'.  'void' is represented as having no
/// arguments, not as having a single void argument.
class FunctionTypeProto : public FunctionType, public FoldingSetNode {
  FunctionTypeProto(TypeRef Result, TypeRef *ArgArray, unsigned numArgs,
                    bool isVariadic, Type *Canonical)
    : FunctionType(FunctionProto, Result, isVariadic, Canonical),
      NumArgs(numArgs) {
    for (unsigned i = 0; i != numArgs; ++i)
      ArgInfo[i] = ArgArray[i];
  }
  
  /// NumArgs - The number of arguments this function has, not counting '...'.
  unsigned NumArgs;
  
  /// ArgInfo - This array holds the argument types.  Note that this is actually
  /// a variable-sized array, so it must be the last instance variable in the
  /// class.
  TypeRef ArgInfo[1];
  friend class ASTContext;  // ASTContext creates these.
public:
  unsigned getNumArgs() const { return NumArgs; }
  TypeRef getArgType(unsigned i) const {
    assert(i < NumArgs && "Invalid argument number!");
    return ArgInfo[i];
  }
    
  bool isVariadic() const { return getSubClassData(); }
  
  virtual void getAsString(std::string &InnerString) const;

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionProto;
  }
  static bool classof(const FunctionTypeProto *) { return true; }
  
  void Profile(FoldingSetNodeID &ID) {
    Profile(ID, getResultType(), ArgInfo, NumArgs, isVariadic());
  }
  static void Profile(FoldingSetNodeID &ID, TypeRef Result, TypeRef* ArgTys,
                      unsigned NumArgs, bool isVariadic);
};


class TypedefType : public Type {
  TypedefDecl *Decl;
  TypedefType(TypedefDecl *D, Type *can) : Type(TypeName, can), Decl(D) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  
  TypedefDecl *getDecl() const { return Decl; }
    
  virtual void getAsString(std::string &InnerString) const;

  static bool classof(const Type *T) { return T->getTypeClass() == TypeName; }
  static bool classof(const TypedefType *) { return true; }
};


class TaggedType : public Type {
  TagDecl *Decl;
  TaggedType(TagDecl *D, Type *can) : Type(Tagged, can), Decl(D) {}
  friend class ASTContext;  // ASTContext creates these.
public:
    
  TagDecl *getDecl() const { return Decl; }
  
  virtual void getAsString(std::string &InnerString) const;
  
  static bool classof(const Type *T) { return T->getTypeClass() == Tagged; }
  static bool classof(const TaggedType *) { return true; }
};

/// RecordType - This is a helper class that allows the use of isa/cast/dyncast
/// to detect TaggedType objects of structs/unions/classes.
class RecordType : public TaggedType {
  RecordType(); // DO NOT IMPLEMENT
public:
    
  RecordDecl *getDecl() const {
    return reinterpret_cast<RecordDecl*>(TaggedType::getDecl());
  }
  
  static bool classof(const Type *T);
  static bool classof(const RecordType *) { return true; }
};


/// ...

// TODO: When we support C++, we should have types for uses of template with
// default parameters.  We should be able to distinguish source use of
// 'std::vector<int>' from 'std::vector<int, std::allocator<int> >'. Though they
// specify the same type, we want to print the default argument only if
// specified in the source code.


/// getCanonicalType - Return the canonical version of this type, with the
/// appropriate type qualifiers on it.
inline TypeRef TypeRef::getCanonicalType() const {
  return TypeRef(getTypePtr()->getCanonicalType(), getQualifiers());
}

  
}  // end namespace clang
}  // end namespace llvm

#endif
