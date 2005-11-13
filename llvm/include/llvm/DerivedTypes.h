//===-- llvm/DerivedTypes.h - Classes for handling data types ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of classes that represent "derived
// types".  These are things like "arrays of x" or "structure of x, y, z" or
// "method returning x taking (y,z) as parameters", etc...
//
// The implementations of these classes live in the Type.cpp file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DERIVED_TYPES_H
#define LLVM_DERIVED_TYPES_H

#include "llvm/Type.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class Value;
template<class ValType, class TypeClass> class TypeMap;
class FunctionValType;
class ArrayValType;
class StructValType;
class PointerValType;
class PackedValType;

class DerivedType : public Type, public AbstractTypeUser {
  friend class Type;

protected:
  DerivedType(TypeID id) : Type(id) {}

  /// notifyUsesThatTypeBecameConcrete - Notify AbstractTypeUsers of this type
  /// that the current type has transitioned from being abstract to being
  /// concrete.
  ///
  void notifyUsesThatTypeBecameConcrete();

  /// dropAllTypeUses - When this (abstract) type is resolved to be equal to
  /// another (more concrete) type, we must eliminate all references to other
  /// types, to avoid some circular reference problems.
  ///
  void dropAllTypeUses();

public:

  //===--------------------------------------------------------------------===//
  // Abstract Type handling methods - These types have special lifetimes, which
  // are managed by (add|remove)AbstractTypeUser. See comments in
  // AbstractTypeUser.h for more information.

  /// refineAbstractTypeTo - This function is used to when it is discovered that
  /// the 'this' abstract type is actually equivalent to the NewType specified.
  /// This causes all users of 'this' to switch to reference the more concrete
  /// type NewType and for 'this' to be deleted.
  ///
  void refineAbstractTypeTo(const Type *NewType);

  void dump() const { Type::dump(); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const DerivedType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->isDerivedType();
  }
};


/// FunctionType - Class to represent function types
///
class FunctionType : public DerivedType {
  friend class TypeMap<FunctionValType, FunctionType>;
  bool isVarArgs;

  FunctionType(const FunctionType &);                   // Do not implement
  const FunctionType &operator=(const FunctionType &);  // Do not implement
protected:
  /// This should really be private, but it squelches a bogus warning
  /// from GCC to make them protected:  warning: `class FunctionType' only
  /// defines private constructors and has no friends
  ///
  /// Private ctor - Only can be created by a static member...
  ///
  FunctionType(const Type *Result, const std::vector<const Type*> &Params,
               bool IsVarArgs);

public:
  /// FunctionType::get - This static method is the primary way of constructing
  /// a FunctionType
  ///
  static FunctionType *get(const Type *Result,
                           const std::vector<const Type*> &Params,
                           bool isVarArg);

  inline bool isVarArg() const { return isVarArgs; }
  inline const Type *getReturnType() const { return ContainedTys[0]; }

  typedef std::vector<PATypeHandle>::const_iterator param_iterator;
  param_iterator param_begin() const { return ContainedTys.begin()+1; }
  param_iterator param_end() const { return ContainedTys.end(); }

  // Parameter type accessors...
  const Type *getParamType(unsigned i) const { return ContainedTys[i+1]; }

  /// getNumParams - Return the number of fixed parameters this function type
  /// requires.  This does not consider varargs.
  ///
  unsigned getNumParams() const { return unsigned(ContainedTys.size()-1); }

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FunctionType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == FunctionTyID;
  }
};


/// CompositeType - Common super class of ArrayType, StructType, PointerType
/// and PackedType
class CompositeType : public DerivedType {
protected:
  inline CompositeType(TypeID id) : DerivedType(id) { }
public:

  /// getTypeAtIndex - Given an index value into the type, return the type of
  /// the element.
  ///
  virtual const Type *getTypeAtIndex(const Value *V) const = 0;
  virtual bool indexValid(const Value *V) const = 0;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CompositeType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == ArrayTyID ||
           T->getTypeID() == StructTyID ||
           T->getTypeID() == PointerTyID ||
           T->getTypeID() == PackedTyID;
  }
};


/// StructType - Class to represent struct types
///
class StructType : public CompositeType {
  friend class TypeMap<StructValType, StructType>;
  StructType(const StructType &);                   // Do not implement
  const StructType &operator=(const StructType &);  // Do not implement

protected:
  /// This should really be private, but it squelches a bogus warning
  /// from GCC to make them protected:  warning: `class StructType' only
  /// defines private constructors and has no friends
  ///
  /// Private ctor - Only can be created by a static member...
  ///
  StructType(const std::vector<const Type*> &Types);

public:
  /// StructType::get - This static method is the primary way to create a
  /// StructType.
  ///
  static StructType *get(const std::vector<const Type*> &Params);

  // Iterator access to the elements
  typedef std::vector<PATypeHandle>::const_iterator element_iterator;
  element_iterator element_begin() const { return ContainedTys.begin(); }
  element_iterator element_end() const { return ContainedTys.end(); }

  // Random access to the elements
  unsigned getNumElements() const { return unsigned(ContainedTys.size()); }
  const Type *getElementType(unsigned N) const {
    assert(N < ContainedTys.size() && "Element number out of range!");
    return ContainedTys[N];
  }

  /// getTypeAtIndex - Given an index value into the type, return the type of
  /// the element.  For a structure type, this must be a constant value...
  ///
  virtual const Type *getTypeAtIndex(const Value *V) const ;
  virtual bool indexValid(const Value *V) const;

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const StructType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == StructTyID;
  }
};


/// SequentialType - This is the superclass of the array, pointer and packed
/// type classes.  All of these represent "arrays" in memory.  The array type
/// represents a specifically sized array, pointer types are unsized/unknown
/// size arrays, packed types represent specifically sized arrays that
/// allow for use of SIMD instructions.  SequentialType holds the common
/// features of all, which stem from the fact that all three lay their
/// components out in memory identically.
///
class SequentialType : public CompositeType {
  SequentialType(const SequentialType &);                  // Do not implement!
  const SequentialType &operator=(const SequentialType &); // Do not implement!
protected:
  SequentialType(TypeID TID, const Type *ElType) : CompositeType(TID) {
    ContainedTys.reserve(1);
    ContainedTys.push_back(PATypeHandle(ElType, this));
  }

public:
  inline const Type *getElementType() const { return ContainedTys[0]; }

  virtual bool indexValid(const Value *V) const;

  /// getTypeAtIndex - Given an index value into the type, return the type of
  /// the element.  For sequential types, there is only one subtype...
  ///
  virtual const Type *getTypeAtIndex(const Value *V) const {
    return ContainedTys[0];
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SequentialType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == ArrayTyID ||
           T->getTypeID() == PointerTyID ||
           T->getTypeID() == PackedTyID;
  }
};


/// ArrayType - Class to represent array types
///
class ArrayType : public SequentialType {
  friend class TypeMap<ArrayValType, ArrayType>;
  uint64_t NumElements;

  ArrayType(const ArrayType &);                   // Do not implement
  const ArrayType &operator=(const ArrayType &);  // Do not implement
protected:
  /// This should really be private, but it squelches a bogus warning
  /// from GCC to make them protected:  warning: `class ArrayType' only
  /// defines private constructors and has no friends
  ///
  /// Private ctor - Only can be created by a static member...
  ///
  ArrayType(const Type *ElType, uint64_t NumEl);

public:
  /// ArrayType::get - This static method is the primary way to construct an
  /// ArrayType
  ///
  static ArrayType *get(const Type *ElementType, uint64_t NumElements);

  inline uint64_t getNumElements() const { return NumElements; }

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ArrayType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == ArrayTyID;
  }
};

/// PackedType - Class to represent packed types
///
class PackedType : public SequentialType {
  friend class TypeMap<PackedValType, PackedType>;
  unsigned NumElements;

  PackedType(const PackedType &);                   // Do not implement
  const PackedType &operator=(const PackedType &);  // Do not implement
protected:
  /// This should really be private, but it squelches a bogus warning
  /// from GCC to make them protected:  warning: `class PackedType' only
  /// defines private constructors and has no friends
  ///
  /// Private ctor - Only can be created by a static member...
  ///
  PackedType(const Type *ElType, unsigned NumEl);

public:
  /// PackedType::get - This static method is the primary way to construct an
  /// PackedType
  ///
  static PackedType *get(const Type *ElementType, unsigned NumElements);

  inline unsigned getNumElements() const { return NumElements; }

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const PackedType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == PackedTyID;
  }
};


/// PointerType - Class to represent pointers
///
class PointerType : public SequentialType {
  friend class TypeMap<PointerValType, PointerType>;
  PointerType(const PointerType &);                   // Do not implement
  const PointerType &operator=(const PointerType &);  // Do not implement
protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class PointerType' only
  // defines private constructors and has no friends

  // Private ctor - Only can be created by a static member...
  PointerType(const Type *ElType);

public:
  /// PointerType::get - This is the only way to construct a new pointer type.
  static PointerType *get(const Type *ElementType);

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Implement support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const PointerType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == PointerTyID;
  }
};


/// OpaqueType - Class to represent abstract types
///
class OpaqueType : public DerivedType {
  OpaqueType(const OpaqueType &);                   // DO NOT IMPLEMENT
  const OpaqueType &operator=(const OpaqueType &);  // DO NOT IMPLEMENT
protected:
  /// This should really be private, but it squelches a bogus warning
  /// from GCC to make them protected:  warning: `class OpaqueType' only
  /// defines private constructors and has no friends
  ///
  /// Private ctor - Only can be created by a static member...
  OpaqueType();

public:
  /// OpaqueType::get - Static factory method for the OpaqueType class...
  ///
  static OpaqueType *get() {
    return new OpaqueType();           // All opaque types are distinct
  }

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
    abort();   // FIXME: this is not really an AbstractTypeUser!
  }
  virtual void typeBecameConcrete(const DerivedType *AbsTy) {
    abort();   // FIXME: this is not really an AbstractTypeUser!
  }

  // Implement support for type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const OpaqueType *T) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == OpaqueTyID;
  }
};

} // End llvm namespace

#endif
