//===-- llvm/DerivedTypes.h - Classes for handling data types ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of classes that represent "derived
// types".  These are things like "arrays of x" or "structure of x, y, z" or
// "function returning x taking (y,z) as parameters", etc...
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
class APInt;
class LLVMContext;
template<typename T> class ArrayRef;
class StringRef;

/// Class to represent integer types. Note that this class is also used to
/// represent the built-in integer types: Int1Ty, Int8Ty, Int16Ty, Int32Ty and
/// Int64Ty.
/// @brief Integer representation type
class IntegerType : public Type {
  friend class LLVMContextImpl;
  
protected:
  explicit IntegerType(LLVMContext &C, unsigned NumBits) : Type(C, IntegerTyID){
    setSubclassData(NumBits);
  }
public:
  /// This enum is just used to hold constants we need for IntegerType.
  enum {
    MIN_INT_BITS = 1,        ///< Minimum number of bits that can be specified
    MAX_INT_BITS = (1<<23)-1 ///< Maximum number of bits that can be specified
      ///< Note that bit width is stored in the Type classes SubclassData field
      ///< which has 23 bits. This yields a maximum bit width of 8,388,607 bits.
  };

  /// This static method is the primary way of constructing an IntegerType.
  /// If an IntegerType with the same NumBits value was previously instantiated,
  /// that instance will be returned. Otherwise a new one will be created. Only
  /// one instance with a given NumBits value is ever created.
  /// @brief Get or create an IntegerType instance.
  static IntegerType *get(LLVMContext &C, unsigned NumBits);

  /// @brief Get the number of bits in this IntegerType
  unsigned getBitWidth() const { return getSubclassData(); }

  /// getBitMask - Return a bitmask with ones set for all of the bits
  /// that can be set by an unsigned version of this type.  This is 0xFF for
  /// i8, 0xFFFF for i16, etc.
  uint64_t getBitMask() const {
    return ~uint64_t(0UL) >> (64-getBitWidth());
  }

  /// getSignBit - Return a uint64_t with just the most significant bit set (the
  /// sign bit, if the value is treated as a signed number).
  uint64_t getSignBit() const {
    return 1ULL << (getBitWidth()-1);
  }

  /// For example, this is 0xFF for an 8 bit integer, 0xFFFF for i16, etc.
  /// @returns a bit mask with ones set for all the bits of this type.
  /// @brief Get a bit mask for this type.
  APInt getMask() const;

  /// This method determines if the width of this IntegerType is a power-of-2
  /// in terms of 8 bit bytes.
  /// @returns true if this is a power-of-2 byte width.
  /// @brief Is this a power-of-2 byte-width IntegerType ?
  bool isPowerOf2ByteWidth() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const IntegerType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == IntegerTyID;
  }
};


/// FunctionType - Class to represent function types
///
class FunctionType : public Type {
  FunctionType(const FunctionType &);                   // Do not implement
  const FunctionType &operator=(const FunctionType &);  // Do not implement
  FunctionType(const Type *Result, ArrayRef<Type*> Params, bool IsVarArgs);

public:
  /// FunctionType::get - This static method is the primary way of constructing
  /// a FunctionType.
  ///
  static FunctionType *get(const Type *Result,
                           ArrayRef<const Type*> Params, bool isVarArg);
  static FunctionType *get(const Type *Result,
                           ArrayRef<Type*> Params, bool isVarArg);

  /// FunctionType::get - Create a FunctionType taking no parameters.
  ///
  static FunctionType *get(const Type *Result, bool isVarArg);
  
  /// isValidReturnType - Return true if the specified type is valid as a return
  /// type.
  static bool isValidReturnType(const Type *RetTy);

  /// isValidArgumentType - Return true if the specified type is valid as an
  /// argument type.
  static bool isValidArgumentType(const Type *ArgTy);

  bool isVarArg() const { return getSubclassData(); }
  Type *getReturnType() const { return ContainedTys[0]; }

  typedef Type::subtype_iterator param_iterator;
  param_iterator param_begin() const { return ContainedTys + 1; }
  param_iterator param_end() const { return &ContainedTys[NumContainedTys]; }

  // Parameter type accessors.
  Type *getParamType(unsigned i) const { return ContainedTys[i+1]; }

  /// getNumParams - Return the number of fixed parameters this function type
  /// requires.  This does not consider varargs.
  ///
  unsigned getNumParams() const { return NumContainedTys - 1; }

  // Methods for support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const FunctionType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == FunctionTyID;
  }
};


/// CompositeType - Common super class of ArrayType, StructType, PointerType
/// and VectorType.
class CompositeType : public Type {
protected:
  explicit CompositeType(LLVMContext &C, TypeID tid) : Type(C, tid) { }
public:

  /// getTypeAtIndex - Given an index value into the type, return the type of
  /// the element.
  ///
  Type *getTypeAtIndex(const Value *V) const;
  Type *getTypeAtIndex(unsigned Idx) const;
  bool indexValid(const Value *V) const;
  bool indexValid(unsigned Idx) const;

  // Methods for support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const CompositeType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == ArrayTyID ||
           T->getTypeID() == StructTyID ||
           T->getTypeID() == PointerTyID ||
           T->getTypeID() == VectorTyID;
  }
};


/// StructType - Class to represent struct types, both normal and packed.
/// Besides being optionally packed, structs can be either "anonymous" or may
/// have an identity.  Anonymous structs are uniqued by structural equivalence,
/// but types are each unique when created, and optionally have a name.
///
class StructType : public CompositeType {
  StructType(const StructType &);                   // Do not implement
  const StructType &operator=(const StructType &);  // Do not implement
  StructType(LLVMContext &C)
    : CompositeType(C, StructTyID), SymbolTableEntry(0) {}
  enum {
    // This is the contents of the SubClassData field.
    SCDB_HasBody = 1,
    SCDB_Packed = 2,
    SCDB_IsAnonymous = 4
  };
  
  /// SymbolTableEntry - For a named struct that actually has a name, this is a
  /// pointer to the symbol table entry (maintained by LLVMContext) for the
  /// struct.  This is null if the type is an anonymous struct or if it is
  /// a named type that has an empty name.
  /// 
  void *SymbolTableEntry;
public:
  /// StructType::createNamed - This creates a named struct with no body
  /// specified.  If the name is empty, it creates an unnamed struct, which has
  /// a unique identity but no actual name.
  static StructType *createNamed(LLVMContext &Context, StringRef Name);
  
  static StructType *createNamed(StringRef Name, ArrayRef<Type*> Elements,
                                 bool isPacked = false);
  static StructType *createNamed(LLVMContext &Context, StringRef Name,
                                 ArrayRef<Type*> Elements,
                                 bool isPacked = false);
  static StructType *createNamed(StringRef Name, Type *elt1, ...) END_WITH_NULL;

  /// StructType::get - This static method is the primary way to create a
  /// StructType.
  ///
  /// FIXME: Remove the 'const Type*' version of this when types are pervasively
  /// de-constified.
  static StructType *get(LLVMContext &Context, ArrayRef<const Type*> Elements,
                         bool isPacked = false);
  static StructType *get(LLVMContext &Context, ArrayRef<Type*> Elements,
                         bool isPacked = false);

  /// StructType::get - Create an empty structure type.
  ///
  static StructType *get(LLVMContext &Context, bool isPacked = false);
  
  /// StructType::get - This static method is a convenience method for creating
  /// structure types by specifying the elements as arguments.  Note that this
  /// method always returns a non-packed struct, and requires at least one
  /// element type.
  static StructType *get(const Type *elt1, ...) END_WITH_NULL;

  bool isPacked() const { return (getSubclassData() & SCDB_Packed) != 0; }
  
  /// isAnonymous - Return true if this type is uniqued by structural
  /// equivalence, false if it has an identity.
  bool isAnonymous() const {return (getSubclassData() & SCDB_IsAnonymous) != 0;}
  
  /// isOpaque - Return true if this is a type with an identity that has no body
  /// specified yet.  These prints as 'opaque' in .ll files.
  bool isOpaque() const { return (getSubclassData() & SCDB_HasBody) == 0; }
  
  /// hasName - Return true if this is a named struct that has a non-empty name.
  bool hasName() const { return SymbolTableEntry != 0; }
  
  /// getName - Return the name for this struct type if it has an identity.
  /// This may return an empty string for an unnamed struct type.  Do not call
  /// this on an anonymous type.
  StringRef getName() const;
  
  /// setName - Change the name of this type to the specified name, or to a name
  /// with a suffix if there is a collision.  Do not call this on an anonymous
  /// type.
  void setName(StringRef Name);

  /// setBody - Specify a body for an opaque type.
  void setBody(ArrayRef<Type*> Elements, bool isPacked = false);
  void setBody(Type *elt1, ...) END_WITH_NULL;
  
  /// isValidElementType - Return true if the specified type is valid as a
  /// element type.
  static bool isValidElementType(const Type *ElemTy);
  

  // Iterator access to the elements.
  typedef Type::subtype_iterator element_iterator;
  element_iterator element_begin() const { return ContainedTys; }
  element_iterator element_end() const { return &ContainedTys[NumContainedTys];}

  /// isLayoutIdentical - Return true if this is layout identical to the
  /// specified struct.
  bool isLayoutIdentical(const StructType *Other) const;  
  
  // Random access to the elements
  unsigned getNumElements() const { return NumContainedTys; }
  Type *getElementType(unsigned N) const {
    assert(N < NumContainedTys && "Element number out of range!");
    return ContainedTys[N];
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const StructType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == StructTyID;
  }
};

/// SequentialType - This is the superclass of the array, pointer and vector
/// type classes.  All of these represent "arrays" in memory.  The array type
/// represents a specifically sized array, pointer types are unsized/unknown
/// size arrays, vector types represent specifically sized arrays that
/// allow for use of SIMD instructions.  SequentialType holds the common
/// features of all, which stem from the fact that all three lay their
/// components out in memory identically.
///
class SequentialType : public CompositeType {
  Type *ContainedType;               ///< Storage for the single contained type.
  SequentialType(const SequentialType &);                  // Do not implement!
  const SequentialType &operator=(const SequentialType &); // Do not implement!

protected:
  SequentialType(TypeID TID, Type *ElType)
    : CompositeType(ElType->getContext(), TID), ContainedType(ElType) {
    ContainedTys = &ContainedType;
    NumContainedTys = 1;
  }

public:
  Type *getElementType() const { return ContainedTys[0]; }

  // Methods for support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const SequentialType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == ArrayTyID ||
           T->getTypeID() == PointerTyID ||
           T->getTypeID() == VectorTyID;
  }
};


/// ArrayType - Class to represent array types.
///
class ArrayType : public SequentialType {
  uint64_t NumElements;

  ArrayType(const ArrayType &);                   // Do not implement
  const ArrayType &operator=(const ArrayType &);  // Do not implement
  ArrayType(Type *ElType, uint64_t NumEl);
public:
  /// ArrayType::get - This static method is the primary way to construct an
  /// ArrayType
  ///
  static ArrayType *get(const Type *ElementType, uint64_t NumElements);

  /// isValidElementType - Return true if the specified type is valid as a
  /// element type.
  static bool isValidElementType(const Type *ElemTy);

  uint64_t getNumElements() const { return NumElements; }

  // Methods for support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const ArrayType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == ArrayTyID;
  }
};

/// VectorType - Class to represent vector types.
///
class VectorType : public SequentialType {
  unsigned NumElements;

  VectorType(const VectorType &);                   // Do not implement
  const VectorType &operator=(const VectorType &);  // Do not implement
  VectorType(Type *ElType, unsigned NumEl);
public:
  /// VectorType::get - This static method is the primary way to construct an
  /// VectorType.
  ///
  static VectorType *get(const Type *ElementType, unsigned NumElements);

  /// VectorType::getInteger - This static method gets a VectorType with the
  /// same number of elements as the input type, and the element type is an
  /// integer type of the same width as the input element type.
  ///
  static VectorType *getInteger(const VectorType *VTy) {
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    Type *EltTy = IntegerType::get(VTy->getContext(), EltBits);
    return VectorType::get(EltTy, VTy->getNumElements());
  }

  /// VectorType::getExtendedElementVectorType - This static method is like
  /// getInteger except that the element types are twice as wide as the
  /// elements in the input type.
  ///
  static VectorType *getExtendedElementVectorType(const VectorType *VTy) {
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    Type *EltTy = IntegerType::get(VTy->getContext(), EltBits * 2);
    return VectorType::get(EltTy, VTy->getNumElements());
  }

  /// VectorType::getTruncatedElementVectorType - This static method is like
  /// getInteger except that the element types are half as wide as the
  /// elements in the input type.
  ///
  static VectorType *getTruncatedElementVectorType(const VectorType *VTy) {
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    assert((EltBits & 1) == 0 &&
           "Cannot truncate vector element with odd bit-width");
    Type *EltTy = IntegerType::get(VTy->getContext(), EltBits / 2);
    return VectorType::get(EltTy, VTy->getNumElements());
  }

  /// isValidElementType - Return true if the specified type is valid as a
  /// element type.
  static bool isValidElementType(const Type *ElemTy);

  /// @brief Return the number of elements in the Vector type.
  unsigned getNumElements() const { return NumElements; }

  /// @brief Return the number of bits in the Vector type.
  unsigned getBitWidth() const {
    return NumElements * getElementType()->getPrimitiveSizeInBits();
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const VectorType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == VectorTyID;
  }
};


/// PointerType - Class to represent pointers.
///
class PointerType : public SequentialType {
  PointerType(const PointerType &);                   // Do not implement
  const PointerType &operator=(const PointerType &);  // Do not implement
  explicit PointerType(Type *ElType, unsigned AddrSpace);
public:
  /// PointerType::get - This constructs a pointer to an object of the specified
  /// type in a numbered address space.
  static PointerType *get(const Type *ElementType, unsigned AddressSpace);

  /// PointerType::getUnqual - This constructs a pointer to an object of the
  /// specified type in the generic address space (address space zero).
  static PointerType *getUnqual(const Type *ElementType) {
    return PointerType::get(ElementType, 0);
  }

  /// isValidElementType - Return true if the specified type is valid as a
  /// element type.
  static bool isValidElementType(const Type *ElemTy);

  /// @brief Return the address space of the Pointer type.
  inline unsigned getAddressSpace() const { return getSubclassData(); }

  // Implement support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const PointerType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == PointerTyID;
  }
};

} // End llvm namespace

#endif
