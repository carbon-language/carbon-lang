//===-- llvm/Type.h - Classes for handling data types -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Type class.  For more "Type"
// stuff, look in DerivedTypes.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TYPE_H
#define LLVM_TYPE_H

#include "llvm/Support/Casting.h"

namespace llvm {

class DerivedType;
class PointerType;
class IntegerType;
class raw_ostream;
class Module;
class LLVMContext;
class LLVMContextImpl;
template<class GraphType> struct GraphTraits;

/// The instances of the Type class are immutable: once they are created,
/// they are never changed.  Also note that only one instance of a particular
/// type is ever created.  Thus seeing if two types are equal is a matter of
/// doing a trivial pointer comparison. To enforce that no two equal instances
/// are created, Type instances can only be created via static factory methods 
/// in class Type and in derived classes.  Once allocated, Types are never
/// free'd.
/// 
class Type {
public:
  //===--------------------------------------------------------------------===//
  /// Definitions of all of the base types for the Type system.  Based on this
  /// value, you can cast to a "DerivedType" subclass (see DerivedTypes.h)
  /// Note: If you add an element to this, you need to add an element to the
  /// Type::getPrimitiveType function, or else things will break!
  /// Also update LLVMTypeKind and LLVMGetTypeKind () in the C binding.
  ///
  enum TypeID {
    // PrimitiveTypes - make sure LastPrimitiveTyID stays up to date.
    VoidTyID = 0,    ///<  0: type with no size
    FloatTyID,       ///<  1: 32-bit floating point type
    DoubleTyID,      ///<  2: 64-bit floating point type
    X86_FP80TyID,    ///<  3: 80-bit floating point type (X87)
    FP128TyID,       ///<  4: 128-bit floating point type (112-bit mantissa)
    PPC_FP128TyID,   ///<  5: 128-bit floating point type (two 64-bits, PowerPC)
    LabelTyID,       ///<  6: Labels
    MetadataTyID,    ///<  7: Metadata
    X86_MMXTyID,     ///<  8: MMX vectors (64 bits, X86 specific)

    // Derived types... see DerivedTypes.h file.
    // Make sure FirstDerivedTyID stays up to date!
    IntegerTyID,     ///<  9: Arbitrary bit width integers
    FunctionTyID,    ///< 10: Functions
    StructTyID,      ///< 11: Structures
    ArrayTyID,       ///< 12: Arrays
    PointerTyID,     ///< 13: Pointers
    VectorTyID,      ///< 14: SIMD 'packed' format, or other vector type

    NumTypeIDs,                         // Must remain as last defined ID
    LastPrimitiveTyID = X86_MMXTyID,
    FirstDerivedTyID = IntegerTyID
  };

private:
  /// Context - This refers to the LLVMContext in which this type was uniqued.
  LLVMContext &Context;

  TypeID   ID : 8;            // The current base type of this type.
  unsigned SubclassData : 24; // Space for subclasses to store data

protected:
  friend class LLVMContextImpl;
  explicit Type(LLVMContext &C, TypeID tid)
    : Context(C), ID(tid), SubclassData(0),
      NumContainedTys(0), ContainedTys(0) {}
  ~Type() {}

  unsigned getSubclassData() const { return SubclassData; }
  void setSubclassData(unsigned val) {
    SubclassData = val;
    // Ensure we don't have any accidental truncation.
    assert(SubclassData == val && "Subclass data too large for field");
  }

  /// NumContainedTys - Keeps track of how many Type*'s there are in the
  /// ContainedTys list.
  unsigned NumContainedTys;

  /// ContainedTys - A pointer to the array of Types contained by this Type.
  /// For example, this includes the arguments of a function type, the elements
  /// of a structure, the pointee of a pointer, the element type of an array,
  /// etc.  This pointer may be 0 for types that don't contain other types
  /// (Integer, Double, Float).
  Type * const *ContainedTys;

public:
  void print(raw_ostream &O) const;
  void dump() const;

  /// getContext - Return the LLVMContext in which this type was uniqued.
  LLVMContext &getContext() const { return Context; }

  //===--------------------------------------------------------------------===//
  // Accessors for working with types.
  //

  /// getTypeID - Return the type id for the type.  This will return one
  /// of the TypeID enum elements defined above.
  ///
  TypeID getTypeID() const { return ID; }

  /// isVoidTy - Return true if this is 'void'.
  bool isVoidTy() const { return ID == VoidTyID; }

  /// isFloatTy - Return true if this is 'float', a 32-bit IEEE fp type.
  bool isFloatTy() const { return ID == FloatTyID; }
  
  /// isDoubleTy - Return true if this is 'double', a 64-bit IEEE fp type.
  bool isDoubleTy() const { return ID == DoubleTyID; }

  /// isX86_FP80Ty - Return true if this is x86 long double.
  bool isX86_FP80Ty() const { return ID == X86_FP80TyID; }

  /// isFP128Ty - Return true if this is 'fp128'.
  bool isFP128Ty() const { return ID == FP128TyID; }

  /// isPPC_FP128Ty - Return true if this is powerpc long double.
  bool isPPC_FP128Ty() const { return ID == PPC_FP128TyID; }

  /// isFloatingPointTy - Return true if this is one of the five floating point
  /// types
  bool isFloatingPointTy() const {
    return ID == FloatTyID || ID == DoubleTyID ||
      ID == X86_FP80TyID || ID == FP128TyID || ID == PPC_FP128TyID;
  }

  /// isX86_MMXTy - Return true if this is X86 MMX.
  bool isX86_MMXTy() const { return ID == X86_MMXTyID; }

  /// isFPOrFPVectorTy - Return true if this is a FP type or a vector of FP.
  ///
  bool isFPOrFPVectorTy() const;
 
  /// isLabelTy - Return true if this is 'label'.
  bool isLabelTy() const { return ID == LabelTyID; }

  /// isMetadataTy - Return true if this is 'metadata'.
  bool isMetadataTy() const { return ID == MetadataTyID; }

  /// isIntegerTy - True if this is an instance of IntegerType.
  ///
  bool isIntegerTy() const { return ID == IntegerTyID; } 

  /// isIntegerTy - Return true if this is an IntegerType of the given width.
  bool isIntegerTy(unsigned Bitwidth) const;

  /// isIntOrIntVectorTy - Return true if this is an integer type or a vector of
  /// integer types.
  ///
  bool isIntOrIntVectorTy() const;
  
  /// isFunctionTy - True if this is an instance of FunctionType.
  ///
  bool isFunctionTy() const { return ID == FunctionTyID; }

  /// isStructTy - True if this is an instance of StructType.
  ///
  bool isStructTy() const { return ID == StructTyID; }

  /// isArrayTy - True if this is an instance of ArrayType.
  ///
  bool isArrayTy() const { return ID == ArrayTyID; }

  /// isPointerTy - True if this is an instance of PointerType.
  ///
  bool isPointerTy() const { return ID == PointerTyID; }

  /// isVectorTy - True if this is an instance of VectorType.
  ///
  bool isVectorTy() const { return ID == VectorTyID; }

  /// canLosslesslyBitCastTo - Return true if this type could be converted 
  /// with a lossless BitCast to type 'Ty'. For example, i8* to i32*. BitCasts 
  /// are valid for types of the same size only where no re-interpretation of 
  /// the bits is done.
  /// @brief Determine if this type could be losslessly bitcast to Ty
  bool canLosslesslyBitCastTo(const Type *Ty) const;

  /// isEmptyTy - Return true if this type is empty, that is, it has no
  /// elements or all its elements are empty.
  bool isEmptyTy() const;

  /// Here are some useful little methods to query what type derived types are
  /// Note that all other types can just compare to see if this == Type::xxxTy;
  ///
  bool isPrimitiveType() const { return ID <= LastPrimitiveTyID; }
  bool isDerivedType()   const { return ID >= FirstDerivedTyID; }

  /// isFirstClassType - Return true if the type is "first class", meaning it
  /// is a valid type for a Value.
  ///
  bool isFirstClassType() const {
    return ID != FunctionTyID && ID != VoidTyID;
  }

  /// isSingleValueType - Return true if the type is a valid type for a
  /// register in codegen.  This includes all first-class types except struct
  /// and array types.
  ///
  bool isSingleValueType() const {
    return (ID != VoidTyID && isPrimitiveType()) ||
            ID == IntegerTyID || ID == PointerTyID || ID == VectorTyID;
  }

  /// isAggregateType - Return true if the type is an aggregate type. This
  /// means it is valid as the first operand of an insertvalue or
  /// extractvalue instruction. This includes struct and array types, but
  /// does not include vector types.
  ///
  bool isAggregateType() const {
    return ID == StructTyID || ID == ArrayTyID;
  }

  /// isSized - Return true if it makes sense to take the size of this type.  To
  /// get the actual size for a particular target, it is reasonable to use the
  /// TargetData subsystem to do this.
  ///
  bool isSized() const {
    // If it's a primitive, it is always sized.
    if (ID == IntegerTyID || isFloatingPointTy() || ID == PointerTyID ||
        ID == X86_MMXTyID)
      return true;
    // If it is not something that can have a size (e.g. a function or label),
    // it doesn't have a size.
    if (ID != StructTyID && ID != ArrayTyID && ID != VectorTyID)
      return false;
    // Otherwise we have to try harder to decide.
    return isSizedDerivedType();
  }

  /// getPrimitiveSizeInBits - Return the basic size of this type if it is a
  /// primitive type.  These are fixed by LLVM and are not target dependent.
  /// This will return zero if the type does not have a size or is not a
  /// primitive type.
  ///
  /// Note that this may not reflect the size of memory allocated for an
  /// instance of the type or the number of bytes that are written when an
  /// instance of the type is stored to memory. The TargetData class provides
  /// additional query functions to provide this information.
  ///
  unsigned getPrimitiveSizeInBits() const;

  /// getScalarSizeInBits - If this is a vector type, return the
  /// getPrimitiveSizeInBits value for the element type. Otherwise return the
  /// getPrimitiveSizeInBits value for this type.
  unsigned getScalarSizeInBits() const;

  /// getFPMantissaWidth - Return the width of the mantissa of this type.  This
  /// is only valid on floating point types.  If the FP type does not
  /// have a stable mantissa (e.g. ppc long double), this method returns -1.
  int getFPMantissaWidth() const;

  /// getScalarType - If this is a vector type, return the element type,
  /// otherwise return 'this'.
  const Type *getScalarType() const;

  //===--------------------------------------------------------------------===//
  // Type Iteration support.
  //
  typedef Type * const *subtype_iterator;
  subtype_iterator subtype_begin() const { return ContainedTys; }
  subtype_iterator subtype_end() const { return &ContainedTys[NumContainedTys];}

  /// getContainedType - This method is used to implement the type iterator
  /// (defined a the end of the file).  For derived types, this returns the
  /// types 'contained' in the derived type.
  ///
  Type *getContainedType(unsigned i) const {
    assert(i < NumContainedTys && "Index out of range!");
    return ContainedTys[i];
  }

  /// getNumContainedTypes - Return the number of types in the derived type.
  ///
  unsigned getNumContainedTypes() const { return NumContainedTys; }

  //===--------------------------------------------------------------------===//
  // Static members exported by the Type class itself.  Useful for getting
  // instances of Type.
  //

  /// getPrimitiveType - Return a type based on an identifier.
  static Type *getPrimitiveType(LLVMContext &C, TypeID IDNumber);

  //===--------------------------------------------------------------------===//
  // These are the builtin types that are always available.
  //
  static Type *getVoidTy(LLVMContext &C);
  static Type *getLabelTy(LLVMContext &C);
  static Type *getFloatTy(LLVMContext &C);
  static Type *getDoubleTy(LLVMContext &C);
  static Type *getMetadataTy(LLVMContext &C);
  static Type *getX86_FP80Ty(LLVMContext &C);
  static Type *getFP128Ty(LLVMContext &C);
  static Type *getPPC_FP128Ty(LLVMContext &C);
  static Type *getX86_MMXTy(LLVMContext &C);
  static IntegerType *getIntNTy(LLVMContext &C, unsigned N);
  static IntegerType *getInt1Ty(LLVMContext &C);
  static IntegerType *getInt8Ty(LLVMContext &C);
  static IntegerType *getInt16Ty(LLVMContext &C);
  static IntegerType *getInt32Ty(LLVMContext &C);
  static IntegerType *getInt64Ty(LLVMContext &C);

  //===--------------------------------------------------------------------===//
  // Convenience methods for getting pointer types with one of the above builtin
  // types as pointee.
  //
  static PointerType *getFloatPtrTy(LLVMContext &C, unsigned AS = 0);
  static PointerType *getDoublePtrTy(LLVMContext &C, unsigned AS = 0);
  static PointerType *getX86_FP80PtrTy(LLVMContext &C, unsigned AS = 0);
  static PointerType *getFP128PtrTy(LLVMContext &C, unsigned AS = 0);
  static PointerType *getPPC_FP128PtrTy(LLVMContext &C, unsigned AS = 0);
  static PointerType *getX86_MMXPtrTy(LLVMContext &C, unsigned AS = 0);
  static PointerType *getIntNPtrTy(LLVMContext &C, unsigned N, unsigned AS = 0);
  static PointerType *getInt1PtrTy(LLVMContext &C, unsigned AS = 0);
  static PointerType *getInt8PtrTy(LLVMContext &C, unsigned AS = 0);
  static PointerType *getInt16PtrTy(LLVMContext &C, unsigned AS = 0);
  static PointerType *getInt32PtrTy(LLVMContext &C, unsigned AS = 0);
  static PointerType *getInt64PtrTy(LLVMContext &C, unsigned AS = 0);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Type *) { return true; }

  /// getPointerTo - Return a pointer to the current type.  This is equivalent
  /// to PointerType::get(Foo, AddrSpace).
  PointerType *getPointerTo(unsigned AddrSpace = 0) const;

private:
  /// isSizedDerivedType - Derived types like structures and arrays are sized
  /// iff all of the members of the type are sized as well.  Since asking for
  /// their size is relatively uncommon, move this operation out of line.
  bool isSizedDerivedType() const;
};

// Printing of types.
static inline raw_ostream &operator<<(raw_ostream &OS, const Type &T) {
  T.print(OS);
  return OS;
}

// allow isa<PointerType>(x) to work without DerivedTypes.h included.
template <> struct isa_impl<PointerType, Type> {
  static inline bool doit(const Type &Ty) {
    return Ty.getTypeID() == Type::PointerTyID;
  }
};

  
//===----------------------------------------------------------------------===//
// Provide specializations of GraphTraits to be able to treat a type as a
// graph of sub types.


template <> struct GraphTraits<Type*> {
  typedef Type NodeType;
  typedef Type::subtype_iterator ChildIteratorType;

  static inline NodeType *getEntryNode(Type *T) { return T; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->subtype_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->subtype_end();
  }
};

template <> struct GraphTraits<const Type*> {
  typedef const Type NodeType;
  typedef Type::subtype_iterator ChildIteratorType;

  static inline NodeType *getEntryNode(const Type *T) { return T; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->subtype_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->subtype_end();
  }
};

} // End llvm namespace

#endif
