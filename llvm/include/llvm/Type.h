//===-- llvm/Type.h - Classes for handling data types -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TYPE_H
#define LLVM_TYPE_H

#include "llvm/AbstractTypeUser.h"
#include "llvm/Support/Casting.h"
#include "llvm/System/DataTypes.h"
#include "llvm/ADT/GraphTraits.h"
#include <string>
#include <vector>

namespace llvm {

class DerivedType;
class PointerType;
class IntegerType;
class TypeMapBase;
class raw_ostream;
class Module;
class LLVMContext;

/// This file contains the declaration of the Type class.  For more "Type" type
/// stuff, look in DerivedTypes.h.
///
/// The instances of the Type class are immutable: once they are created,
/// they are never changed.  Also note that only one instance of a particular
/// type is ever created.  Thus seeing if two types are equal is a matter of
/// doing a trivial pointer comparison. To enforce that no two equal instances
/// are created, Type instances can only be created via static factory methods 
/// in class Type and in derived classes.
/// 
/// Once allocated, Types are never free'd, unless they are an abstract type
/// that is resolved to a more concrete type.
/// 
/// Types themself don't have a name, and can be named either by:
/// - using SymbolTable instance, typically from some Module,
/// - using convenience methods in the Module class (which uses module's 
///    SymbolTable too).
///
/// Opaque types are simple derived types with no state.  There may be many
/// different Opaque type objects floating around, but two are only considered
/// identical if they are pointer equals of each other.  This allows us to have
/// two opaque types that end up resolving to different concrete types later.
///
/// Opaque types are also kinda weird and scary and different because they have
/// to keep a list of uses of the type.  When, through linking, parsing, or
/// bitcode reading, they become resolved, they need to find and update all
/// users of the unknown type, causing them to reference a new, more concrete
/// type.  Opaque types are deleted when their use list dwindles to zero users.
///
/// @brief Root of type hierarchy
class Type : public AbstractTypeUser {
public:
  //===-------------------------------------------------------------------===//
  /// Definitions of all of the base types for the Type system.  Based on this
  /// value, you can cast to a "DerivedType" subclass (see DerivedTypes.h)
  /// Note: If you add an element to this, you need to add an element to the
  /// Type::getPrimitiveType function, or else things will break!
  /// Also update LLVMTypeKind and LLVMGetTypeKind () in the C binding.
  ///
  enum TypeID {
    // PrimitiveTypes .. make sure LastPrimitiveTyID stays up to date
    VoidTyID = 0,    ///<  0: type with no size
    FloatTyID,       ///<  1: 32 bit floating point type
    DoubleTyID,      ///<  2: 64 bit floating point type
    X86_FP80TyID,    ///<  3: 80 bit floating point type (X87)
    FP128TyID,       ///<  4: 128 bit floating point type (112-bit mantissa)
    PPC_FP128TyID,   ///<  5: 128 bit floating point type (two 64-bits)
    LabelTyID,       ///<  6: Labels
    MetadataTyID,    ///<  7: Metadata

    // Derived types... see DerivedTypes.h file...
    // Make sure FirstDerivedTyID stays up to date!!!
    IntegerTyID,     ///<  8: Arbitrary bit width integers
    FunctionTyID,    ///<  9: Functions
    StructTyID,      ///< 10: Structures
    UnionTyID,       ///< 11: Unions
    ArrayTyID,       ///< 12: Arrays
    PointerTyID,     ///< 13: Pointers
    OpaqueTyID,      ///< 14: Opaque: type with unknown structure
    VectorTyID,      ///< 15: SIMD 'packed' format, or other vector type

    NumTypeIDs,                         // Must remain as last defined ID
    LastPrimitiveTyID = MetadataTyID,
    FirstDerivedTyID = IntegerTyID
  };

private:
  TypeID   ID : 8;    // The current base type of this type.
  bool     Abstract : 1;  // True if type contains an OpaqueType
  unsigned SubclassData : 23; //Space for subclasses to store data

  /// RefCount - This counts the number of PATypeHolders that are pointing to
  /// this type.  When this number falls to zero, if the type is abstract and
  /// has no AbstractTypeUsers, the type is deleted.  This is only sensical for
  /// derived types.
  ///
  mutable unsigned RefCount;

  /// Context - This refers to the LLVMContext in which this type was uniqued.
  LLVMContext &Context;
  friend class LLVMContextImpl;

  const Type *getForwardedTypeInternal() const;

  // Some Type instances are allocated as arrays, some aren't. So we provide
  // this method to get the right kind of destruction for the type of Type.
  void destroy() const; // const is a lie, this does "delete this"!

protected:
  explicit Type(LLVMContext &C, TypeID id) :
                             ID(id), Abstract(false), SubclassData(0),
                             RefCount(0), Context(C),
                             ForwardType(0), NumContainedTys(0),
                             ContainedTys(0) {}
  virtual ~Type() {
    assert(AbstractTypeUsers.empty() && "Abstract types remain");
  }

  /// Types can become nonabstract later, if they are refined.
  ///
  inline void setAbstract(bool Val) { Abstract = Val; }

  unsigned getRefCount() const { return RefCount; }

  unsigned getSubclassData() const { return SubclassData; }
  void setSubclassData(unsigned val) { SubclassData = val; }

  /// ForwardType - This field is used to implement the union find scheme for
  /// abstract types.  When types are refined to other types, this field is set
  /// to the more refined type.  Only abstract types can be forwarded.
  mutable const Type *ForwardType;


  /// AbstractTypeUsers - Implement a list of the users that need to be notified
  /// if I am a type, and I get resolved into a more concrete type.
  ///
  mutable std::vector<AbstractTypeUser *> AbstractTypeUsers;

  /// NumContainedTys - Keeps track of how many PATypeHandle instances there
  /// are at the end of this type instance for the list of contained types. It
  /// is the subclasses responsibility to set this up. Set to 0 if there are no
  /// contained types in this type.
  unsigned NumContainedTys;

  /// ContainedTys - A pointer to the array of Types (PATypeHandle) contained 
  /// by this Type.  For example, this includes the arguments of a function 
  /// type, the elements of a structure, the pointee of a pointer, the element
  /// type of an array, etc.  This pointer may be 0 for types that don't 
  /// contain other types (Integer, Double, Float).  In general, the subclass 
  /// should arrange for space for the PATypeHandles to be included in the 
  /// allocation of the type object and set this pointer to the address of the 
  /// first element. This allows the Type class to manipulate the ContainedTys 
  /// without understanding the subclass's placement for this array.  keeping 
  /// it here also allows the subtype_* members to be implemented MUCH more 
  /// efficiently, and dynamically very few types do not contain any elements.
  PATypeHandle *ContainedTys;

public:
  void print(raw_ostream &O) const;

  /// @brief Debugging support: print to stderr
  void dump() const;

  /// @brief Debugging support: print to stderr (use type names from context
  /// module).
  void dump(const Module *Context) const;

  /// getContext - Fetch the LLVMContext in which this type was uniqued.
  LLVMContext &getContext() const { return Context; }

  //===--------------------------------------------------------------------===//
  // Property accessors for dealing with types... Some of these virtual methods
  // are defined in private classes defined in Type.cpp for primitive types.
  //

  /// getDescription - Return the string representation of the type.
  std::string getDescription() const;

  /// getTypeID - Return the type id for the type.  This will return one
  /// of the TypeID enum elements defined above.
  ///
  inline TypeID getTypeID() const { return ID; }

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
  bool isFloatingPointTy() const { return ID == FloatTyID || ID == DoubleTyID ||
      ID == X86_FP80TyID || ID == FP128TyID || ID == PPC_FP128TyID; }

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

  /// isUnionTy - True if this is an instance of UnionType.
  ///
  bool isUnionTy() const { return ID == UnionTyID; }

  /// isArrayTy - True if this is an instance of ArrayType.
  ///
  bool isArrayTy() const { return ID == ArrayTyID; }

  /// isPointerTy - True if this is an instance of PointerType.
  ///
  bool isPointerTy() const { return ID == PointerTyID; }

  /// isOpaqueTy - True if this is an instance of OpaqueType.
  ///
  bool isOpaqueTy() const { return ID == OpaqueTyID; }

  /// isVectorTy - True if this is an instance of VectorType.
  ///
  bool isVectorTy() const { return ID == VectorTyID; }

  /// isAbstract - True if the type is either an Opaque type, or is a derived
  /// type that includes an opaque type somewhere in it.
  ///
  inline bool isAbstract() const { return Abstract; }

  /// canLosslesslyBitCastTo - Return true if this type could be converted 
  /// with a lossless BitCast to type 'Ty'. For example, i8* to i32*. BitCasts 
  /// are valid for types of the same size only where no re-interpretation of 
  /// the bits is done.
  /// @brief Determine if this type could be losslessly bitcast to Ty
  bool canLosslesslyBitCastTo(const Type *Ty) const;


  /// Here are some useful little methods to query what type derived types are
  /// Note that all other types can just compare to see if this == Type::xxxTy;
  ///
  inline bool isPrimitiveType() const { return ID <= LastPrimitiveTyID; }
  inline bool isDerivedType()   const { return ID >= FirstDerivedTyID; }

  /// isFirstClassType - Return true if the type is "first class", meaning it
  /// is a valid type for a Value.
  ///
  inline bool isFirstClassType() const {
    // There are more first-class kinds than non-first-class kinds, so a
    // negative test is simpler than a positive one.
    return ID != FunctionTyID && ID != VoidTyID && ID != OpaqueTyID;
  }

  /// isSingleValueType - Return true if the type is a valid type for a
  /// virtual register in codegen.  This includes all first-class types
  /// except struct and array types.
  ///
  inline bool isSingleValueType() const {
    return (ID != VoidTyID && ID <= LastPrimitiveTyID) ||
            ID == IntegerTyID || ID == PointerTyID || ID == VectorTyID;
  }

  /// isAggregateType - Return true if the type is an aggregate type. This
  /// means it is valid as the first operand of an insertvalue or
  /// extractvalue instruction. This includes struct and array types, but
  /// does not include vector types.
  ///
  inline bool isAggregateType() const {
    return ID == StructTyID || ID == ArrayTyID || ID == UnionTyID;
  }

  /// isSized - Return true if it makes sense to take the size of this type.  To
  /// get the actual size for a particular target, it is reasonable to use the
  /// TargetData subsystem to do this.
  ///
  bool isSized() const {
    // If it's a primitive, it is always sized.
    if (ID == IntegerTyID || isFloatingPointTy() || ID == PointerTyID)
      return true;
    // If it is not something that can have a size (e.g. a function or label),
    // it doesn't have a size.
    if (ID != StructTyID && ID != ArrayTyID && ID != VectorTyID &&
        ID != UnionTyID)
      return false;
    // If it is something that can have a size and it's concrete, it definitely
    // has a size, otherwise we have to try harder to decide.
    return !isAbstract() || isSizedDerivedType();
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

  /// getForwardedType - Return the type that this type has been resolved to if
  /// it has been resolved to anything.  This is used to implement the
  /// union-find algorithm for type resolution, and shouldn't be used by general
  /// purpose clients.
  const Type *getForwardedType() const {
    if (!ForwardType) return 0;
    return getForwardedTypeInternal();
  }

  /// getVAArgsPromotedType - Return the type an argument of this type
  /// will be promoted to if passed through a variable argument
  /// function.
  const Type *getVAArgsPromotedType(LLVMContext &C) const; 

  /// getScalarType - If this is a vector type, return the element type,
  /// otherwise return this.
  const Type *getScalarType() const;

  //===--------------------------------------------------------------------===//
  // Type Iteration support
  //
  typedef PATypeHandle *subtype_iterator;
  subtype_iterator subtype_begin() const { return ContainedTys; }
  subtype_iterator subtype_end() const { return &ContainedTys[NumContainedTys];}

  /// getContainedType - This method is used to implement the type iterator
  /// (defined a the end of the file).  For derived types, this returns the
  /// types 'contained' in the derived type.
  ///
  const Type *getContainedType(unsigned i) const {
    assert(i < NumContainedTys && "Index out of range!");
    return ContainedTys[i].get();
  }

  /// getNumContainedTypes - Return the number of types in the derived type.
  ///
  unsigned getNumContainedTypes() const { return NumContainedTys; }

  //===--------------------------------------------------------------------===//
  // Static members exported by the Type class itself.  Useful for getting
  // instances of Type.
  //

  /// getPrimitiveType - Return a type based on an identifier.
  static const Type *getPrimitiveType(LLVMContext &C, TypeID IDNumber);

  //===--------------------------------------------------------------------===//
  // These are the builtin types that are always available...
  //
  static const Type *getVoidTy(LLVMContext &C);
  static const Type *getLabelTy(LLVMContext &C);
  static const Type *getFloatTy(LLVMContext &C);
  static const Type *getDoubleTy(LLVMContext &C);
  static const Type *getMetadataTy(LLVMContext &C);
  static const Type *getX86_FP80Ty(LLVMContext &C);
  static const Type *getFP128Ty(LLVMContext &C);
  static const Type *getPPC_FP128Ty(LLVMContext &C);
  static const IntegerType *getInt1Ty(LLVMContext &C);
  static const IntegerType *getInt8Ty(LLVMContext &C);
  static const IntegerType *getInt16Ty(LLVMContext &C);
  static const IntegerType *getInt32Ty(LLVMContext &C);
  static const IntegerType *getInt64Ty(LLVMContext &C);

  //===--------------------------------------------------------------------===//
  // Convenience methods for getting pointer types with one of the above builtin
  // types as pointee.
  //
  static const PointerType *getFloatPtrTy(LLVMContext &C, unsigned AS = 0);
  static const PointerType *getDoublePtrTy(LLVMContext &C, unsigned AS = 0);
  static const PointerType *getX86_FP80PtrTy(LLVMContext &C, unsigned AS = 0);
  static const PointerType *getFP128PtrTy(LLVMContext &C, unsigned AS = 0);
  static const PointerType *getPPC_FP128PtrTy(LLVMContext &C, unsigned AS = 0);
  static const PointerType *getInt1PtrTy(LLVMContext &C, unsigned AS = 0);
  static const PointerType *getInt8PtrTy(LLVMContext &C, unsigned AS = 0);
  static const PointerType *getInt16PtrTy(LLVMContext &C, unsigned AS = 0);
  static const PointerType *getInt32PtrTy(LLVMContext &C, unsigned AS = 0);
  static const PointerType *getInt64PtrTy(LLVMContext &C, unsigned AS = 0);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Type *) { return true; }

  void addRef() const {
    assert(isAbstract() && "Cannot add a reference to a non-abstract type!");
    ++RefCount;
  }

  void dropRef() const {
    assert(isAbstract() && "Cannot drop a reference to a non-abstract type!");
    assert(RefCount && "No objects are currently referencing this object!");

    // If this is the last PATypeHolder using this object, and there are no
    // PATypeHandles using it, the type is dead, delete it now.
    if (--RefCount == 0 && AbstractTypeUsers.empty())
      this->destroy();
  }
  
  /// addAbstractTypeUser - Notify an abstract type that there is a new user of
  /// it.  This function is called primarily by the PATypeHandle class.
  ///
  void addAbstractTypeUser(AbstractTypeUser *U) const;
  
  /// removeAbstractTypeUser - Notify an abstract type that a user of the class
  /// no longer has a handle to the type.  This function is called primarily by
  /// the PATypeHandle class.  When there are no users of the abstract type, it
  /// is annihilated, because there is no way to get a reference to it ever
  /// again.
  ///
  void removeAbstractTypeUser(AbstractTypeUser *U) const;

  /// getPointerTo - Return a pointer to the current type.  This is equivalent
  /// to PointerType::get(Foo, AddrSpace).
  const PointerType *getPointerTo(unsigned AddrSpace = 0) const;

private:
  /// isSizedDerivedType - Derived types like structures and arrays are sized
  /// iff all of the members of the type are sized as well.  Since asking for
  /// their size is relatively uncommon, move this operation out of line.
  bool isSizedDerivedType() const;

  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

protected:
  // PromoteAbstractToConcrete - This is an internal method used to calculate
  // change "Abstract" from true to false when types are refined.
  void PromoteAbstractToConcrete();
  friend class TypeMapBase;
};

//===----------------------------------------------------------------------===//
// Define some inline methods for the AbstractTypeUser.h:PATypeHandle class.
// These are defined here because they MUST be inlined, yet are dependent on
// the definition of the Type class.
//
inline void PATypeHandle::addUser() {
  assert(Ty && "Type Handle has a null type!");
  if (Ty->isAbstract())
    Ty->addAbstractTypeUser(User);
}
inline void PATypeHandle::removeUser() {
  if (Ty->isAbstract())
    Ty->removeAbstractTypeUser(User);
}

// Define inline methods for PATypeHolder.

/// get - This implements the forwarding part of the union-find algorithm for
/// abstract types.  Before every access to the Type*, we check to see if the
/// type we are pointing to is forwarding to a new type.  If so, we drop our
/// reference to the type.
///
inline Type* PATypeHolder::get() const {
  const Type *NewTy = Ty->getForwardedType();
  if (!NewTy) return const_cast<Type*>(Ty);
  return *const_cast<PATypeHolder*>(this) = NewTy;
}

inline void PATypeHolder::addRef() {
  assert(Ty && "Type Holder has a null type!");
  if (Ty->isAbstract())
    Ty->addRef();
}

inline void PATypeHolder::dropRef() {
  if (Ty->isAbstract())
    Ty->dropRef();
}


//===----------------------------------------------------------------------===//
// Provide specializations of GraphTraits to be able to treat a type as a
// graph of sub types...

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

template <> inline bool isa_impl<PointerType, Type>(const Type &Ty) {
  return Ty.getTypeID() == Type::PointerTyID;
}

raw_ostream &operator<<(raw_ostream &OS, const Type &T);

} // End llvm namespace

#endif
