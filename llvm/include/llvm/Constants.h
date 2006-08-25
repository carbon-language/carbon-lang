//===-- llvm/Constants.h - Constant class subclass definitions --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file This file contains the declarations for the subclasses of Constant, 
/// which represent the different flavors of constant values that live in LLVM.
/// Note that Constants are immutable (once created they never change) and are 
/// fully shared by structural equivalence.  This means that two structurally
/// equivalent constants will always have the same address.  Constant's are
/// created on demand as needed and never deleted: thus clients don't have to
/// worry about the lifetime of the objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONSTANTS_H
#define LLVM_CONSTANTS_H

#include "llvm/Constant.h"
#include "llvm/Type.h"

namespace llvm {

class ArrayType;
class StructType;
class PointerType;
class PackedType;

template<class ConstantClass, class TypeClass, class ValType>
struct ConstantCreator;
template<class ConstantClass, class TypeClass>
struct ConvertConstantType;

//===----------------------------------------------------------------------===//
/// This is the shared superclass of boolean and integer constants. This class 
/// just defines some common interfaces to be implemented by the subclasses.
/// @brief An abstract class for integer constants.
class ConstantIntegral : public Constant {
protected:
  union {
    int64_t  Signed;
    uint64_t Unsigned;
  } Val;
  ConstantIntegral(const Type *Ty, ValueTy VT, uint64_t V);
public:

  /// @brief Return the raw value of the constant as a 64-bit integer value.
  inline uint64_t getRawValue() const { return Val.Unsigned; }
  
  /// Return the constant as a 64-bit unsigned integer value after it
  /// has been zero extended as appropriate for the type of this constant.
  /// @brief Return the zero extended value.
  inline uint64_t getZExtValue() const {
    unsigned Size = getType()->getPrimitiveSizeInBits();
    return Val.Unsigned & (~uint64_t(0UL) >> (64-Size));
  }

  /// Return the constant as a 64-bit integer value after it has been sign
  /// sign extended as appropriate for the type of this constant.
  /// @brief REturn the sign extended value.
  inline int64_t getSExtValue() const {
    unsigned Size = getType()->getPrimitiveSizeInBits();
    return (Val.Signed << (64-Size)) >> (64-Size);
  }
  
  /// This function is implemented by subclasses and will return true iff this
  /// constant represents the the "null" value that would be returned by the
  /// getNullValue method.
  /// @returns true if the constant's value is 0.
  /// @brief Determine if the value is null.
  virtual bool isNullValue() const = 0;

  /// This function is implemented by sublcasses and will return true iff this
  /// constant represents the the largest value that may be represented by this
  /// constant's type.
  /// @returns true if the constant's value is maximal.
  /// @brief Determine if the value is maximal.
  virtual bool isMaxValue() const = 0;

  /// This function is implemented by subclasses and will return true iff this 
  /// constant represents the smallest value that may be represented by this 
  /// constant's type.
  /// @returns true if the constant's value is minimal
  /// @brief Determine if the value is minimal.
  virtual bool isMinValue() const = 0;

  /// This function is implemented by subclasses and will return true iff every
  /// bit in this constant is set to true.
  /// @returns true if all bits of the constant are ones.
  /// @brief Determine if the value is all ones.
  virtual bool isAllOnesValue() const = 0;

  /// @returns the largest value for an integer constant of the given type 
  /// @brief Get the maximal value
  static ConstantIntegral *getMaxValue(const Type *Ty);

  /// @returns the smallest value for an integer constant of the given type 
  /// @brief Get the minimal value
  static ConstantIntegral *getMinValue(const Type *Ty);

  /// @returns the value for an integer constant of the given type that has all
  /// its bits set to true.
  /// @brief Get the all ones value
  static ConstantIntegral *getAllOnesValue(const Type *Ty);

  /// Methods to support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantIntegral *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantBoolVal ||
           V->getValueType() == ConstantSIntVal ||
           V->getValueType() == ConstantUIntVal;
  }
};


//===----------------------------------------------------------------------===//
/// This concrete class represents constant values of type BoolTy. There are 
/// only two instances of this class constructed: the True and False static 
/// members. The constructor is hidden to ensure this invariant.
/// @brief Constant Boolean class
class ConstantBool : public ConstantIntegral {
  ConstantBool(bool V);
public:
  static ConstantBool *True, *False;  ///< The True & False values

  /// This method is provided mostly for compatibility with the other 
  /// ConstantIntegral subclasses.
  /// @brief Static factory method for getting a ConstantBool instance.
  static ConstantBool *get(bool Value) { return Value ? True : False; }

  /// This method is provided mostly for compatibility with the other 
  /// ConstantIntegral subclasses.
  /// @brief Static factory method for getting a ConstantBool instance.
  static ConstantBool *get(const Type *Ty, bool Value) { return get(Value); }

  /// Returns the opposite value of this ConstantBool value.
  /// @brief Get inverse value.
  inline ConstantBool *inverted() const { return (this==True) ? False : True; }

  /// @returns the value of this ConstantBool
  /// @brief return the boolean value of this constant.
  inline bool getValue() const { return static_cast<bool>(getRawValue()); }

  /// @see ConstantIntegral for details
  /// @brief Implement overrides
  virtual bool isNullValue() const { return this == False; }
  virtual bool isMaxValue() const { return this == True; }
  virtual bool isMinValue() const { return this == False; }
  virtual bool isAllOnesValue() const { return this == True; }

  /// @brief Methods to support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantBool *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantBoolVal;
  }
};


//===----------------------------------------------------------------------===//
/// This is the abstract superclass of ConstantSInt & ConstantUInt, to make 
/// dealing with integral constants easier when sign is irrelevant.
/// @brief Abstract clas for constant integers.
class ConstantInt : public ConstantIntegral {
protected:
  ConstantInt(const ConstantInt &);      // DO NOT IMPLEMENT
  ConstantInt(const Type *Ty, ValueTy VT, uint64_t V);
public:
  /// A helper method that can be used to determine if the constant contained 
  /// within is equal to a constant.  This only works for very small values, 
  /// because this is all that can be represented with all types.
  /// @brief Determine if this constant's value is same as an unsigned char.
  bool equalsInt(unsigned char V) const {
    assert(V <= 127 &&
           "equalsInt: Can only be used with very small positive constants!");
    return Val.Unsigned == V;
  }

  /// Return a ConstantInt with the specified value for the specified type. 
  /// This only works for very small values, because this is all that can be 
  /// represented with all types integer types.
  /// @brief Get a ConstantInt for a specific value.
  static ConstantInt *get(const Type *Ty, unsigned char V);

  /// @returns true if this is the null integer value.
  /// @see ConstantIntegral for details
  /// @brief Implement override.
  virtual bool isNullValue() const { return Val.Unsigned == 0; }

  /// @brief Methods to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const ConstantInt *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantSIntVal ||
           V->getValueType() == ConstantUIntVal;
  }
};


//===----------------------------------------------------------------------===//
/// A concrete class to represent constant signed integer values for the types
/// sbyte, short, int, and long.
/// @brief Constant Signed Integer Class.
class ConstantSInt : public ConstantInt {
  ConstantSInt(const ConstantSInt &);      // DO NOT IMPLEMENT
  friend struct ConstantCreator<ConstantSInt, Type, int64_t>;

protected:
  ConstantSInt(const Type *Ty, int64_t V);
public:
  /// This static factory methods returns objects of the specified value. Note
  /// that repeated calls with the same operands return the same object.
  /// @returns A ConstantSInt instant for the type and value requested.
  /// @brief Get a signed integer constant.
  static ConstantSInt *get(
    const Type *Ty,  ///< The type of constant (SByteTy, IntTy, ShortTy, LongTy)
    int64_t V        ///< The value for the constant integer.
  );

  /// This static method returns true if the type Ty is big enough to 
  /// represent the value V. This can be used to avoid having the get method 
  /// assert when V is larger than Ty can represent.
  /// @returns true if V is a valid value for type Ty
  /// @brief Determine if the value is in range for the given type.
  static bool isValueValidForType(const Type *Ty, int64_t V);

  /// @returns the underlying value of this constant.
  /// @brief Get the constant value.
  inline int64_t getValue() const { return Val.Signed; }

  /// @returns true iff this constant's bits are all set to true.
  /// @see ConstantIntegral
  /// @brief Override implementation
  virtual bool isAllOnesValue() const { return getValue() == -1; }

  /// @returns true iff this is the largest value that may be represented 
  /// by this type.
  /// @see ConstantIntegeral
  /// @brief Override implementation
  virtual bool isMaxValue() const {
    int64_t V = getValue();
    if (V < 0) return false;    // Be careful about wrap-around on 'long's
    ++V;
    return !isValueValidForType(getType(), V) || V < 0;
  }

  /// @returns true if this is the smallest value that may be represented by 
  /// this type.
  /// @see ConstantIntegral
  /// @brief Override implementation
  virtual bool isMinValue() const {
    int64_t V = getValue();
    if (V > 0) return false;    // Be careful about wrap-around on 'long's
    --V;
    return !isValueValidForType(getType(), V) || V > 0;
  }

  /// @brief Methods to support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantSInt *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantSIntVal;
  }
};

//===----------------------------------------------------------------------===//
/// A concrete class that represents constant unsigned integer values of type
/// Type::UByteTy, Type::UShortTy, Type::UIntTy, or Type::ULongTy.
/// @brief Constant Unsigned Integer Class
class ConstantUInt : public ConstantInt {
  ConstantUInt(const ConstantUInt &);      // DO NOT IMPLEMENT
  friend struct ConstantCreator<ConstantUInt, Type, uint64_t>;
protected:
  ConstantUInt(const Type *Ty, uint64_t V);
public:
  /// get() - Static factory methods - Return objects of the specified value
  ///
  static ConstantUInt *get(const Type *Ty, uint64_t V);

  /// isValueValidForType - return true if Ty is big enough to represent V.
  ///
  static bool isValueValidForType(const Type *Ty, uint64_t V);

  /// getValue - return the underlying value of this constant.
  ///
  inline uint64_t getValue() const { return Val.Unsigned; }

  /// isMaxValue - Return true if this is the largest value that may be
  /// represented by this type.
  ///
  virtual bool isAllOnesValue() const;
  virtual bool isMaxValue() const { return isAllOnesValue(); }
  virtual bool isMinValue() const { return getValue() == 0; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantUInt *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantUIntVal;
  }
};


//===----------------------------------------------------------------------===//
/// ConstantFP - Floating Point Values [float, double]
///
class ConstantFP : public Constant {
  double Val;
  friend struct ConstantCreator<ConstantFP, Type, uint64_t>;
  friend struct ConstantCreator<ConstantFP, Type, uint32_t>;
  ConstantFP(const ConstantFP &);      // DO NOT IMPLEMENT
protected:
  ConstantFP(const Type *Ty, double V);
public:
  /// get() - Static factory methods - Return objects of the specified value
  static ConstantFP *get(const Type *Ty, double V);

  /// isValueValidForType - return true if Ty is big enough to represent V.
  static bool isValueValidForType(const Type *Ty, double V);
  inline double getValue() const { return Val; }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  Don't depend on == for doubles to tell us it's zero, it
  /// considers -0.0 to be null as well as 0.0.  :(
  virtual bool isNullValue() const;

  /// isExactlyValue - We don't rely on operator== working on double values, as
  /// it returns true for things that are clearly not equal, like -0.0 and 0.0.
  /// As such, this method can be used to do an exact bit-for-bit comparison of
  /// two floating point values.
  bool isExactlyValue(double V) const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantFP *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantFPVal;
  }
};

//===----------------------------------------------------------------------===//
/// ConstantAggregateZero - All zero aggregate value
///
class ConstantAggregateZero : public Constant {
  friend struct ConstantCreator<ConstantAggregateZero, Type, char>;
  ConstantAggregateZero(const ConstantAggregateZero &);      // DO NOT IMPLEMENT
protected:
  ConstantAggregateZero(const Type *Ty)
    : Constant(Ty, ConstantAggregateZeroVal, 0, 0) {}
public:
  /// get() - static factory method for creating a null aggregate.  It is
  /// illegal to call this method with a non-aggregate type.
  static Constant *get(const Type *Ty);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return true; }

  virtual void destroyConstant();

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  ///
  static bool classof(const ConstantAggregateZero *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantAggregateZeroVal;
  }
};


//===----------------------------------------------------------------------===//
/// ConstantArray - Constant Array Declarations
///
class ConstantArray : public Constant {
  friend struct ConstantCreator<ConstantArray, ArrayType,
                                    std::vector<Constant*> >;
  ConstantArray(const ConstantArray &);      // DO NOT IMPLEMENT
protected:
  ConstantArray(const ArrayType *T, const std::vector<Constant*> &Val);
  ~ConstantArray();
public:
  /// get() - Static factory methods - Return objects of the specified value
  static Constant *get(const ArrayType *T, const std::vector<Constant*> &);

  /// This method constructs a ConstantArray and initializes it with a text
  /// string. The default behavior (AddNull==true) causes a null terminator to
  /// be placed at the end of the array. This effectively increases the length
  /// of the array by one (you've been warned).  However, in some situations 
  /// this is not desired so if AddNull==false then the string is copied without
  /// null termination. 
  static Constant *get(const std::string &Initializer, bool AddNull = true);

  /// getType - Specialize the getType() method to always return an ArrayType,
  /// which reduces the amount of casting needed in parts of the compiler.
  ///
  inline const ArrayType *getType() const {
    return reinterpret_cast<const ArrayType*>(Value::getType());
  }

  /// isString - This method returns true if the array is an array of sbyte or
  /// ubyte, and if the elements of the array are all ConstantInt's.
  bool isString() const;

  /// getAsString - If this array is isString(), then this method converts the
  /// array to an std::string and returns it.  Otherwise, it asserts out.
  ///
  std::string getAsString() const;

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because zero arrays are always
  /// created as ConstantAggregateZero objects.
  virtual bool isNullValue() const { return false; }

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantArray *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantArrayVal;
  }
};


//===----------------------------------------------------------------------===//
// ConstantStruct - Constant Struct Declarations
//
class ConstantStruct : public Constant {
  friend struct ConstantCreator<ConstantStruct, StructType,
                                    std::vector<Constant*> >;
  ConstantStruct(const ConstantStruct &);      // DO NOT IMPLEMENT
protected:
  ConstantStruct(const StructType *T, const std::vector<Constant*> &Val);
  ~ConstantStruct();
public:
  /// get() - Static factory methods - Return objects of the specified value
  ///
  static Constant *get(const StructType *T, const std::vector<Constant*> &V);
  static Constant *get(const std::vector<Constant*> &V);

  /// getType() specialization - Reduce amount of casting...
  ///
  inline const StructType *getType() const {
    return reinterpret_cast<const StructType*>(Value::getType());
  }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because zero structs are always
  /// created as ConstantAggregateZero objects.
  virtual bool isNullValue() const {
    return false;
  }

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantStruct *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantStructVal;
  }
};

//===----------------------------------------------------------------------===//
/// ConstantPacked - Constant Packed Declarations
///
class ConstantPacked : public Constant {
  friend struct ConstantCreator<ConstantPacked, PackedType,
                                    std::vector<Constant*> >;
  ConstantPacked(const ConstantPacked &);      // DO NOT IMPLEMENT
protected:
  ConstantPacked(const PackedType *T, const std::vector<Constant*> &Val);
  ~ConstantPacked();
public:
  /// get() - Static factory methods - Return objects of the specified value
  static Constant *get(const PackedType *T, const std::vector<Constant*> &);
  static Constant *get(const std::vector<Constant*> &V);

  /// getType - Specialize the getType() method to always return an PackedType,
  /// which reduces the amount of casting needed in parts of the compiler.
  ///
  inline const PackedType *getType() const {
    return reinterpret_cast<const PackedType*>(Value::getType());
  }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because zero arrays are always
  /// created as ConstantAggregateZero objects.
  virtual bool isNullValue() const { return false; }

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantPacked *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantPackedVal;
  }
};

//===----------------------------------------------------------------------===//
/// ConstantPointerNull - a constant pointer value that points to null
///
class ConstantPointerNull : public Constant {
  friend struct ConstantCreator<ConstantPointerNull, PointerType, char>;
  ConstantPointerNull(const ConstantPointerNull &);      // DO NOT IMPLEMENT
protected:
  ConstantPointerNull(const PointerType *T)
    : Constant(reinterpret_cast<const Type*>(T),
               Value::ConstantPointerNullVal, 0, 0) {}

public:

  /// get() - Static factory methods - Return objects of the specified value
  static ConstantPointerNull *get(const PointerType *T);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return true; }

  virtual void destroyConstant();

  /// getType - Specialize the getType() method to always return an PointerType,
  /// which reduces the amount of casting needed in parts of the compiler.
  ///
  inline const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(Value::getType());
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantPointerNull *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantPointerNullVal;
  }
};


/// ConstantExpr - a constant value that is initialized with an expression using
/// other constant values.
///
/// This class uses the standard Instruction opcodes to define the various
/// constant expressions.  The Opcode field for the ConstantExpr class is
/// maintained in the Value::SubclassData field.
class ConstantExpr : public Constant {
  friend struct ConstantCreator<ConstantExpr,Type,
                            std::pair<unsigned, std::vector<Constant*> > >;
  friend struct ConvertConstantType<ConstantExpr, Type>;

protected:
  ConstantExpr(const Type *Ty, unsigned Opcode, Use *Ops, unsigned NumOps)
    : Constant(Ty, ConstantExprVal, Ops, NumOps) {
    // Operation type (an Instruction opcode) is stored as the SubclassData.
    SubclassData = Opcode;
  }

  // These private methods are used by the type resolution code to create
  // ConstantExprs in intermediate forms.
  static Constant *getTy(const Type *Ty, unsigned Opcode,
                         Constant *C1, Constant *C2);
  static Constant *getShiftTy(const Type *Ty,
                              unsigned Opcode, Constant *C1, Constant *C2);
  static Constant *getSelectTy(const Type *Ty,
                               Constant *C1, Constant *C2, Constant *C3);
  static Constant *getGetElementPtrTy(const Type *Ty, Constant *C,
                                      const std::vector<Value*> &IdxList);
  static Constant *getExtractElementTy(const Type *Ty, Constant *Val,
                                       Constant *Idx);
  static Constant *getInsertElementTy(const Type *Ty, Constant *Val,
                                      Constant *Elt, Constant *Idx);
  static Constant *getShuffleVectorTy(const Type *Ty, Constant *V1,
                                      Constant *V2, Constant *Mask);

public:
  // Static methods to construct a ConstantExpr of different kinds.  Note that
  // these methods may return a object that is not an instance of the
  // ConstantExpr class, because they will attempt to fold the constant
  // expression into something simpler if possible.

  /// Cast constant expr
  ///
  static Constant *getCast(Constant *C, const Type *Ty);
  static Constant *getSignExtend(Constant *C, const Type *Ty);
  static Constant *getZeroExtend(Constant *C, const Type *Ty);

  /// Select constant expr
  ///
  static Constant *getSelect(Constant *C, Constant *V1, Constant *V2) {
    return getSelectTy(V1->getType(), C, V1, V2);
  }

  /// getSizeOf constant expr - computes the size of a type in a target
  /// independent way (Note: the return type is ULong but the object is not
  /// necessarily a ConstantUInt).
  ///
  static Constant *getSizeOf(const Type *Ty);

  /// getPtrPtrFromArrayPtr constant expr - given a pointer to a constant array,
  /// return a pointer to a pointer of the array element type.
  static Constant *getPtrPtrFromArrayPtr(Constant *C);

  /// ConstantExpr::get - Return a binary or shift operator constant expression,
  /// folding if possible.
  ///
  static Constant *get(unsigned Opcode, Constant *C1, Constant *C2);

  /// ConstantExpr::get* - Return some common constants without having to
  /// specify the full Instruction::OPCODE identifier.
  ///
  static Constant *getNeg(Constant *C);
  static Constant *getNot(Constant *C);
  static Constant *getAdd(Constant *C1, Constant *C2);
  static Constant *getSub(Constant *C1, Constant *C2);
  static Constant *getMul(Constant *C1, Constant *C2);
  static Constant *getDiv(Constant *C1, Constant *C2);
  static Constant *getRem(Constant *C1, Constant *C2);
  static Constant *getAnd(Constant *C1, Constant *C2);
  static Constant *getOr(Constant *C1, Constant *C2);
  static Constant *getXor(Constant *C1, Constant *C2);
  static Constant *getSetEQ(Constant *C1, Constant *C2);
  static Constant *getSetNE(Constant *C1, Constant *C2);
  static Constant *getSetLT(Constant *C1, Constant *C2);
  static Constant *getSetGT(Constant *C1, Constant *C2);
  static Constant *getSetLE(Constant *C1, Constant *C2);
  static Constant *getSetGE(Constant *C1, Constant *C2);
  static Constant *getShl(Constant *C1, Constant *C2);
  static Constant *getShr(Constant *C1, Constant *C2);

  static Constant *getUShr(Constant *C1, Constant *C2); // unsigned shr
  static Constant *getSShr(Constant *C1, Constant *C2); // signed shr

  /// Getelementptr form.  std::vector<Value*> is only accepted for convenience:
  /// all elements must be Constant's.
  ///
  static Constant *getGetElementPtr(Constant *C,
                                    const std::vector<Constant*> &IdxList);
  static Constant *getGetElementPtr(Constant *C,
                                    const std::vector<Value*> &IdxList);

  static Constant *getExtractElement(Constant *Vec, Constant *Idx);
  static Constant *getInsertElement(Constant *Vec, Constant *Elt,Constant *Idx);
  static Constant *getShuffleVector(Constant *V1, Constant *V2, Constant *Mask);
  
  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return false; }

  /// getOpcode - Return the opcode at the root of this constant expression
  unsigned getOpcode() const { return SubclassData; }

  /// getOpcodeName - Return a string representation for an opcode.
  const char *getOpcodeName() const;

  /// getWithOperandReplaced - Return a constant expression identical to this
  /// one, but with the specified operand set to the specified value.
  Constant *getWithOperandReplaced(unsigned OpNo, Constant *Op) const;
  
  /// getWithOperands - This returns the current constant expression with the
  /// operands replaced with the specified values.  The specified operands must
  /// match count and type with the existing ones.
  Constant *getWithOperands(const std::vector<Constant*> &Ops) const;
  
  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  /// Override methods to provide more type information...
  inline Constant *getOperand(unsigned i) {
    return cast<Constant>(User::getOperand(i));
  }
  inline Constant *getOperand(unsigned i) const {
    return const_cast<Constant*>(cast<Constant>(User::getOperand(i)));
  }


  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantExpr *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == ConstantExprVal;
  }
};


//===----------------------------------------------------------------------===//
/// UndefValue - 'undef' values are things that do not have specified contents.
/// These are used for a variety of purposes, including global variable
/// initializers and operands to instructions.  'undef' values can occur with
/// any type.
///
class UndefValue : public Constant {
  friend struct ConstantCreator<UndefValue, Type, char>;
  UndefValue(const UndefValue &);      // DO NOT IMPLEMENT
protected:
  UndefValue(const Type *T) : Constant(T, UndefValueVal, 0, 0) {}
public:
  /// get() - Static factory methods - Return an 'undef' object of the specified
  /// type.
  ///
  static UndefValue *get(const Type *T);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return false; }

  virtual void destroyConstant();

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const UndefValue *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == UndefValueVal;
  }
};

} // End llvm namespace

#endif
