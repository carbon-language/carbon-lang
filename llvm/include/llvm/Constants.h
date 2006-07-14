//===-- llvm/Constants.h - Constant class subclass definitions --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for the subclasses of Constant, which
// represent the different flavors of constant values that live in LLVM.  Note
// that Constants are immutable (once created they never change) and are fully
// shared by structural equivalence.  This means that two structurally
// equivalent constants will always have the same address.  Constant's are
// created on demand as needed and never deleted: thus clients don't have to
// worry about the lifetime of the objects.
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
/// ConstantIntegral - Shared superclass of boolean and integer constants.
///
/// This class just defines some common interfaces to be implemented.
///
class ConstantIntegral : public Constant {
protected:
  union {
    int64_t  Signed;
    uint64_t Unsigned;
  } Val;
  ConstantIntegral(const Type *Ty, ValueTy VT, uint64_t V);
public:

  /// getRawValue - return the underlying value of this constant as a 64-bit
  /// unsigned integer value.
  ///
  inline uint64_t getRawValue() const { return Val.Unsigned; }
  
  /// getZExtValue - Return the constant zero extended as appropriate for this
  /// type.
  inline uint64_t getZExtValue() const {
    unsigned Size = getType()->getPrimitiveSizeInBits();
    return Val.Unsigned & (~uint64_t(0UL) >> (64-Size));
  }

  /// getSExtValue - Return the constant sign extended as appropriate for this
  /// type.
  inline int64_t getSExtValue() const {
    unsigned Size = getType()->getPrimitiveSizeInBits();
    return (Val.Signed << (64-Size)) >> (64-Size);
  }
  
  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  ///
  virtual bool isNullValue() const = 0;

  /// isMaxValue - Return true if this is the largest value that may be
  /// represented by this type.
  ///
  virtual bool isMaxValue() const = 0;

  /// isMinValue - Return true if this is the smallest value that may be
  /// represented by this type.
  ///
  virtual bool isMinValue() const = 0;

  /// isAllOnesValue - Return true if every bit in this constant is set to true.
  ///
  virtual bool isAllOnesValue() const = 0;

  /// Static constructor to get the maximum/minimum/allones constant of
  /// specified (integral) type...
  ///
  static ConstantIntegral *getMaxValue(const Type *Ty);
  static ConstantIntegral *getMinValue(const Type *Ty);
  static ConstantIntegral *getAllOnesValue(const Type *Ty);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantIntegral *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantBoolVal ||
           V->getValueType() == ConstantSIntVal ||
           V->getValueType() == ConstantUIntVal;
  }
};


//===----------------------------------------------------------------------===//
/// ConstantBool - Boolean Values
///
class ConstantBool : public ConstantIntegral {
  ConstantBool(bool V);
public:
  static ConstantBool *True, *False;  // The True & False values

  /// get() - Static factory methods - Return objects of the specified value
  static ConstantBool *get(bool Value) { return Value ? True : False; }
  static ConstantBool *get(const Type *Ty, bool Value) { return get(Value); }

  /// inverted - Return the opposite value of the current value.
  inline ConstantBool *inverted() const { return (this==True) ? False : True; }

  /// getValue - return the boolean value of this constant.
  ///
  inline bool getValue() const { return static_cast<bool>(getRawValue()); }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  ///
  virtual bool isNullValue() const { return this == False; }
  virtual bool isMaxValue() const { return this == True; }
  virtual bool isMinValue() const { return this == False; }
  virtual bool isAllOnesValue() const { return this == True; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantBool *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantBoolVal;
  }
};


//===----------------------------------------------------------------------===//
/// ConstantInt - Superclass of ConstantSInt & ConstantUInt, to make dealing
/// with integral constants easier.
///
class ConstantInt : public ConstantIntegral {
protected:
  ConstantInt(const ConstantInt &);      // DO NOT IMPLEMENT
  ConstantInt(const Type *Ty, ValueTy VT, uint64_t V);
public:
  /// equalsInt - Provide a helper method that can be used to determine if the
  /// constant contained within is equal to a constant.  This only works for
  /// very small values, because this is all that can be represented with all
  /// types.
  ///
  bool equalsInt(unsigned char V) const {
    assert(V <= 127 &&
           "equalsInt: Can only be used with very small positive constants!");
    return Val.Unsigned == V;
  }

  /// ConstantInt::get static method: return a ConstantInt with the specified
  /// value.  as above, we work only with very small values here.
  ///
  static ConstantInt *get(const Type *Ty, unsigned char V);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return Val.Unsigned == 0; }
  virtual bool isMaxValue() const = 0;
  virtual bool isMinValue() const = 0;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantInt *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantSIntVal ||
           V->getValueType() == ConstantUIntVal;
  }
};


//===----------------------------------------------------------------------===//
/// ConstantSInt - Signed Integer Values [sbyte, short, int, long]
///
class ConstantSInt : public ConstantInt {
  ConstantSInt(const ConstantSInt &);      // DO NOT IMPLEMENT
  friend struct ConstantCreator<ConstantSInt, Type, int64_t>;

protected:
  ConstantSInt(const Type *Ty, int64_t V);
public:
  /// get() - Static factory methods - Return objects of the specified value
  ///
  static ConstantSInt *get(const Type *Ty, int64_t V);

  /// isValueValidForType - return true if Ty is big enough to represent V.
  ///
  static bool isValueValidForType(const Type *Ty, int64_t V);

  /// getValue - return the underlying value of this constant.
  ///
  inline int64_t getValue() const { return Val.Signed; }

  virtual bool isAllOnesValue() const { return getValue() == -1; }

  /// isMaxValue - Return true if this is the largest value that may be
  /// represented by this type.
  ///
  virtual bool isMaxValue() const {
    int64_t V = getValue();
    if (V < 0) return false;    // Be careful about wrap-around on 'long's
    ++V;
    return !isValueValidForType(getType(), V) || V < 0;
  }

  /// isMinValue - Return true if this is the smallest value that may be
  /// represented by this type.
  ///
  virtual bool isMinValue() const {
    int64_t V = getValue();
    if (V > 0) return false;    // Be careful about wrap-around on 'long's
    --V;
    return !isValueValidForType(getType(), V) || V > 0;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  ///
  static inline bool classof(const ConstantSInt *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantSIntVal;
  }
};

//===----------------------------------------------------------------------===//
/// ConstantUInt - Unsigned Integer Values [ubyte, ushort, uint, ulong]
///
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
