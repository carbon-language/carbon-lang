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
class VectorType;

template<class ConstantClass, class TypeClass, class ValType>
struct ConstantCreator;
template<class ConstantClass, class TypeClass>
struct ConvertConstantType;

//===----------------------------------------------------------------------===//
/// This is the shared class of boolean and integer constants. This class 
/// represents both boolean and integral constants.
/// @brief Class for constant integers.
class ConstantInt : public Constant {
  static ConstantInt *TheTrueVal, *TheFalseVal;
protected:
  uint64_t Val;
protected:
  ConstantInt(const ConstantInt &);      // DO NOT IMPLEMENT
  ConstantInt(const IntegerType *Ty, uint64_t V);
  friend struct ConstantCreator<ConstantInt, IntegerType, uint64_t>;
public:
  /// Return the constant as a 64-bit unsigned integer value after it
  /// has been zero extended as appropriate for the type of this constant.
  /// @brief Return the zero extended value.
  inline uint64_t getZExtValue() const {
    return Val;
  }

  /// Return the constant as a 64-bit integer value after it has been sign
  /// sign extended as appropriate for the type of this constant.
  /// @brief Return the sign extended value.
  inline int64_t getSExtValue() const {
    unsigned Size = Value::getType()->getPrimitiveSizeInBits();
    return (int64_t(Val) << (64-Size)) >> (64-Size);
  }
  /// A helper method that can be used to determine if the constant contained 
  /// within is equal to a constant.  This only works for very small values, 
  /// because this is all that can be represented with all types.
  /// @brief Determine if this constant's value is same as an unsigned char.
  bool equalsInt(unsigned char V) const {
    assert(V <= 127 &&
           "equalsInt: Can only be used with very small positive constants!");
    return Val == V;
  }

  /// getTrue/getFalse - Return the singleton true/false values.
  static inline ConstantInt *getTrue() {
    if (TheTrueVal) return TheTrueVal;
    return CreateTrueFalseVals(true);
  }
  static inline ConstantInt *getFalse() {
    if (TheFalseVal) return TheFalseVal;
    return CreateTrueFalseVals(false);
  }

  /// Return a ConstantInt with the specified value for the specified type. The
  /// value V will be canonicalized to a uint64_t but accessing it with either
  /// getSExtValue() or getZExtValue() (ConstantInt) will yield the correct
  /// sized/signed value for the type Ty.
  /// @brief Get a ConstantInt for a specific value.
  static ConstantInt *get(const Type *Ty, int64_t V);

  /// getType - Specialize the getType() method to always return an IntegerType,
  /// which reduces the amount of casting needed in parts of the compiler.
  ///
  inline const IntegerType *getType() const {
    return reinterpret_cast<const IntegerType*>(Value::getType());
  }

  /// This static method returns true if the type Ty is big enough to 
  /// represent the value V. This can be used to avoid having the get method 
  /// assert when V is larger than Ty can represent. Note that there are two
  /// versions of this method, one for unsigned and one for signed integers.
  /// Although ConstantInt canonicalizes everything to an unsigned integer, 
  /// the signed version avoids callers having to convert a signed quantity
  /// to the appropriate unsigned type before calling the method.
  /// @returns true if V is a valid value for type Ty
  /// @brief Determine if the value is in range for the given type.
  static bool isValueValidForType(const Type *Ty, uint64_t V);
  static bool isValueValidForType(const Type *Ty, int64_t V);

  /// This function will return true iff this constant represents the "null"
  /// value that would be returned by the getNullValue method.
  /// @returns true if this is the null integer value.
  /// @brief Determine if the value is null.
  virtual bool isNullValue() const { 
    return Val == 0; 
  }

  /// This function will return true iff every bit in this constant is set
  /// to true.
  /// @returns true iff this constant's bits are all set to true.
  /// @brief Determine if the value is all ones.
  bool isAllOnesValue() const { 
    return getSExtValue() == -1; 
  }

  /// This function will return true iff this constant represents the largest
  /// value that may be represented by the constant's type.
  /// @returns true iff this is the largest value that may be represented 
  /// by this type.
  /// @brief Determine if the value is maximal.
  bool isMaxValue(bool isSigned) const {
    if (isSigned) {
      int64_t V = getSExtValue();
      if (V < 0) return false;    // Be careful about wrap-around on 'long's
      ++V;
      return !isValueValidForType(Value::getType(), V) || V < 0;
    }
    return isAllOnesValue();
  }

  /// This function will return true iff this constant represents the smallest
  /// value that may be represented by this constant's type.
  /// @returns true if this is the smallest value that may be represented by 
  /// this type.
  /// @brief Determine if the value is minimal.
  bool isMinValue(bool isSigned) const {
    if (isSigned) {
      int64_t V = getSExtValue();
      if (V > 0) return false;    // Be careful about wrap-around on 'long's
      --V;
      return !isValueValidForType(Value::getType(), V) || V > 0;
    }
    return getZExtValue() == 0;
  }

  /// @returns the value for an integer constant of the given type that has all
  /// its bits set to true.
  /// @brief Get the all ones value
  static ConstantInt *getAllOnesValue(const Type *Ty);

  /// @brief Methods to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const ConstantInt *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantIntVal;
  }
  static void ResetTrueFalse() { TheTrueVal = TheFalseVal = 0; }
private:
  static ConstantInt *CreateTrueFalseVals(bool WhichOne);
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
  static Constant *get(const ArrayType *T,
                       Constant*const*Vals, unsigned NumVals) {
    // FIXME: make this the primary ctor method.
    return get(T, std::vector<Constant*>(Vals, Vals+NumVals));
  }

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

  /// isCString - This method returns true if the array is a string (see
  /// isString) and it ends in a null byte \0 and does not contains any other
  /// null bytes except its terminator.
  bool isCString() const;

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
  static Constant *get(const std::vector<Constant*> &V, bool Packed = false);
  static Constant *get(Constant*const* Vals, unsigned NumVals,
                       bool Packed = false) {
    // FIXME: make this the primary ctor method.
    return get(std::vector<Constant*>(Vals, Vals+NumVals), Packed);
  }
  
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
/// ConstantVector - Constant Vector Declarations
///
class ConstantVector : public Constant {
  friend struct ConstantCreator<ConstantVector, VectorType,
                                    std::vector<Constant*> >;
  ConstantVector(const ConstantVector &);      // DO NOT IMPLEMENT
protected:
  ConstantVector(const VectorType *T, const std::vector<Constant*> &Val);
  ~ConstantVector();
public:
  /// get() - Static factory methods - Return objects of the specified value
  static Constant *get(const VectorType *T, const std::vector<Constant*> &);
  static Constant *get(const std::vector<Constant*> &V);
  static Constant *get(Constant*const* Vals, unsigned NumVals) {
    // FIXME: make this the primary ctor method.
    return get(std::vector<Constant*>(Vals, Vals+NumVals));
  }
  
  /// getType - Specialize the getType() method to always return an VectorType,
  /// which reduces the amount of casting needed in parts of the compiler.
  ///
  inline const VectorType *getType() const {
    return reinterpret_cast<const VectorType*>(Value::getType());
  }

  /// @returns the value for an packed integer constant of the given type that
  /// has all its bits set to true.
  /// @brief Get the all ones value
  static ConstantVector *getAllOnesValue(const VectorType *Ty);
  
  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because zero arrays are always
  /// created as ConstantAggregateZero objects.
  virtual bool isNullValue() const { return false; }

  /// This function will return true iff every element in this packed constant
  /// is set to all ones.
  /// @returns true iff this constant's emements are all set to all ones.
  /// @brief Determine if the value is all ones.
  bool isAllOnesValue() const;

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantVector *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueType() == ConstantVectorVal;
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
  static Constant *getCompareTy(unsigned short pred, Constant *C1, 
                                Constant *C2);
  static Constant *getSelectTy(const Type *Ty,
                               Constant *C1, Constant *C2, Constant *C3);
  static Constant *getGetElementPtrTy(const Type *Ty, Constant *C,
                                      Value* const *Idxs, unsigned NumIdxs);
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
  static Constant *getTrunc   (Constant *C, const Type *Ty);
  static Constant *getSExt    (Constant *C, const Type *Ty);
  static Constant *getZExt    (Constant *C, const Type *Ty);
  static Constant *getFPTrunc (Constant *C, const Type *Ty);
  static Constant *getFPExtend(Constant *C, const Type *Ty);
  static Constant *getUIToFP  (Constant *C, const Type *Ty);
  static Constant *getSIToFP  (Constant *C, const Type *Ty);
  static Constant *getFPToUI  (Constant *C, const Type *Ty);
  static Constant *getFPToSI  (Constant *C, const Type *Ty);
  static Constant *getPtrToInt(Constant *C, const Type *Ty);
  static Constant *getIntToPtr(Constant *C, const Type *Ty);
  static Constant *getBitCast (Constant *C, const Type *Ty);

  // @brief Convenience function for getting one of the casting operations
  // using a CastOps opcode.
  static Constant *getCast(
    unsigned ops,  ///< The opcode for the conversion
    Constant *C,   ///< The constant to be converted
    const Type *Ty ///< The type to which the constant is converted
  );

  // @brief Create a ZExt or BitCast cast constant expression
  static Constant *getZExtOrBitCast(
    Constant *C,   ///< The constant to zext or bitcast
    const Type *Ty ///< The type to zext or bitcast C to
  );

  // @brief Create a SExt or BitCast cast constant expression 
  static Constant *getSExtOrBitCast(
    Constant *C,   ///< The constant to sext or bitcast
    const Type *Ty ///< The type to sext or bitcast C to
  );

  // @brief Create a Trunc or BitCast cast constant expression
  static Constant *getTruncOrBitCast(
    Constant *C,   ///< The constant to trunc or bitcast
    const Type *Ty ///< The type to trunc or bitcast C to
  );

  /// @brief Create a BitCast or a PtrToInt cast constant expression
  static Constant *getPointerCast(
    Constant *C,   ///< The pointer value to be casted (operand 0)
    const Type *Ty ///< The type to which cast should be made
  );

  /// @brief Create a ZExt, Bitcast or Trunc for integer -> integer casts
  static Constant *getIntegerCast(
    Constant *C,    ///< The integer constant to be casted 
    const Type *Ty, ///< The integer type to cast to
    bool isSigned   ///< Whether C should be treated as signed or not
  );

  /// @brief Create a FPExt, Bitcast or FPTrunc for fp -> fp casts
  static Constant *getFPCast(
    Constant *C,    ///< The integer constant to be casted 
    const Type *Ty ///< The integer type to cast to
  );

  /// @brief Return true if this is a convert constant expression
  bool isCast() const;

  /// @brief Return true if this is a compare constant expression
  bool isCompare() const;

  /// Select constant expr
  ///
  static Constant *getSelect(Constant *C, Constant *V1, Constant *V2) {
    return getSelectTy(V1->getType(), C, V1, V2);
  }

  /// getSizeOf constant expr - computes the size of a type in a target
  /// independent way (Note: the return type is a ULong).
  ///
  static Constant *getSizeOf(const Type *Ty);

  /// ConstantExpr::get - Return a binary or shift operator constant expression,
  /// folding if possible.
  ///
  static Constant *get(unsigned Opcode, Constant *C1, Constant *C2);

  /// @brief Return an ICmp or FCmp comparison operator constant expression.
  static Constant *getCompare(unsigned short pred, Constant *C1, Constant *C2);

  /// ConstantExpr::get* - Return some common constants without having to
  /// specify the full Instruction::OPCODE identifier.
  ///
  static Constant *getNeg(Constant *C);
  static Constant *getNot(Constant *C);
  static Constant *getAdd(Constant *C1, Constant *C2);
  static Constant *getSub(Constant *C1, Constant *C2);
  static Constant *getMul(Constant *C1, Constant *C2);
  static Constant *getUDiv(Constant *C1, Constant *C2);
  static Constant *getSDiv(Constant *C1, Constant *C2);
  static Constant *getFDiv(Constant *C1, Constant *C2);
  static Constant *getURem(Constant *C1, Constant *C2); // unsigned rem
  static Constant *getSRem(Constant *C1, Constant *C2); // signed rem
  static Constant *getFRem(Constant *C1, Constant *C2);
  static Constant *getAnd(Constant *C1, Constant *C2);
  static Constant *getOr(Constant *C1, Constant *C2);
  static Constant *getXor(Constant *C1, Constant *C2);
  static Constant* getICmp(unsigned short pred, Constant* LHS, Constant* RHS);
  static Constant* getFCmp(unsigned short pred, Constant* LHS, Constant* RHS);
  static Constant *getShl(Constant *C1, Constant *C2);
  static Constant *getLShr(Constant *C1, Constant *C2);
  static Constant *getAShr(Constant *C1, Constant *C2);

  /// Getelementptr form.  std::vector<Value*> is only accepted for convenience:
  /// all elements must be Constant's.
  ///
  static Constant *getGetElementPtr(Constant *C,
                                    Constant* const *IdxList, unsigned NumIdx);
  static Constant *getGetElementPtr(Constant *C,
                                    Value* const *IdxList, unsigned NumIdx);
  
  static Constant *getExtractElement(Constant *Vec, Constant *Idx);
  static Constant *getInsertElement(Constant *Vec, Constant *Elt,Constant *Idx);
  static Constant *getShuffleVector(Constant *V1, Constant *V2, Constant *Mask);

  /// Floating point negation must be implemented with f(x) = -0.0 - x. This
  /// method returns the negative zero constant for floating point or packed
  /// floating point types; for all other types, it returns the null value.
  static Constant *getZeroValueForNegationExpr(const Type *Ty);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return false; }

  /// getOpcode - Return the opcode at the root of this constant expression
  unsigned getOpcode() const { return SubclassData; }

  /// getPredicate - Return the ICMP or FCMP predicate value. Assert if this is
  /// not an ICMP or FCMP constant expression.
  unsigned getPredicate() const;

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
