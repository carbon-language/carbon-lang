//===-- llvm/Constants.h - Constant class subclass definitions ---*- C++ -*--=//
//
// This file contains the declarations for the subclasses of Constant, which
// represent the different type of constant pool values
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONSTANTS_H
#define LLVM_CONSTANTS_H

#include "llvm/Constant.h"
#include "Support/DataTypes.h"

class ArrayType;
class StructType;
class PointerType;


//===---------------------------------------------------------------------------
/// ConstantIntegral - Shared superclass of boolean and integer constants.
///
/// This class just defines some common interfaces to be implemented.
///
class ConstantIntegral : public Constant {
protected:
  ConstantIntegral(const Type *Ty) : Constant(Ty) {}
public:

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
  static bool classof(const Constant *CPV);  // defined in Constants.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};


//===---------------------------------------------------------------------------
/// ConstantBool - Boolean Values
///
class ConstantBool : public ConstantIntegral {
  bool Val;
  ConstantBool(bool V);
  ~ConstantBool() {}
public:
  static ConstantBool *True, *False;  // The True & False values

  /// get() - Static factory methods - Return objects of the specified value
  static ConstantBool *get(bool Value) { return Value ? True : False; }
  static ConstantBool *get(const Type *Ty, bool Value) { return get(Value); }

  /// inverted - Return the opposite value of the current value.
  inline ConstantBool *inverted() const { return (this==True) ? False : True; }

  /// getValue - return the boolean value of this constant.
  ///
  inline bool getValue() const { return Val; }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  ///
  virtual bool isNullValue() const { return this == False; }
  virtual bool isMaxValue() const { return this == True; }
  virtual bool isMinValue() const { return this == False; }
  virtual bool isAllOnesValue() const { return this == True; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantBool *) { return true; }
  static bool classof(const Constant *CPV) {
    return (CPV == True) | (CPV == False);
  }
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};


//===---------------------------------------------------------------------------
/// ConstantInt - Superclass of ConstantSInt & ConstantUInt, to make dealing
/// with integral constants easier.
///
class ConstantInt : public ConstantIntegral {
protected:
  union {
    int64_t  Signed;
    uint64_t Unsigned;
  } Val;
  ConstantInt(const ConstantInt &);      // DO NOT IMPLEMENT
  ConstantInt(const Type *Ty, uint64_t V);
  ~ConstantInt() {}
public:
  /// equalsInt - Provide a helper method that can be used to determine if the
  /// constant contained within is equal to a constant.  This only works for
  /// very small values, because this is all that can be represented with all
  /// types.
  ///
  bool equalsInt(unsigned char V) const {
    assert(V <= 127 &&
	   "equals: Can only be used with very small positive constants!");
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
  static bool classof(const Constant *CPV);  // defined in Constants.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};


//===---------------------------------------------------------------------------
/// ConstantSInt - Signed Integer Values [sbyte, short, int, long]
///
class ConstantSInt : public ConstantInt {
  ConstantSInt(const ConstantSInt &);      // DO NOT IMPLEMENT
protected:
  ConstantSInt(const Type *Ty, int64_t V);
  ~ConstantSInt() {}
public:
  /// get() - Static factory methods - Return objects of the specified value
  static ConstantSInt *get(const Type *Ty, int64_t V);

  /// isValueValidForType - return true if Ty is big enough to represent V.
  static bool isValueValidForType(const Type *Ty, int64_t V);

  /// getValue - return the underlying value of this constant.
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
  static inline bool classof(const ConstantSInt *) { return true; }
  static bool classof(const Constant *CPV);  // defined in Constants.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};

//===---------------------------------------------------------------------------
/// ConstantUInt - Unsigned Integer Values [ubyte, ushort, uint, ulong]
///
class ConstantUInt : public ConstantInt {
  ConstantUInt(const ConstantUInt &);      // DO NOT IMPLEMENT
protected:
  ConstantUInt(const Type *Ty, uint64_t V);
  ~ConstantUInt() {}
public:
  /// get() - Static factory methods - Return objects of the specified value
  static ConstantUInt *get(const Type *Ty, uint64_t V);

  /// isValueValidForType - return true if Ty is big enough to represent V.
  static bool isValueValidForType(const Type *Ty, uint64_t V);

  /// getValue - return the underlying value of this constant.
  inline uint64_t getValue() const { return Val.Unsigned; }

  /// isMaxValue - Return true if this is the largest value that may be
  /// represented by this type.
  ///
  virtual bool isAllOnesValue() const;
  virtual bool isMaxValue() const { return isAllOnesValue(); }
  virtual bool isMinValue() const { return getValue() == 0; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantUInt *) { return true; }
  static bool classof(const Constant *CPV);  // defined in Constants.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};


//===---------------------------------------------------------------------------
/// ConstantFP - Floating Point Values [float, double]
///
class ConstantFP : public Constant {
  double Val;
  ConstantFP(const ConstantFP &);      // DO NOT IMPLEMENT
protected:
  ConstantFP(const Type *Ty, double V);
  ~ConstantFP() {}
public:
  /// get() - Static factory methods - Return objects of the specified value
  static ConstantFP *get(const Type *Ty, double V);

  /// isValueValidForType - return true if Ty is big enough to represent V.
  static bool isValueValidForType(const Type *Ty, double V);
  inline double getValue() const { return Val; }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return Val == 0; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantFP *) { return true; }
  static bool classof(const Constant *CPV);  // defined in Constants.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};


//===---------------------------------------------------------------------------
/// ConstantArray - Constant Array Declarations
///
class ConstantArray : public Constant {
  ConstantArray(const ConstantArray &);      // DO NOT IMPLEMENT
protected:
  ConstantArray(const ArrayType *T, const std::vector<Constant*> &Val);
  ~ConstantArray() {}

public:
  /// get() - Static factory methods - Return objects of the specified value
  static ConstantArray *get(const ArrayType *T, const std::vector<Constant*> &);
  static ConstantArray *get(const std::string &Initializer);
  
  /// getType - Specialize the getType() method to always return an ArrayType,
  /// which reduces the amount of casting needed in parts of the compiler.
  ///
  inline const ArrayType *getType() const {
    return (ArrayType*)Value::getType();
  }

  /// getAsString - If the sub-element type of this array is either sbyte or
  /// ubyte, then this method converts the array to an std::string and returns
  /// it.  Otherwise, it asserts out.
  ///
  std::string getAsString() const;

  /// getValues - Return a vector of the component constants that make up this
  /// array.
  inline const std::vector<Use> &getValues() const { return Operands; }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const {
    // FIXME: This should be made to be MUCH faster.  Just check against well
    // known null value!
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      if (!cast<Constant>(getOperand(i))->isNullValue())
        return false; 
    return true;
  }

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantArray *) { return true; }
  static bool classof(const Constant *CPV);  // defined in Constants.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};


//===---------------------------------------------------------------------------
// ConstantStruct - Constant Struct Declarations
//
class ConstantStruct : public Constant {
  ConstantStruct(const ConstantStruct &);      // DO NOT IMPLEMENT
protected:
  ConstantStruct(const StructType *T, const std::vector<Constant*> &Val);
  ~ConstantStruct() {}

public:
  /// get() - Static factory methods - Return objects of the specified value
  static ConstantStruct *get(const StructType *T,
                             const std::vector<Constant*> &V);

  /// getType() specialization - Reduce amount of casting...
  inline const StructType *getType() const {
    return (StructType*)Value::getType();
  }

  /// getValues - Return a vector of the component constants that make up this
  /// structure.
  inline const std::vector<Use> &getValues() const { return Operands; }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const {
    // FIXME: This should be made to be MUCH faster.  Just check against well
    // known null value!
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      if (!cast<Constant>(getOperand(i))->isNullValue())
        return false; 
    return true;
  }

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To);
  
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantStruct *) { return true; }
  static bool classof(const Constant *CPV);  // defined in Constants.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};

//===---------------------------------------------------------------------------
/// ConstantPointer - Constant Pointer Declarations
///
/// The ConstantPointer class represents a null pointer of a specific type. For
/// a more specific/useful instance, a subclass of ConstantPointer should be
/// used.
///
class ConstantPointer : public Constant {
  ConstantPointer(const ConstantPointer &);      // DO NOT IMPLEMENT
protected:
  inline ConstantPointer(const PointerType *T) : Constant((const Type*)T){}
  ~ConstantPointer() {}
public:
  inline const PointerType *getType() const {
    return (PointerType*)Value::getType();
  }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return false; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantPointer *) { return true; }
  static bool classof(const Constant *CPV);  // defined in Constants.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};

/// ConstantPointerNull - a constant pointer value that points to null
///
class ConstantPointerNull : public ConstantPointer {
  ConstantPointerNull(const ConstantPointerNull &);      // DO NOT IMPLEMENT
protected:
  inline ConstantPointerNull(const PointerType *T) : ConstantPointer(T) {}
  inline ~ConstantPointerNull() {}
public:

  /// get() - Static factory methods - Return objects of the specified value
  static ConstantPointerNull *get(const PointerType *T);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return true; }

  virtual void destroyConstant();

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantPointerNull *) { return true; }
  static inline bool classof(const ConstantPointer *P) {
    return (P->getNumOperands() == 0 && P->isNullValue());
  }
  static inline bool classof(const Constant *CPV) {
    return isa<ConstantPointer>(CPV) && classof(cast<ConstantPointer>(CPV));
  }
  static inline bool classof(const Value *V) {
    return isa<ConstantPointer>(V) && classof(cast<ConstantPointer>(V));
  }
};


/// ConstantPointerRef - a constant pointer value that is initialized to
/// point to a global value, which lies at a constant, fixed address.
///
class ConstantPointerRef : public ConstantPointer {
  friend class Module;   // Modules maintain these references
  ConstantPointerRef(const ConstantPointerRef &); // DNI!

protected:
  ConstantPointerRef(GlobalValue *GV);
  ~ConstantPointerRef() {}
public:
  /// get() - Static factory methods - Return objects of the specified value
  static ConstantPointerRef *get(GlobalValue *GV);

  const GlobalValue *getValue() const { 
    return cast<GlobalValue>(Operands[0].get());
  }

  GlobalValue *getValue() {
    return cast<GlobalValue>(Operands[0].get());
  }

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantPointerRef *) { return true; }
  static inline bool classof(const ConstantPointer *CPV) {
    // check for a single operand (the target value)
    return (CPV->getNumOperands() == 1);
  }
  static inline bool classof(const Constant *CPV) {
    return isa<ConstantPointer>(CPV) && classof(cast<ConstantPointer>(CPV));
  }
  static inline bool classof(const Value *V) {
    return isa<ConstantPointer>(V) && classof(cast<ConstantPointer>(V));
  }
};


// ConstantExpr - a constant value that is initialized with
// an expression using other constant values.  This is only used
// to represent values that cannot be evaluated at compile-time
// (e.g., something derived from an address) because it does
// not have a mechanism to store the actual value.
// Use the appropriate Constant subclass above for known constants.
//
class ConstantExpr : public Constant {
  unsigned iType;      // Operation type
  
protected:
  ConstantExpr(unsigned Opcode, Constant *C,  const Type *Ty);
  ConstantExpr(unsigned Opcode, Constant *C1, Constant *C2);
  ConstantExpr(Constant *C, const std::vector<Constant*> &IdxList,
               const Type *DestTy);
  ~ConstantExpr() {}
  
public:
  // Static methods to construct a ConstantExpr of different kinds.  Note that
  // these methods can return a constant of an arbitrary type, because they will
  // attempt to fold the constant expression into something simple if they can.
  
  /// Cast constant expr
  static Constant *getCast(Constant *C, const Type *Ty);

  /// Binary constant expr - Use with binary operators...
  static Constant *get(unsigned Opcode, Constant *C1, Constant *C2);

  /// Getelementptr form...
  static Constant *getGetElementPtr(Constant *C,
                                    const std::vector<Constant*> &IdxList);
  
  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return false; }
  
  /// getOpcode - Return the opcode at the root of this constant expression
  unsigned getOpcode() const { return iType; }

  /// getOpcodeName - Return a string representation for an opcode.
  const char *getOpcodeName() const;
  
  /// isConstantExpr - Return true if this is a ConstantExpr
  virtual bool isConstantExpr() const { return true; }

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To);
    
  /// Override methods to provide more type information...
  inline Constant *getOperand(unsigned i) { 
    return cast<Constant>(User::getOperand(i));
  }
  inline Constant *getOperand(unsigned i) const {
    return const_cast<Constant*>(cast<Constant>(User::getOperand(i)));
  }
  

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantExpr *) { return true; }
  static inline bool classof(const Constant *CPV) {
    return CPV->isConstantExpr();
  }
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};

#endif
