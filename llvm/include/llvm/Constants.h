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
class ConstantExpr;

//===---------------------------------------------------------------------------
// ConstantBool - Boolean Values
//
class ConstantBool : public Constant {
  bool Val;
  ConstantBool(bool V);
  ~ConstantBool() {}
public:
  static ConstantBool *True, *False;  // The True & False values

  // Factory objects - Return objects of the specified value
  static ConstantBool *get(bool Value) { return Value ? True : False; }
  static ConstantBool *get(const Type *Ty, bool Value) { return get(Value); }

  // inverted - Return the opposite value of the current value.
  inline ConstantBool *inverted() const { return (this==True) ? False : True; }

  inline bool getValue() const { return Val; }

  // isNullValue - Return true if this is the value that would be returned by
  // getNullValue.
  virtual bool isNullValue() const { return this == False; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantBool *) { return true; }
  static bool classof(const Constant *CPV) {
    return (CPV == True) | (CPV == False);
  }
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};


//===---------------------------------------------------------------------------
// ConstantInt - Superclass of ConstantSInt & ConstantUInt, to make dealing
// with integral constants easier.
//
class ConstantInt : public Constant {
protected:
  union {
    int64_t  Signed;
    uint64_t Unsigned;
  } Val;
  ConstantInt(const ConstantInt &);      // DO NOT IMPLEMENT
  ConstantInt(const Type *Ty, uint64_t V);
  ~ConstantInt() {}
public:
  // equalsInt - Provide a helper method that can be used to determine if the 
  // constant contained within is equal to a constant.  This only works for very
  // small values, because this is all that can be represented with all types.
  //
  bool equalsInt(unsigned char V) const {
    assert(V <= 127 &&
	   "equals: Can only be used with very small positive constants!");
    return Val.Unsigned == V;
  }

  // ConstantInt::get static method: return a constant pool int with the
  // specified value.  as above, we work only with very small values here.
  //
  static ConstantInt *get(const Type *Ty, unsigned char V);

  // isNullValue - Return true if this is the value that would be returned by
  // getNullValue.
  virtual bool isNullValue() const { return Val.Unsigned == 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantInt *) { return true; }
  static bool classof(const Constant *CPV);  // defined in CPV.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};


//===---------------------------------------------------------------------------
// ConstantSInt - Signed Integer Values [sbyte, short, int, long]
//
class ConstantSInt : public ConstantInt {
  ConstantSInt(const ConstantSInt &);      // DO NOT IMPLEMENT
protected:
  ConstantSInt(const Type *Ty, int64_t V);
  ~ConstantSInt() {}
public:
  static ConstantSInt *get(const Type *Ty, int64_t V);

  static bool isValueValidForType(const Type *Ty, int64_t V);
  inline int64_t getValue() const { return Val.Signed; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantSInt *) { return true; }
  static bool classof(const Constant *CPV);  // defined in CPV.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};

//===---------------------------------------------------------------------------
// ConstantUInt - Unsigned Integer Values [ubyte, ushort, uint, ulong]
//
class ConstantUInt : public ConstantInt {
  ConstantUInt(const ConstantUInt &);      // DO NOT IMPLEMENT
protected:
  ConstantUInt(const Type *Ty, uint64_t V);
  ~ConstantUInt() {}
public:
  static ConstantUInt *get(const Type *Ty, uint64_t V);

  static bool isValueValidForType(const Type *Ty, uint64_t V);
  inline uint64_t getValue() const { return Val.Unsigned; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantUInt *) { return true; }
  static bool classof(const Constant *CPV);  // defined in CPV.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};


//===---------------------------------------------------------------------------
// ConstantFP - Floating Point Values [float, double]
//
class ConstantFP : public Constant {
  double Val;
  ConstantFP(const ConstantFP &);      // DO NOT IMPLEMENT
protected:
  ConstantFP(const Type *Ty, double V);
  ~ConstantFP() {}
public:
  static ConstantFP *get(const Type *Ty, double V);

  static bool isValueValidForType(const Type *Ty, double V);
  inline double getValue() const { return Val; }

  // isNullValue - Return true if this is the value that would be returned by
  // getNullValue.
  virtual bool isNullValue() const { return Val == 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantFP *) { return true; }
  static bool classof(const Constant *CPV);  // defined in CPV.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};


//===---------------------------------------------------------------------------
// ConstantArray - Constant Array Declarations
//
class ConstantArray : public Constant {
  ConstantArray(const ConstantArray &);      // DO NOT IMPLEMENT
protected:
  ConstantArray(const ArrayType *T, const std::vector<Constant*> &Val);
  ~ConstantArray() {}

  virtual void destroyConstant();
public:
  static ConstantArray *get(const ArrayType *T, const std::vector<Constant*> &);
  static ConstantArray *get(const std::string &Initializer);
  
  inline const ArrayType *getType() const {
    return (ArrayType*)Value::getType();
  }

  inline const std::vector<Use> &getValues() const { return Operands; }

  // isNullValue - Return true if this is the value that would be returned by
  // getNullValue.
  virtual bool isNullValue() const { return false; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantArray *) { return true; }
  static bool classof(const Constant *CPV);  // defined in CPV.cpp
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

  virtual void destroyConstant();
public:
  static ConstantStruct *get(const StructType *T,
                             const std::vector<Constant*> &V);

  inline const StructType *getType() const {
    return (StructType*)Value::getType();
  }

  inline const std::vector<Use> &getValues() const { return Operands; }

  // isNullValue - Return true if this is the value that would be returned by
  // getNullValue.
  virtual bool isNullValue() const { return false; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantStruct *) { return true; }
  static bool classof(const Constant *CPV);  // defined in CPV.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};

//===---------------------------------------------------------------------------
// ConstantPointer - Constant Pointer Declarations
//
// The ConstantPointer class represents a null pointer of a specific type. For
// a more specific/useful instance, a subclass of ConstantPointer should be
// used.
//
class ConstantPointer : public Constant {
  ConstantPointer(const ConstantPointer &);      // DO NOT IMPLEMENT
protected:
  inline ConstantPointer(const PointerType *T) : Constant((const Type*)T){}
  ~ConstantPointer() {}
public:
  inline const PointerType *getType() const {
    return (PointerType*)Value::getType();
  }

  // isNullValue - Return true if this is the value that would be returned by
  // getNullValue.
  virtual bool isNullValue() const { return false; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantPointer *) { return true; }
  static bool classof(const Constant *CPV);  // defined in Constants.cpp
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }
};

// ConstantPointerNull - a constant pointer value that points to null
//
class ConstantPointerNull : public ConstantPointer {
  ConstantPointerNull(const ConstantPointerNull &);      // DO NOT IMPLEMENT
protected:
  inline ConstantPointerNull(const PointerType *T) : ConstantPointer(T) {}
  inline ~ConstantPointerNull() {}
public:

  static ConstantPointerNull *get(const PointerType *T);

  // isNullValue - Return true if this is the value that would be returned by
  // getNullValue.
  virtual bool isNullValue() const { return true; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
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


// ConstantPointerRef - a constant pointer value that is initialized to
// point to a global value, which lies at a constant, fixed address.
//
class ConstantPointerRef : public ConstantPointer {
  friend class Module;   // Modules maintain these references
  ConstantPointerRef(const ConstantPointerRef &); // DNI!

protected:
  ConstantPointerRef(GlobalValue *GV);
  ~ConstantPointerRef() {}

  virtual void destroyConstant() { destroyConstantImpl(); }
public:
  static ConstantPointerRef *get(GlobalValue *GV);

  const GlobalValue *getValue() const { 
    return cast<GlobalValue>(Operands[0].get());
  }
  GlobalValue *getValue() {
    return cast<GlobalValue>(Operands[0].get());
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
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

  // WARNING: Only to be used by Bytecode & Assembly Parsers!  USER CODE SHOULD
  // NOT USE THIS!!
  // Returns the number of uses of OldV that were replaced.
  virtual unsigned mutateReferences(Value* OldV, Value *NewV);
  // END WARNING!!
};


// ConstantExpr - a constant value that is initialized with
// an expression using other constant values.  This is only used
// to represent values that cannot be evaluated at compile-time
// (e.g., something derived from an address) because it does
// not have a mechanism to store the actual value.
// Use the appropriate Constant subclass above for known constants.
//
class ConstantExpr : public Constant {
protected:
  unsigned iType;      // operation type
  
protected:
  ConstantExpr(unsigned opCode, Constant *C,  const Type *Ty);
  ConstantExpr(unsigned opCode, Constant* C1, Constant* C2, const Type *Ty);
  ConstantExpr(unsigned opCode, Constant* C,
               const std::vector<Value*>& IdxList, const Type *Ty);
  ~ConstantExpr() {}
  
  virtual void destroyConstant();
  
public:
  // Static methods to construct a ConstantExpr of different kinds.
  static ConstantExpr *get(unsigned opCode, Constant *C, const Type *Ty);
  static ConstantExpr *get(unsigned opCode,
                           Constant *C1, Constant *C2, const Type *Ty);
  static ConstantExpr *get(unsigned opCode, Constant* C,
                       const std::vector<Value*>& idxList, const Type *Ty);
  
  // isNullValue - Return true if this is the value that would be returned by
  // getNullValue.
  virtual bool isNullValue() const { return false; }
  
  // getOpcode - Return the opcode at the root of this constant expression
  unsigned getOpcode() const { return iType; }

  // getOpcodeName - Return a string representation for an opcode.
  static const char* getOpcodeName(unsigned opCode);
  const char* getOpcodeName() const {
    return getOpcodeName(getOpcode());
  }
  
  // isConstantExpr - Return true if this is a ConstantExpr
  virtual bool isConstantExpr() const { return true; }
  
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantExpr *) { return true; }
  static inline bool classof(const Constant *CPV) {
    return CPV->isConstantExpr();
  }
  static inline bool classof(const Value *V) {
    return isa<Constant>(V) && classof(cast<Constant>(V));
  }

public:
  // WARNING: Only to be used by Bytecode & Assembly Parsers!  USER CODE SHOULD
  // NOT USE THIS!!
  // Returns the number of uses of OldV that were replaced.
  virtual unsigned mutateReferences(Value* OldV, Value *NewV);
  // END WARNING!!
};


#endif
