//===-- llvm/ConstPoolVals.h - Constant Value nodes --------------*- C++ -*--=//
//
// This file contains the declarations for the ConstPoolVal class and all of
// its subclasses, which represent the different type of constant pool values
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONSTPOOLVALS_H
#define LLVM_CONSTPOOLVALS_H

#include "llvm/User.h"
#include "llvm/Support/DataTypes.h"

class ArrayType;
class StructType;
class PointerType;

//===----------------------------------------------------------------------===//
//                            ConstPoolVal Class
//===----------------------------------------------------------------------===//

class ConstPoolVal : public User {
protected:
  inline ConstPoolVal(const Type *Ty) : User(Ty, Value::ConstantVal) {}
  ~ConstPoolVal() {}

public:
  // Specialize setName to handle symbol table majik...
  virtual void setName(const string &name, SymbolTable *ST = 0);

  virtual string getStrValue() const = 0;

  // Static constructor to get a '0' constant of arbitrary type...
  static ConstPoolVal *getNullConstant(const Type *Ty);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstPoolVal *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::ConstantVal;
  }
};



//===----------------------------------------------------------------------===//
//              Classes to represent constant pool variable defs
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
// ConstPoolBool - Boolean Values
//
class ConstPoolBool : public ConstPoolVal {
  bool Val;
  ConstPoolBool(const ConstPoolBool &);     // DO NOT IMPLEMENT
  ConstPoolBool(bool V);
  ~ConstPoolBool() {}
public:
  static ConstPoolBool *True, *False;  // The True & False values

  // Factory objects - Return objects of the specified value
  static ConstPoolBool *get(bool Value) { return Value ? True : False; }
  static ConstPoolBool *get(const Type *Ty, bool Value) { return get(Value); }

  // inverted - Return the opposite value of the current value.
  inline ConstPoolBool *inverted() const { return (this==True) ? False : True; }

  virtual string getStrValue() const;
  inline bool getValue() const { return Val; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstPoolBool *) { return true; }
  static bool classof(const ConstPoolVal *CPV) {
    return (CPV == True) | (CPV == False);
  }
  static inline bool classof(const Value *V) {
    return isa<ConstPoolVal>(V) && classof(cast<ConstPoolVal>(V));
  }
};


//===---------------------------------------------------------------------------
// ConstPoolInt - Superclass of ConstPoolSInt & ConstPoolUInt, to make dealing
// with integral constants easier.
//
class ConstPoolInt : public ConstPoolVal {
protected:
  union {
    int64_t  Signed;
    uint64_t Unsigned;
  } Val;
  ConstPoolInt(const ConstPoolInt &);      // DO NOT IMPLEMENT
  ConstPoolInt(const Type *Ty, uint64_t V);
  ~ConstPoolInt() {}
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

  // ConstPoolInt::get static method: return a constant pool int with the
  // specified value.  as above, we work only with very small values here.
  //
  static ConstPoolInt *get(const Type *Ty, unsigned char V);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstPoolInt *) { return true; }
  static bool classof(const ConstPoolVal *CPV);  // defined in CPV.cpp
  static inline bool classof(const Value *V) {
    return isa<ConstPoolVal>(V) && classof(cast<ConstPoolVal>(V));
  }
};


//===---------------------------------------------------------------------------
// ConstPoolSInt - Signed Integer Values [sbyte, short, int, long]
//
class ConstPoolSInt : public ConstPoolInt {
  ConstPoolSInt(const ConstPoolSInt &);      // DO NOT IMPLEMENT
protected:
  ConstPoolSInt(const Type *Ty, int64_t V);
  ~ConstPoolSInt() {}
public:
  static ConstPoolSInt *get(const Type *Ty, int64_t V);

  virtual string getStrValue() const;

  static bool isValueValidForType(const Type *Ty, int64_t V);
  inline int64_t getValue() const { return Val.Signed; }
};

//===---------------------------------------------------------------------------
// ConstPoolUInt - Unsigned Integer Values [ubyte, ushort, uint, ulong]
//
class ConstPoolUInt : public ConstPoolInt {
  ConstPoolUInt(const ConstPoolUInt &);      // DO NOT IMPLEMENT
protected:
  ConstPoolUInt(const Type *Ty, uint64_t V);
  ~ConstPoolUInt() {}
public:
  static ConstPoolUInt *get(const Type *Ty, uint64_t V);

  virtual string getStrValue() const;

  static bool isValueValidForType(const Type *Ty, uint64_t V);
  inline uint64_t getValue() const { return Val.Unsigned; }
};


//===---------------------------------------------------------------------------
// ConstPoolFP - Floating Point Values [float, double]
//
class ConstPoolFP : public ConstPoolVal {
  double Val;
  ConstPoolFP(const ConstPoolFP &);      // DO NOT IMPLEMENT
protected:
  ConstPoolFP(const Type *Ty, double V);
  ~ConstPoolFP() {}
public:
  static ConstPoolFP *get(const Type *Ty, double V);

  virtual string getStrValue() const;

  static bool isValueValidForType(const Type *Ty, double V);
  inline double getValue() const { return Val; }
};


//===---------------------------------------------------------------------------
// ConstPoolArray - Constant Array Declarations
//
class ConstPoolArray : public ConstPoolVal {
  ConstPoolArray(const ConstPoolArray &);      // DO NOT IMPLEMENT
protected:
  ConstPoolArray(const ArrayType *T, const vector<ConstPoolVal*> &Val);
  ~ConstPoolArray() {}
public:
  static ConstPoolArray *get(const ArrayType *T, const vector<ConstPoolVal*> &);

  virtual string getStrValue() const;

  inline const vector<Use> &getValues() const { return Operands; }
};


//===---------------------------------------------------------------------------
// ConstPoolStruct - Constant Struct Declarations
//
class ConstPoolStruct : public ConstPoolVal {
  ConstPoolStruct(const ConstPoolStruct &);      // DO NOT IMPLEMENT
protected:
  ConstPoolStruct(const StructType *T, const vector<ConstPoolVal*> &Val);
  ~ConstPoolStruct() {}
public:
  static ConstPoolStruct *get(const StructType *T,
			      const vector<ConstPoolVal*> &V);

  virtual string getStrValue() const;

  inline const vector<Use> &getValues() const { return Operands; }
};

//===---------------------------------------------------------------------------
// ConstPoolPointer - Constant Pointer Declarations
//
// The ConstPoolPointer class represents a null pointer of a specific type. For
// a more specific/useful instance, a subclass of ConstPoolPointer should be
// used.
//
class ConstPoolPointer : public ConstPoolVal {
  ConstPoolPointer(const ConstPoolPointer &);      // DO NOT IMPLEMENT
protected:
  ConstPoolPointer(const PointerType *T);
  ~ConstPoolPointer() {}
public:
  static ConstPoolPointer *getNullPointer(const PointerType *T) {
    // FIXME: These should all be shared!
    return new ConstPoolPointer(T);
  }

  virtual string getStrValue() const;
};


// ConstPoolPointerReference - a constant pointer value that is initialized to
// point to a global value, which lies at a constant, fixed address.
//
class ConstPoolPointerReference : public ConstPoolPointer {
  ConstPoolPointerReference(const ConstPoolPointerReference &); // DNI!
protected:
  ConstPoolPointerReference(GlobalValue *GV);
  ~ConstPoolPointerReference() {}
public:
  static ConstPoolPointerReference *get(GlobalValue *GV) {
    // FIXME: These should all be shared!
    return new ConstPoolPointerReference(GV);
  }

  virtual string getStrValue() const;

  const GlobalValue *getValue() const { 
    return cast<GlobalValue>(Operands[0].get());
  }
  GlobalValue *getValue() {
    return cast<GlobalValue>(Operands[0].get());
  }
};



#endif
