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
#include <vector>

class ArrayType;
class StructType;

//===----------------------------------------------------------------------===//
//                            ConstPoolVal Class
//===----------------------------------------------------------------------===//

class ConstPoolVal : public User {
protected:
  inline ConstPoolVal(const Type *Ty) : User(Ty, Value::ConstantVal) {}

public:
  // Specialize setName to handle symbol table majik...
  virtual void setName(const string &name, SymbolTable *ST = 0);

  virtual string getStrValue() const = 0;

  // Static constructor to get a '0' constant of arbitrary type...
  static ConstPoolVal *getNullConstant(const Type *Ty);
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
public:
  static ConstPoolBool *True, *False;  // The True & False values

  // Factory objects - Return objects of the specified value
  static ConstPoolBool *get(bool Value) { return Value ? True : False; }
  static ConstPoolBool *get(const Type *Ty, bool Value) { return get(Value); }

  // inverted - Return the opposite value of the current value.
  inline ConstPoolBool *inverted() const { return (this==True) ? False : True; }

  virtual string getStrValue() const;
  inline bool getValue() const { return Val; }
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
};


//===---------------------------------------------------------------------------
// ConstPoolSInt - Signed Integer Values [sbyte, short, int, long]
//
class ConstPoolSInt : public ConstPoolInt {
  ConstPoolSInt(const ConstPoolSInt &);      // DO NOT IMPLEMENT
protected:
  ConstPoolSInt(const Type *Ty, int64_t V);
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
public:
  static ConstPoolStruct *get(const StructType *T,
			      const vector<ConstPoolVal*> &V);

  virtual string getStrValue() const;

  inline const vector<Use> &getValues() const { return Operands; }
};

#endif
