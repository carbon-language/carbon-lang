//===-- llvm/ConstantVals.h - Constant Value nodes ---------------*- C++ -*--=//
//
// This file contains the declarations for the Constant class and all of
// its subclasses, which represent the different type of constant pool values
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONSTPOOLVALS_H
#define LLVM_CONSTPOOLVALS_H

#include "llvm/User.h"
#include "Support/DataTypes.h"

class ArrayType;
class StructType;
class PointerType;

//===----------------------------------------------------------------------===//
//                                Constant Class
//===----------------------------------------------------------------------===//

class Constant : public User {
protected:
  inline Constant(const Type *Ty) : User(Ty, Value::ConstantVal) {}
  ~Constant() {}

  // destroyConstant - Called if some element of this constant is no longer
  // valid.  At this point only other constants may be on the use_list for this
  // constant.  Any constants on our Use list must also be destroy'd.  The
  // implementation must be sure to remove the constant from the list of
  // available cached constants.  Implementations should call
  // destroyConstantImpl as the last thing they do, to destroy all users and
  // delete this.
  //
  virtual void destroyConstant() { assert(0 && "Not reached!"); }
  void destroyConstantImpl();
public:
  // Specialize setName to handle symbol table majik...
  virtual void setName(const std::string &name, SymbolTable *ST = 0);

  virtual std::string getStrValue() const = 0;

  // Static constructor to get a '0' constant of arbitrary type...
  static Constant *getNullConstant(const Type *Ty);

  // isNullValue - Return true if this is the value that would be returned by
  // getNullConstant.
  virtual bool isNullValue() const = 0;

  virtual void print(std::ostream &O) const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Constant *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::ConstantVal;
  }
};



//===----------------------------------------------------------------------===//
//              Classes to represent constant pool variable defs
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
// ConstantBool - Boolean Values
//
class ConstantBool : public Constant {
  bool Val;
  ConstantBool(const ConstantBool &);     // DO NOT IMPLEMENT
  ConstantBool(bool V);
  ~ConstantBool() {}
public:
  static ConstantBool *True, *False;  // The True & False values

  // Factory objects - Return objects of the specified value
  static ConstantBool *get(bool Value) { return Value ? True : False; }
  static ConstantBool *get(const Type *Ty, bool Value) { return get(Value); }

  // inverted - Return the opposite value of the current value.
  inline ConstantBool *inverted() const { return (this==True) ? False : True; }

  virtual std::string getStrValue() const;
  inline bool getValue() const { return Val; }

  // isNullValue - Return true if this is the value that would be returned by
  // getNullConstant.
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
  // getNullConstant.
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

  virtual std::string getStrValue() const;

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

  virtual std::string getStrValue() const;

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

  virtual std::string getStrValue() const;

  static bool isValueValidForType(const Type *Ty, double V);
  inline double getValue() const { return Val; }

  // isNullValue - Return true if this is the value that would be returned by
  // getNullConstant.
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
  
  virtual std::string getStrValue() const;
  inline const ArrayType *getType() const {
    return (ArrayType*)Value::getType();
  }

  inline const std::vector<Use> &getValues() const { return Operands; }

  // isNullValue - Return true if this is the value that would be returned by
  // getNullConstant.
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

  virtual std::string getStrValue() const;
  inline const StructType *getType() const {
    return (StructType*)Value::getType();
  }

  inline const std::vector<Use> &getValues() const { return Operands; }

  // isNullValue - Return true if this is the value that would be returned by
  // getNullConstant.
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
  virtual std::string getStrValue() const = 0;
  inline const PointerType *getType() const {
    return (PointerType*)Value::getType();
  }

  // isNullValue - Return true if this is the value that would be returned by
  // getNullConstant.
  virtual bool isNullValue() const { return false; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantPointer *) { return true; }
  static bool classof(const Constant *CPV);  // defined in CPV.cpp
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
  virtual std::string getStrValue() const;

  static ConstantPointerNull *get(const PointerType *T);

  // isNullValue - Return true if this is the value that would be returned by
  // getNullConstant.
  virtual bool isNullValue() const { return true; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantPointerNull *) { return true; }
  static inline bool classof(const ConstantPointer *P) {
    return P->getNumOperands() == 0;
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

  virtual std::string getStrValue() const;

  const GlobalValue *getValue() const { 
    return cast<GlobalValue>(Operands[0].get());
  }
  GlobalValue *getValue() {
    return cast<GlobalValue>(Operands[0].get());
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantPointerRef *) { return true; }
  static inline bool classof(const ConstantPointer *CPV) {
    return CPV->getNumOperands() == 1;
  }
  static inline bool classof(const Constant *CPV) {
    return isa<ConstantPointer>(CPV) && classof(cast<ConstantPointer>(CPV));
  }
  static inline bool classof(const Value *V) {
    return isa<ConstantPointer>(V) && classof(cast<ConstantPointer>(V));
  }

  // WARNING: Only to be used by Bytecode & Assembly Parsers!  USER CODE SHOULD
  // NOT USE THIS!!
  void mutateReference(GlobalValue *NewGV);
  // END WARNING!!
};



#endif
