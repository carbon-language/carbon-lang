//===-- llvm/Value.h - Definition of the Value class -------------*- C++ -*--=//
//
// This file defines the very important Value class.  This is subclassed by a
// bunch of other important classes, like Instruction, Function, Type, etc...
//
// This file also defines the Use<> template for users of value.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_VALUE_H
#define LLVM_VALUE_H

#include "llvm/Annotation.h"
#include "llvm/AbstractTypeUser.h"
#include "Support/Casting.h"
#include <iostream>
#include <vector>

class User;
class Type;
class Constant;
class Argument;
class Instruction;
class BasicBlock;
class GlobalValue;
class Function;
class GlobalVariable;
class SymbolTable;

//===----------------------------------------------------------------------===//
//                                 Value Class
//===----------------------------------------------------------------------===//

/// Value - The base class of all values computed by a program that may be used
/// as operands to other values.
///
class Value : public Annotable,         // Values are annotable
	      public AbstractTypeUser { // Values use potentially abstract types
public:
  enum ValueTy {
    TypeVal,                // This is an instance of Type
    ConstantVal,            // This is an instance of Constant
    ArgumentVal,            // This is an instance of Argument
    InstructionVal,         // This is an instance of Instruction
    BasicBlockVal,          // This is an instance of BasicBlock
    FunctionVal,            // This is an instance of Function
    GlobalVariableVal,      // This is an instance of GlobalVariable
  };

private:
  std::vector<User *> Uses;
  std::string Name;
  PATypeHandle<Type> Ty;
  ValueTy VTy;

  void operator=(const Value &);     // Do not implement
  Value(const Value &);              // Do not implement
public:
  Value(const Type *Ty, ValueTy vty, const std::string &name = "");
  virtual ~Value();
  
  /// dump - Support for debugging, callable in GDB: V->dump()
  //
  void dump() const;

  /// print - Implement operator<< on Value...
  ///
  virtual void print(std::ostream &O) const = 0;
  
  /// All values are typed, get the type of this value.
  ///
  inline const Type *getType() const { return Ty; }
  
  // All values can potentially be named...
  inline bool               hasName() const { return Name != ""; }
  inline const std::string &getName() const { return Name; }

  virtual void setName(const std::string &name, SymbolTable * = 0) {
    Name = name;
  }
  
  /// getValueType - Return the immediate subclass of this Value.
  ///
  inline ValueTy getValueType() const { return VTy; }
  
  /// replaceAllUsesWith - Go through the uses list for this definition and make
  /// each use point to "V" instead of "this".  After this completes, 'this's 
  /// use list is guaranteed to be empty.
  ///
  void replaceAllUsesWith(Value *V);

  /// refineAbstractType - This function is implemented because we use
  /// potentially abstract types, and these types may be resolved to more
  /// concrete types after we are constructed.
  ///
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  
  //----------------------------------------------------------------------
  // Methods for handling the vector of uses of this Value.
  //
  typedef std::vector<User*>::iterator       use_iterator;
  typedef std::vector<User*>::const_iterator use_const_iterator;

  inline unsigned           use_size()  const { return Uses.size();  }
  inline bool               use_empty() const { return Uses.empty(); }
  inline use_iterator       use_begin()       { return Uses.begin(); }
  inline use_const_iterator use_begin() const { return Uses.begin(); }
  inline use_iterator       use_end()         { return Uses.end();   }
  inline use_const_iterator use_end()   const { return Uses.end();   }
  inline User              *use_back()        { return Uses.back();  }
  inline const User        *use_back()  const { return Uses.back();  }

  inline void use_push_back(User *I)   { Uses.push_back(I); }
  User *use_remove(use_iterator &I);

  inline void addUse(User *I)      { Uses.push_back(I); }
  void killUse(User *I);
};

inline std::ostream &operator<<(std::ostream &OS, const Value *V) {
  if (V == 0)
    OS << "<null> value!\n";
  else
    V->print(OS);
  return OS;
}

inline std::ostream &operator<<(std::ostream &OS, const Value &V) {
  V.print(OS);
  return OS;
}


//===----------------------------------------------------------------------===//
//                                 UseTy Class
//===----------------------------------------------------------------------===//

// UseTy and it's friendly typedefs (Use) are here to make keeping the "use" 
// list of a definition node up-to-date really easy.
//
template<class ValueSubclass>
class UseTy {
  ValueSubclass *Val;
  User *U;
public:
  inline UseTy<ValueSubclass>(ValueSubclass *v, User *user) {
    Val = v; U = user;
    if (Val) Val->addUse(U);
  }

  inline ~UseTy<ValueSubclass>() { if (Val) Val->killUse(U); }

  inline operator ValueSubclass *() const { return Val; }

  inline UseTy<ValueSubclass>(const UseTy<ValueSubclass> &user) {
    Val = 0;
    U = user.U;
    operator=(user.Val);
  }
  inline ValueSubclass *operator=(ValueSubclass *V) { 
    if (Val) Val->killUse(U);
    Val = V;
    if (V) V->addUse(U);
    return V;
  }

  inline       ValueSubclass *operator->()       { return Val; }
  inline const ValueSubclass *operator->() const { return Val; }

  inline       ValueSubclass *get()       { return Val; }
  inline const ValueSubclass *get() const { return Val; }

  inline UseTy<ValueSubclass> &operator=(const UseTy<ValueSubclass> &user) {
    if (Val) Val->killUse(U);
    Val = user.Val;
    Val->addUse(U);
    return *this;
  }
};

typedef UseTy<Value> Use;    // Provide Use as a common UseTy type

template<typename From> struct simplify_type<UseTy<From> > {
  typedef typename simplify_type<From*>::SimpleType SimpleType;
  
  static SimpleType getSimplifiedValue(const UseTy<From> &Val) {
    return (SimpleType)Val.get();
  }
};
template<typename From> struct simplify_type<const UseTy<From> > {
  typedef typename simplify_type<From*>::SimpleType SimpleType;
  
  static SimpleType getSimplifiedValue(const UseTy<From> &Val) {
    return (SimpleType)Val.get();
  }
};

// isa - Provide some specializations of isa so that we don't have to include
// the subtype header files to test to see if the value is a subclass...
//
template <> inline bool isa_impl<Type, Value>(const Value &Val) { 
  return Val.getValueType() == Value::TypeVal;
}
template <> inline bool isa_impl<Constant, Value>(const Value &Val) { 
  return Val.getValueType() == Value::ConstantVal; 
}
template <> inline bool isa_impl<Argument, Value>(const Value &Val) { 
  return Val.getValueType() == Value::ArgumentVal;
}
template <> inline bool isa_impl<Instruction, Value>(const Value &Val) { 
  return Val.getValueType() == Value::InstructionVal;
}
template <> inline bool isa_impl<BasicBlock, Value>(const Value &Val) { 
  return Val.getValueType() == Value::BasicBlockVal;
}
template <> inline bool isa_impl<Function, Value>(const Value &Val) { 
  return Val.getValueType() == Value::FunctionVal;
}
template <> inline bool isa_impl<GlobalVariable, Value>(const Value &Val) { 
  return Val.getValueType() == Value::GlobalVariableVal;
}
template <> inline bool isa_impl<GlobalValue, Value>(const Value &Val) { 
  return isa<GlobalVariable>(Val) || isa<Function>(Val);
}

#endif
