//===-- llvm/Value.h - Definition of the Value class -------------*- C++ -*--=//
//
// This file defines the very important Value class.  This is subclassed by a
// bunch of other important classes, like Def, Method, Module, Type, etc...
//
// This file also defines the Use<> template for users of value.
//
// This file also defines the isa<X>(), cast<X>(), and dyn_cast<X>() templates.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_VALUE_H
#define LLVM_VALUE_H

#include <vector>
#include "llvm/Annotation.h"
#include "llvm/AbstractTypeUser.h"

class User;
class Type;
class Constant;
class MethodArgument;
class Instruction;
class BasicBlock;
class GlobalValue;
class Method;
class GlobalVariable;
class Module;
class SymbolTable;
template<class ValueSubclass, class ItemParentType, class SymTabType> 
  class ValueHolder;

//===----------------------------------------------------------------------===//
//                                 Value Class
//===----------------------------------------------------------------------===//

class Value : public Annotable,         // Values are annotable
	      public AbstractTypeUser { // Values use potentially abstract types
public:
  enum ValueTy {
    TypeVal,                // This is an instance of Type
    ConstantVal,            // This is an instance of Constant
    MethodArgumentVal,      // This is an instance of MethodArgument
    InstructionVal,         // This is an instance of Instruction
    BasicBlockVal,          // This is an instance of BasicBlock
    MethodVal,              // This is an instance of Method
    GlobalVariableVal,      // This is an instance of GlobalVariable
    ModuleVal,              // This is an instance of Module
  };

private:
  vector<User *> Uses;
  string Name;
  PATypeHandle<Type> Ty;
  ValueTy VTy;

  Value(const Value &);              // Do not implement
protected:
  inline void setType(const Type *ty) { Ty = ty; }
public:
  Value(const Type *Ty, ValueTy vty, const string &name = "");
  virtual ~Value();
  
  // Support for debugging 
  void dump() const;
  
  // All values can potentially be typed
  inline const Type *getType() const { return Ty; }
  
  // All values can potentially be named...
  inline bool          hasName() const { return Name != ""; }
  inline const string &getName() const { return Name; }

  virtual void setName(const string &name, SymbolTable * = 0) {
    Name = name;
  }
  
  // Methods for determining the subtype of this Value.  The getValueType()
  // method returns the type of the value directly.  The cast*() methods are
  // equivalent to using dynamic_cast<>... if the cast is successful, this is
  // returned, otherwise you get a null pointer.
  //
  // The family of functions Val->cast<type>Asserting() is used in the same
  // way as the Val->cast<type>() instructions, but they assert the expected
  // type instead of checking it at runtime.
  //
  inline ValueTy getValueType() const { return VTy; }
  
  // replaceAllUsesWith - Go through the uses list for this definition and make
  // each use point to "D" instead of "this".  After this completes, 'this's 
  // use list should be empty.
  //
  void replaceAllUsesWith(Value *D);

  // refineAbstractType - This function is implemented because we use
  // potentially abstract types, and these types may be resolved to more
  // concrete types after we are constructed.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  
  //----------------------------------------------------------------------
  // Methods for handling the vector of uses of this Value.
  //
  typedef vector<User*>::iterator       use_iterator;
  typedef vector<User*>::const_iterator use_const_iterator;

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

// real_type - Provide a macro to get the real type of a value that might be 
// a use.  This provides a typedef 'Type' that is the argument type for all
// non UseTy types, and is the contained pointer type of the use if it is a
// UseTy.
//
template <class X> class real_type { typedef X Type; };
template <class X> class real_type <class UseTy<X> > { typedef X *Type; };

//===----------------------------------------------------------------------===//
//                          Type Checking Templates
//===----------------------------------------------------------------------===//

// isa<X> - Return true if the parameter to the template is an instance of the
// template type argument.  Used like this:
//
//  if (isa<Type>(myVal)) { ... }
//
template <class X, class Y>
inline bool isa(Y Val) {
  assert(Val && "isa<Ty>(NULL) invoked!");
  return X::classof(Val);
}


// cast<X> - Return the argument parameter cast to the specified type.  This
// casting operator asserts that the type is correct, so it does not return null
// on failure.  But it will correctly return NULL when the input is NULL.
// Used Like this:
//
//  cast<      Instruction>(myVal)->getParent()
//  cast<const Instruction>(myVal)->getParent()
//
template <class X, class Y>
inline X *cast(Y Val) {
  assert(isa<X>(Val) && "cast<Ty>() argument of uncompatible type!");
  return (X*)(real_type<Y>::Type)Val;
}

// cast_or_null<X> - Functionally identical to cast, except that a null value is
// accepted.
//
template <class X, class Y>
inline X *cast_or_null(Y Val) {
  assert((Val == 0 || isa<X>(Val)) &&
         "cast_or_null<Ty>() argument of uncompatible type!");
  return (X*)(real_type<Y>::Type)Val;
}


// dyn_cast<X> - Return the argument parameter cast to the specified type.  This
// casting operator returns null if the argument is of the wrong type, so it can
// be used to test for a type as well as cast if successful.  This should be
// used in the context of an if statement like this:
//
//  if (const Instruction *I = dyn_cast<const Instruction>(myVal)) { ... }
//

template <class X, class Y>
inline X *dyn_cast(Y Val) {
  return isa<X>(Val) ? cast<X>(Val) : 0;
}

// dyn_cast_or_null<X> - Functionally identical to dyn_cast, except that a null
// value is accepted.
//
template <class X, class Y>
inline X *dyn_cast_or_null(Y Val) {
  return (Val && isa<X>(Val)) ? cast<X>(Val) : 0;
}


// isa - Provide some specializations of isa so that we have to include the
// subtype header files to test to see if the value is a subclass...
//
template <> inline bool isa<Type, const Value*>(const Value *Val) { 
  return Val->getValueType() == Value::TypeVal;
}
template <> inline bool isa<Type, Value*>(Value *Val) { 
  return Val->getValueType() == Value::TypeVal;
}
template <> inline bool isa<Constant, const Value*>(const Value *Val) { 
  return Val->getValueType() == Value::ConstantVal; 
}
template <> inline bool isa<Constant, Value*>(Value *Val) { 
  return Val->getValueType() == Value::ConstantVal; 
}
template <> inline bool isa<MethodArgument, const Value*>(const Value *Val) { 
  return Val->getValueType() == Value::MethodArgumentVal;
}
template <> inline bool isa<MethodArgument, Value*>(Value *Val) { 
  return Val->getValueType() == Value::MethodArgumentVal;
}
template <> inline bool isa<Instruction, const Value*>(const Value *Val) { 
  return Val->getValueType() == Value::InstructionVal;
}
template <> inline bool isa<Instruction, Value*>(Value *Val) { 
  return Val->getValueType() == Value::InstructionVal;
}
template <> inline bool isa<BasicBlock, const Value*>(const Value *Val) { 
  return Val->getValueType() == Value::BasicBlockVal;
}
template <> inline bool isa<BasicBlock, Value*>(Value *Val) { 
  return Val->getValueType() == Value::BasicBlockVal;
}
template <> inline bool isa<Method, const Value*>(const Value *Val) { 
  return Val->getValueType() == Value::MethodVal;
}
template <> inline bool isa<Method, Value*>(Value *Val) { 
  return Val->getValueType() == Value::MethodVal;
}
template <> inline bool isa<GlobalVariable, const Value*>(const Value *Val) { 
  return Val->getValueType() == Value::GlobalVariableVal;
}
template <> inline bool isa<GlobalVariable, Value*>(Value *Val) { 
  return Val->getValueType() == Value::GlobalVariableVal;
}
template <> inline bool isa<GlobalValue, const Value*>(const Value *Val) { 
  return isa<GlobalVariable>(Val) || isa<Method>(Val);
}
template <> inline bool isa<GlobalValue, Value*>(Value *Val) { 
  return isa<GlobalVariable>(Val) || isa<Method>(Val);
}
template <> inline bool isa<Module, const Value*>(const Value *Val) { 
  return Val->getValueType() == Value::ModuleVal;
}
template <> inline bool isa<Module, Value*>(Value *Val) { 
  return Val->getValueType() == Value::ModuleVal;
}

#endif
