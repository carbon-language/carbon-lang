//===-- llvm/Value.h - Definition of the Value class -------------*- C++ -*--=//
//
// This file defines the very important Value class.  This is subclassed by a
// bunch of other important classes, like Def, Method, Module, Type, etc...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_VALUE_H
#define LLVM_VALUE_H

#include <list>
#include "llvm/Annotation.h"
#include "llvm/AbstractTypeUser.h"

class User;
class Type;
class ConstPoolVal;
class MethodArgument;
class Instruction;
class BasicBlock;
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
    ConstantVal,            // This is an instance of ConstPoolVal
    MethodArgumentVal,      // This is an instance of MethodArgument
    InstructionVal,         // This is an instance of Instruction
    BasicBlockVal,          // This is an instance of BasicBlock
    MethodVal,              // This is an instance of Method
    GlobalVal,              // This is an instance of GlobalVariable
    ModuleVal,              // This is an instance of Module
  };

private:
  list<User *> Uses;
  string Name;
  PATypeHandle<Type> Ty;
  ValueTy VTy;

  Value(const Value &);              // Do not implement
protected:
  inline void setType(const Type *ty) { Ty = ty; }
public:
  Value(const Type *Ty, ValueTy vty, const string &name = "");
  virtual ~Value();

  inline const Type *getType() const { return Ty; }

  // All values can potentially be named...
  inline bool hasName() const { return Name != ""; }
  inline const string &getName() const { return Name; }
  virtual void setName(const string &name, SymbolTable * = 0) { Name = name; }

  // Methods for determining the subtype of this Value.  The getValueType()
  // method returns the type of the value directly.  The cast*() methods are
  // equilivent to using dynamic_cast<>... if the cast is successful, this is
  // returned, otherwise you get a null pointer, allowing expressions like this:
  //
  // if (Instruction *I = Val->castInstruction()) { ... }
  //
  // This section also defines a family of isType, isConstant, isMethodArgument,
  // etc functions...
  //
  // The family of functions Val->cast<type>Asserting() is used in the same
  // way as the Val->cast<type>() instructions, but they assert the expected
  // type instead of checking it at runtime.
  //
  inline ValueTy getValueType() const { return VTy; }

  // Use a macro to define the functions, otherwise these definitions are just
  // really long and ugly.
#define CAST_FN(NAME, CLASS)                                              \
  inline bool is##NAME() const { return VTy == NAME##Val; }               \
  inline const CLASS *cast##NAME() const { /*const version */             \
    return is##NAME() ? (const CLASS*)this : 0;                           \
  }                                                                       \
  inline CLASS *cast##NAME() {         /* nonconst version */             \
    return is##NAME() ? (CLASS*)this : 0;                                 \
  }                                                                       \
  inline const CLASS *cast##NAME##Asserting() const { /*const version */  \
    assert(is##NAME() && "Expected Value Type: " #NAME);                  \
    return (const CLASS*)this;                                            \
  }                                                                       \
  inline CLASS *cast##NAME##Asserting() {         /* nonconst version */  \
    assert(is##NAME() && "Expected Value Type: " #NAME);                  \
    return (CLASS*)this;                                                  \
  }                                                                       \

  CAST_FN(Constant      ,       ConstPoolVal  )
  CAST_FN(MethodArgument,       MethodArgument)
  CAST_FN(Instruction   ,       Instruction   )
  CAST_FN(BasicBlock    ,       BasicBlock    )
  CAST_FN(Method        ,       Method        )
  CAST_FN(Global        ,       GlobalVariable)
  CAST_FN(Module        ,       Module        )
#undef CAST_FN

  // Type value is special, because there is no nonconst version of functions!
  inline bool isType() const { return VTy == TypeVal; }
  inline const Type *castType() const {
    return (VTy == TypeVal) ? (const Type*)this : 0;
  }
  inline const Type *castTypeAsserting() const {
    assert(isType() && "Expected Value Type: Type");
    return (const Type*)this;
  }

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
  // Methods for handling the list of uses of this DEF.
  //
  typedef list<User*>::iterator       use_iterator;
  typedef list<User*>::const_iterator use_const_iterator;

  inline unsigned           use_size()  const { return Uses.size();  }
  inline bool               use_empty() const { return Uses.empty(); }
  inline use_iterator       use_begin()       { return Uses.begin(); }
  inline use_const_iterator use_begin() const { return Uses.begin(); }
  inline use_iterator       use_end()         { return Uses.end();   }
  inline use_const_iterator use_end()   const { return Uses.end();   }

  inline void use_push_back(User *I)   { Uses.push_back(I); }
  User *use_remove(use_iterator &I);

  inline void addUse(User *I)      { Uses.push_back(I); }
  void killUse(User *I);
};

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

  inline UseTy<ValueSubclass> &operator=(const UseTy<ValueSubclass> &user) {
    if (Val) Val->killUse(U);
    Val = user.Val;
    Val->addUse(U);
    return *this;
  }
};

typedef UseTy<Value> Use;

#endif
