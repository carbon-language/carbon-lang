//===-- llvm/Value.h - Definition of the Value class -------------*- C++ -*--=//
//
// This file defines the very important Value class.  This is subclassed by a
// bunch of other important classes, like Def, Method, Module, Type, etc...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_VALUE_H
#define LLVM_VALUE_H

#include <string>
#include <list>

class User;
class Type;
template<class ValueSubclass, class ItemParentType> class ValueHolder;

//===----------------------------------------------------------------------===//
//                                 Value Class
//===----------------------------------------------------------------------===//

class Value {
public:
  enum ValueTy {
    TypeVal,                // This is an instance of Type
    ConstantVal,            // This is an instance of ConstPoolVal
    MethodArgumentVal,      // This is an instance of MethodArgument
    InstructionVal,         // This is an instance of Instruction

    BasicBlockVal,          // This is an instance of BasicBlock
    MethodVal,              // This is an instance of Method
    ModuleVal,              // This is an instance of Module
  };

private:
  list<User *> Uses;
  string Name;
  const Type *Ty;
  ValueTy VTy;

  Value(const Value &);              // Do not implement
protected:
  inline void setType(const Type *ty) { Ty = ty; }
public:
  Value(const Type *Ty, ValueTy vty, const string &name = "");
  virtual ~Value();

  inline const Type *getType() const { return Ty; }
  inline ValueTy getValueType() const { return VTy; }

  inline bool hasName() const { return Name != ""; }
  inline const string &getName() const { return Name; }
  virtual void setName(const string &name) { Name = name; }


  // replaceAllUsesWith - Go through the uses list for this definition and make
  // each use point to "D" instead of "this".  After this completes, 'this's 
  // use list should be empty.
  //
  void replaceAllUsesWith(Value *D);

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
