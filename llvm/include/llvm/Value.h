//===-- llvm/Value.h - Definition of the Value class ------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the very important Value class.  This is subclassed by a
// bunch of other important classes, like Instruction, Function, Type, etc...
//
// This file also defines the Use<> template for users of value.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_VALUE_H
#define LLVM_VALUE_H

#include "llvm/AbstractTypeUser.h"
#include "llvm/Use.h"
#include "Support/Annotation.h"
#include "Support/Casting.h"
#include <iostream>

namespace llvm {

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
struct Value : public Annotable {         // Values are annotable
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
  iplist<Use> Uses;
  std::string Name;
  PATypeHolder Ty;
  ValueTy VTy;

  void operator=(const Value &);     // Do not implement
  Value(const Value &);              // Do not implement
public:
  Value(const Type *Ty, ValueTy vty, const std::string &name = "");
  virtual ~Value();
  
  /// dump - Support for debugging, callable in GDB: V->dump()
  //
  virtual void dump() const;

  /// print - Implement operator<< on Value...
  ///
  virtual void print(std::ostream &O) const = 0;
  
  /// All values are typed, get the type of this value.
  ///
  inline const Type *getType() const { return Ty; }
  
  // All values can potentially be named...
  inline bool               hasName() const { return !Name.empty(); }
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

  // uncheckedReplaceAllUsesWith - Just like replaceAllUsesWith but dangerous.
  // Only use when in type resolution situations!
  void uncheckedReplaceAllUsesWith(Value *V);

  //----------------------------------------------------------------------
  // Methods for handling the vector of uses of this Value.
  //
  typedef UseListIteratorWrapper      use_iterator;
  typedef UseListConstIteratorWrapper use_const_iterator;

  unsigned           use_size()  const { return Uses.size();  }
  bool               use_empty() const { return Uses.empty(); }
  use_iterator       use_begin()       { return Uses.begin(); }
  use_const_iterator use_begin() const { return Uses.begin(); }
  use_iterator       use_end()         { return Uses.end();   }
  use_const_iterator use_end()   const { return Uses.end();   }
  User             *use_back()         { return Uses.back().getUser(); }
  const User       *use_back()  const  { return Uses.back().getUser(); }

  /// hasOneUse - Return true if there is exactly one user of this value.  This
  /// is specialized because it is a common request and does not require
  /// traversing the whole use list.
  ///
  bool hasOneUse() const {
    iplist<Use>::const_iterator I = Uses.begin(), E = Uses.end();
    if (I == E) return false;
    return ++I == E;
  }

  /// addUse/killUse - These two methods should only be used by the Use class.
  ///
  void addUse(Use &U)  { Uses.push_back(&U); }
  void killUse(Use &U) { Uses.remove(&U); }
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


inline User *UseListIteratorWrapper::operator*() const {
  return Super::operator*().getUser();
}

inline const User *UseListConstIteratorWrapper::operator*() const {
  return Super::operator*().getUser();
}


Use::Use(Value *v, User *user) : Val(v), U(user) {
  if (Val) Val->addUse(*this);
}

Use::Use(const Use &u) : Val(u.Val), U(u.U) {
  if (Val) Val->addUse(*this);
}

Use::~Use() {
  if (Val) Val->killUse(*this);
}

void Use::set(Value *V) { 
  if (Val) Val->killUse(*this);
  Val = V;
  if (V) V->addUse(*this);
}


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

} // End llvm namespace

#endif
