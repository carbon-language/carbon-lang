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
#include "llvm/Support/Casting.h"
#include <string>

namespace llvm {

class Constant;
class Argument;
class Instruction;
class BasicBlock;
class GlobalValue;
class Function;
class GlobalVariable;
class InlineAsm;
class SymbolTable;

//===----------------------------------------------------------------------===//
//                                 Value Class
//===----------------------------------------------------------------------===//

/// Value - The base class of all values computed by a program that may be used
/// as operands to other values.
///
class Value {
  unsigned short SubclassID;         // Subclass identifier (for isa/dyn_cast)
protected:
  /// SubclassData - This member is defined by this class, but is not used for
  /// anything.  Subclasses can use it to hold whatever state they find useful.
  /// This field is initialized to zero by the ctor.
  unsigned short SubclassData;
private:
  PATypeHolder Ty;
  Use *UseList;

  friend class ValueSymbolTable; // Allow ValueSymbolTable to directly mod Name.
  friend class SymbolTable;      // Allow SymbolTable to directly poke Name.
  std::string Name;

  void operator=(const Value &);     // Do not implement
  Value(const Value &);              // Do not implement

public:
  Value(const Type *Ty, unsigned scid, const std::string &name = "");
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

  void setName(const std::string &name);

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
  typedef value_use_iterator<User>       use_iterator;
  typedef value_use_iterator<const User> use_const_iterator;

  bool               use_empty() const { return UseList == 0; }
  use_iterator       use_begin()       { return use_iterator(UseList); }
  use_const_iterator use_begin() const { return use_const_iterator(UseList); }
  use_iterator       use_end()         { return use_iterator(0);   }
  use_const_iterator use_end()   const { return use_const_iterator(0);   }
  User              *use_back()        { return *use_begin(); }
  const User        *use_back() const  { return *use_begin(); }

  /// hasOneUse - Return true if there is exactly one user of this value.  This
  /// is specialized because it is a common request and does not require
  /// traversing the whole use list.
  ///
  bool hasOneUse() const {
    use_const_iterator I = use_begin(), E = use_end();
    if (I == E) return false;
    return ++I == E;
  }

  /// hasNUses - Return true if this Value has exactly N users.
  ///
  bool hasNUses(unsigned N) const;

  /// hasNUsesOrMore - Return true if this value has N users or more.  This is
  /// logically equivalent to getNumUses() >= N.
  ///
  bool hasNUsesOrMore(unsigned N) const;

  /// getNumUses - This method computes the number of uses of this Value.  This
  /// is a linear time operation.  Use hasOneUse, hasNUses, or hasMoreThanNUses
  /// to check for specific values.
  unsigned getNumUses() const;

  /// addUse/killUse - These two methods should only be used by the Use class.
  ///
  void addUse(Use &U) { U.addToList(&UseList); }

  /// getValueType - Return an ID for the concrete type of this object.  This is
  /// used to implement the classof checks.  This should not be used for any
  /// other purpose, as the values may change as LLVM evolves.  Also, note that
  /// starting with the InstructionVal value, the value stored is actually the
  /// Instruction opcode, so there are more than just these values possible here
  /// (and Instruction must be last).
  ///
  enum ValueTy {
    ArgumentVal,              // This is an instance of Argument
    BasicBlockVal,            // This is an instance of BasicBlock
    FunctionVal,              // This is an instance of Function
    GlobalVariableVal,        // This is an instance of GlobalVariable
    UndefValueVal,            // This is an instance of UndefValue
    ConstantExprVal,          // This is an instance of ConstantExpr
    ConstantAggregateZeroVal, // This is an instance of ConstantAggregateNull
    ConstantBoolVal,          // This is an instance of ConstantBool
    ConstantSIntVal,          // This is an instance of ConstantSInt
    ConstantUIntVal,          // This is an instance of ConstantUInt
    ConstantFPVal,            // This is an instance of ConstantFP
    ConstantArrayVal,         // This is an instance of ConstantArray
    ConstantStructVal,        // This is an instance of ConstantStruct
    ConstantPackedVal,        // This is an instance of ConstantPacked
    ConstantPointerNullVal,   // This is an instance of ConstantPointerNull
    InlineAsmVal,             // This is an instance of InlineAsm
    InstructionVal,           // This is an instance of Instruction
    
    // Markers:
    ConstantFirstVal = FunctionVal,
    ConstantLastVal  = ConstantPointerNullVal
  };
  unsigned getValueType() const {
    return SubclassID;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *) {
    return true; // Values are always values.
  }

  /// getRawType - This should only be used to implement the vmcore library.
  ///
  const Type *getRawType() const { return Ty.getRawType(); }

private:
  /// FIXME: this is a gross hack, needed by another gross hack.  Eliminate!
  void setValueType(unsigned short VT) { SubclassID = VT; }
  friend class Instruction;
};

inline std::ostream &operator<<(std::ostream &OS, const Value &V) {
  V.print(OS);
  return OS;
}

void Use::init(Value *v, User *user) {
  Val = v;
  U = user;
  if (Val) Val->addUse(*this);
}

Use::~Use() {
  if (Val) removeFromList();
}

void Use::set(Value *V) {
  if (Val) removeFromList();
  Val = V;
  if (V) V->addUse(*this);
}


// isa - Provide some specializations of isa so that we don't have to include
// the subtype header files to test to see if the value is a subclass...
//
template <> inline bool isa_impl<Constant, Value>(const Value &Val) {
  return Val.getValueType() >= Value::ConstantFirstVal &&
         Val.getValueType() <= Value::ConstantLastVal;
}
template <> inline bool isa_impl<Argument, Value>(const Value &Val) {
  return Val.getValueType() == Value::ArgumentVal;
}
template <> inline bool isa_impl<InlineAsm, Value>(const Value &Val) {
  return Val.getValueType() == Value::InlineAsmVal;
}
template <> inline bool isa_impl<Instruction, Value>(const Value &Val) {
  return Val.getValueType() >= Value::InstructionVal;
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
