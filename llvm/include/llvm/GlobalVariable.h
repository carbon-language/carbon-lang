//===-- llvm/Global.h - Class to represent a global variable ----*- C++ -*-===//
//
// This file contains the declaration of the GlobalVariable class, which
// represents a single global variable (or constant) in the VM.
//
// Global variables are constant pointers that refer to hunks of space that are
// allocated by either the VM, or by the linker in a static compiler.  A global
// variable may have an intial value, which is copied into the executables .data
// area.  Global Constants are required to have initializers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GLOBAL_VARIABLE_H
#define LLVM_GLOBAL_VARIABLE_H

#include "llvm/GlobalValue.h"

class Module;
class Constant;
class PointerType;
template<typename SC> struct ilist_traits;
template<typename ValueSubClass, typename ItemParentClass, typename SymTabClass,
         typename SubClass> class SymbolTableListTraits;

class GlobalVariable : public GlobalValue {
  friend class SymbolTableListTraits<GlobalVariable, Module, Module,
                                     ilist_traits<GlobalVariable> >;
  void setParent(Module *parent);

  GlobalVariable *Prev, *Next;
  void setNext(GlobalVariable *N) { Next = N; }
  void setPrev(GlobalVariable *N) { Prev = N; }

  bool isConstantGlobal;               // Is this a global constant?
public:
  /// GlobalVariable ctor - If a parent module is specified, the global is
  /// automatically inserted into the end of the specified modules global list.
  ///
  GlobalVariable(const Type *Ty, bool isConstant, LinkageTypes Linkage,
		 Constant *Initializer = 0, const std::string &Name = "",
                 Module *Parent = 0);

  // Specialize setName to handle symbol table majik...
  virtual void setName(const std::string &name, SymbolTable *ST = 0);

  /// isExternal - Is this global variable lacking an initializer?  If so, the
  /// global variable is defined in some other translation unit, and is thus
  /// externally defined here.
  ///
  virtual bool isExternal() const { return Operands.empty(); }

  /// hasInitializer - Unless a global variable isExternal(), it has an
  /// initializer.  The initializer for the global variable/constant is held by
  /// Operands[0] if an initializer is specified.
  ///
  inline bool hasInitializer() const { return !isExternal(); }

  /// getInitializer - Return the initializer for this global variable.  It is
  /// illegal to call this method if the global is external, because we cannot
  /// tell what the value is initialized to!
  ///
  inline Constant *getInitializer() const {
    assert(hasInitializer() && "GV doesn't have initializer!");
    return (Constant*)Operands[0].get();
  }
  inline Constant *getInitializer() {
    assert(hasInitializer() && "GV doesn't have initializer!");
    return (Constant*)Operands[0].get();
  }
  inline void setInitializer(Constant *CPV) {
    if (CPV == 0) {
      if (hasInitializer()) Operands.pop_back();
    } else {
      if (!hasInitializer()) Operands.push_back(Use(0, this));
      Operands[0] = (Value*)CPV;
    }
  }

  // getNext/Prev - Return the next or previous global variable in the list.
        GlobalVariable *getNext()       { return Next; }
  const GlobalVariable *getNext() const { return Next; }
        GlobalVariable *getPrev()       { return Prev; }
  const GlobalVariable *getPrev() const { return Prev; }

  /// If the value is a global constant, its value is immutable throughout the
  /// runtime execution of the program.  Assigning a value into the constant
  /// leads to undefined behavior.
  ///
  bool isConstant() const { return isConstantGlobal; }
  void setConstant(bool Value) { isConstantGlobal = Value; }
  
  virtual void print(std::ostream &OS) const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GlobalVariable *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::GlobalVariableVal;
  }
};

#endif
