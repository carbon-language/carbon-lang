//===-- llvm/GlobalVariable.h - GlobalVariable class ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
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

namespace llvm {

class Module;
class Constant;
class PointerType;
template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;

class GlobalVariable : public GlobalValue {
  friend class SymbolTableListTraits<GlobalVariable, Module>;
  void operator=(const GlobalVariable &);     // Do not implement
  GlobalVariable(const GlobalVariable &);     // Do not implement

  void setParent(Module *parent);

  GlobalVariable *Prev, *Next;
  void setNext(GlobalVariable *N) { Next = N; }
  void setPrev(GlobalVariable *N) { Prev = N; }

  bool isConstantGlobal : 1;           // Is this a global constant?
  bool isThreadLocalSymbol : 1;        // Is this symbol "Thread Local"?
  Use Initializer;

public:
  /// GlobalVariable ctor - If a parent module is specified, the global is
  /// automatically inserted into the end of the specified modules global list.
  GlobalVariable(const Type *Ty, bool isConstant, LinkageTypes Linkage,
                 Constant *Initializer = 0, const std::string &Name = "",
                 Module *Parent = 0, bool ThreadLocal = false, 
                 unsigned AddressSpace = 0);
  /// GlobalVariable ctor - This creates a global and inserts it before the
  /// specified other global.
  GlobalVariable(const Type *Ty, bool isConstant, LinkageTypes Linkage,
                 Constant *Initializer, const std::string &Name,
                 GlobalVariable *InsertBefore, bool ThreadLocal = false, 
                 unsigned AddressSpace = 0);
  
  /// isDeclaration - Is this global variable lacking an initializer?  If so, 
  /// the global variable is defined in some other translation unit, and is thus
  /// only a declaration here.
  virtual bool isDeclaration() const { return getNumOperands() == 0; }

  /// hasInitializer - Unless a global variable isExternal(), it has an
  /// initializer.  The initializer for the global variable/constant is held by
  /// Initializer if an initializer is specified.
  ///
  inline bool hasInitializer() const { return !isDeclaration(); }

  /// getInitializer - Return the initializer for this global variable.  It is
  /// illegal to call this method if the global is external, because we cannot
  /// tell what the value is initialized to!
  ///
  inline Constant *getInitializer() const {
    assert(hasInitializer() && "GV doesn't have initializer!");
    return reinterpret_cast<Constant*>(Initializer.get());
  }
  inline Constant *getInitializer() {
    assert(hasInitializer() && "GV doesn't have initializer!");
    return reinterpret_cast<Constant*>(Initializer.get());
  }
  inline void setInitializer(Constant *CPV) {
    if (CPV == 0) {
      if (hasInitializer()) {
        Initializer.set(0);
        NumOperands = 0;
      }
    } else {
      if (!hasInitializer())
        NumOperands = 1;
      Initializer.set(CPV);
    }
  }

  /// If the value is a global constant, its value is immutable throughout the
  /// runtime execution of the program.  Assigning a value into the constant
  /// leads to undefined behavior.
  ///
  bool isConstant() const { return isConstantGlobal; }
  void setConstant(bool Value) { isConstantGlobal = Value; }

  /// If the value is "Thread Local", its value isn't shared by the threads.
  bool isThreadLocal() const { return isThreadLocalSymbol; }
  void setThreadLocal(bool Value) { isThreadLocalSymbol = Value; }

  /// removeFromParent - This method unlinks 'this' from the containing module,
  /// but does not delete it.
  ///
  void removeFromParent();

  /// eraseFromParent - This method unlinks 'this' from the containing module
  /// and deletes it.
  ///
  void eraseFromParent();

  /// Override Constant's implementation of this method so we can
  /// replace constant initializers.
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  virtual void print(std::ostream &OS) const;
  void print(std::ostream *OS) const { if (OS) print(*OS); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GlobalVariable *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::GlobalVariableVal;
  }
private:
  // getNext/Prev - Return the next or previous global variable in the list.
        GlobalVariable *getNext()       { return Next; }
  const GlobalVariable *getNext() const { return Next; }
        GlobalVariable *getPrev()       { return Prev; }
  const GlobalVariable *getPrev() const { return Prev; }
};

} // End llvm namespace

#endif
