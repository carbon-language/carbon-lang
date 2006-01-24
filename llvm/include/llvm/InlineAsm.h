//===-- llvm/InlineAsm.h - Class to represent inline asm strings-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class represents the inline asm strings, which are Value*'s that are
// used as the callee operand of call instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INLINEASM_H
#define LLVM_INLINEASM_H

#include "llvm/Value.h"

namespace llvm {

class AssemblyAnnotationWriter;
class PointerType;
class FunctionType;
class Module;
template<typename SC> struct ilist_traits;
template<typename ValueSubClass, typename ItemParentClass, typename SymTabClass,
         typename SubClass> class SymbolTableListTraits;

class InlineAsm : public Value {
  friend class SymbolTableListTraits<InlineAsm, Module, Module,
                                     ilist_traits<InlineAsm> >;
  InlineAsm(const InlineAsm &);             // do not implement
  void operator=(const InlineAsm&);         // do not implement

  void setParent(Module *Parent);
  InlineAsm *Prev, *Next;
  void setNext(InlineAsm *N) { Next = N; }
  void setPrev(InlineAsm *N) { Prev = N; }
        InlineAsm *getNext()       { return Next; }
  const InlineAsm *getNext() const { return Next; }
        InlineAsm *getPrev()       { return Prev; }
  const InlineAsm *getPrev() const { return Prev; }
  
  Module *Parent;
  std::string AsmString, Constraints;
  bool AsmHasSideEffects;
public:
  InlineAsm(const FunctionType *Ty, const std::string &AsmString,
            const std::string &Constraints, bool hasSideEffects,
            const std::string &Name = "", Module *ParentModule = 0);
  
  bool getHasSideEffects() const { return AsmHasSideEffects; }
  void setSideEffects(bool X) { AsmHasSideEffects = X; }
  
  /// getType - InlineAsm's are always pointers.
  ///
  const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(Value::getType());
  }
  
  /// getFunctionType - InlineAsm's are always pointers to functions.
  ///
  const FunctionType *getFunctionType() const;

  /// getParent - Get the module that this global value is contained inside
  /// of...
  Module *getParent() { return Parent; }
  const Module *getParent() const { return Parent; }

  
  /// removeFromParent/eraseFromParent - Unlink and unlink/delete this object
  /// from the module it is embedded into.
  void removeFromParent();
  void eraseFromParent();
  
  virtual void print(std::ostream &O) const { print(O, 0); }
  void print(std::ostream &OS, AssemblyAnnotationWriter *AAW) const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const InlineAsm *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::InlineAsmVal;
  }
};

} // End llvm namespace

#endif
