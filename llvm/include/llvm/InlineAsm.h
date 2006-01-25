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
// used as the callee operand of call instructions.  InlineAsm's are uniqued
// like constants, and created via InlineAsm::get(...).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INLINEASM_H
#define LLVM_INLINEASM_H

#include "llvm/Value.h"

namespace llvm {

struct AssemblyAnnotationWriter;
class PointerType;
class FunctionType;
class Module;

class InlineAsm : public Value {
  InlineAsm(const InlineAsm &);             // do not implement
  void operator=(const InlineAsm&);         // do not implement

  std::string AsmString, Constraints;
  bool HasSideEffects;
  
  InlineAsm(const FunctionType *Ty, const std::string &AsmString,
            const std::string &Constraints, bool hasSideEffects);
public:

  /// InlineAsm::get - Return the the specified uniqued inline asm string.
  ///
  static InlineAsm *get(const FunctionType *Ty, const std::string &AsmString,
                        const std::string &Constraints, bool hasSideEffects);
  
  bool hasSideEffects() const { return HasSideEffects; }
  
  /// getType - InlineAsm's are always pointers.
  ///
  const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(Value::getType());
  }
  
  /// getFunctionType - InlineAsm's are always pointers to functions.
  ///
  const FunctionType *getFunctionType() const;

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
