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
#include "llvm/System/IncludeFile.h"
#include <vector>

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
  virtual ~InlineAsm();
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
  
  const std::string &getAsmString() const { return AsmString; }
  const std::string &getConstraintString() const { return Constraints; }

  virtual void print(std::ostream &O) const { print(O, 0); }
  void print(std::ostream &OS, AssemblyAnnotationWriter *AAW) const;

  /// Verify - This static method can be used by the parser to check to see if
  /// the specified constraint string is legal for the type.  This returns true
  /// if legal, false if not.
  ///
  static bool Verify(const FunctionType *Ty, const std::string &Constraints);

  // Constraint String Parsing 
  enum ConstraintPrefix {
    isInput,            // 'x'
    isOutput,           // '=x'
    isClobber           // '~x'
  };
  
  struct ConstraintInfo {
    /// Type - The basic type of the constraint: input/output/clobber
    ///
    ConstraintPrefix Type;
    
    /// isEarlyClobber - "&": output operand writes result before inputs are all
    /// read.  This is only ever set for an output operand.
    bool isEarlyClobber; 
    
    /// isIndirectOutput - If this is true for an output constraint, the address
    /// to store the output result is passed as an operand to the call.
    bool isIndirectOutput;
    
    /// hasMatchingInput - This is set to true for an output constraint iff
    /// there is an input constraint that is required to match it (e.g. "0").
    bool hasMatchingInput;
    
    /// isCommutative - This is set to true for a constraint that is commutative
    /// with the next operand.
    bool isCommutative;
    
    /// Code - The constraint code, either the register name (in braces) or the
    /// constraint letter/number.
    std::vector<std::string> Codes;
    
    /// Parse - Analyze the specified string (e.g. "==&{eax}") and fill in the
    /// fields in this structure.  If the constraint string is not understood,
    /// return true, otherwise return false.
    bool Parse(const std::string &Str, 
               std::vector<InlineAsm::ConstraintInfo> &ConstraintsSoFar);
  };
  
  /// ParseConstraints - Split up the constraint string into the specific
  /// constraints and their prefixes.  If this returns an empty vector, and if
  /// the constraint string itself isn't empty, there was an error parsing.
  static std::vector<ConstraintInfo> 
    ParseConstraints(const std::string &ConstraintString);
  
  /// ParseConstraints - Parse the constraints of this inlineasm object, 
  /// returning them the same way that ParseConstraints(str) does.
  std::vector<ConstraintInfo> 
  ParseConstraints() const {
    return ParseConstraints(Constraints);
  }
  
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const InlineAsm *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::InlineAsmVal;
  }
};

} // End llvm namespace

// Make sure the InlineAsm.cpp file is linked when this one is #included.
FORCE_DEFINING_FILE_TO_BE_LINKED(InlineAsm)

#endif
