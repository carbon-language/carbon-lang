//===-- InlineAsm.cpp - Implement the InlineAsm class ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the InlineAsm class.
//
//===----------------------------------------------------------------------===//

#include "llvm/InlineAsm.h"
#include "llvm/DerivedTypes.h"
#include <cctype>
using namespace llvm;

// NOTE: when memoizing the function type, we have to be careful to handle the
// case when the type gets refined.

InlineAsm *InlineAsm::get(const FunctionType *Ty, const std::string &AsmString,
                          const std::string &Constraints, bool hasSideEffects) {
  // FIXME: memoize!
  return new InlineAsm(Ty, AsmString, Constraints, hasSideEffects);  
}

InlineAsm::InlineAsm(const FunctionType *Ty, const std::string &asmString,
                     const std::string &constraints, bool hasSideEffects)
  : Value(PointerType::get(Ty), Value::InlineAsmVal), AsmString(asmString), 
    Constraints(constraints), HasSideEffects(hasSideEffects) {

  // Do various checks on the constraint string and type.
  assert(Verify(Ty, constraints) && "Function type not legal for constraints!");
}

const FunctionType *InlineAsm::getFunctionType() const {
  return cast<FunctionType>(getType()->getElementType());
}

/// Verify - Verify that the specified constraint string is reasonable for the
/// specified function type, and otherwise validate the constraint string.
bool InlineAsm::Verify(const FunctionType *Ty, const std::string &Constraints) {
  if (Ty->isVarArg()) return false;
  
  unsigned NumOutputs = 0, NumInputs = 0, NumClobbers = 0;
  
  // Scan the constraints string.
  for (std::string::const_iterator I = Constraints.begin(), 
         E = Constraints.end(); I != E; ) {
    if (*I == ',') return false;  // Empty constraint like ",,"
    
    // Parse the prefix.
    enum {
      isInput,            // 'x'
      isOutput,           // '=x'
      isIndirectOutput,   // '==x'
      isClobber,          // '~x'
    } ConstraintType = isInput;
    
    if (*I == '~') {
      ConstraintType = isClobber;
      ++I;
    } else if (*I == '=') {
      ++I;
      if (I != E && *I == '=') {
        ConstraintType = isIndirectOutput;
        ++I;
      } else {
        ConstraintType = isOutput;
      }
    }
    
    if (I == E) return false;   // Just a prefix, like "==" or "~".
    
    switch (ConstraintType) {
    case isOutput:
      if (NumInputs || NumClobbers) return false;  // outputs come first.
      ++NumOutputs;
      break;
    case isInput:
    case isIndirectOutput:
      if (NumClobbers) return false;               // inputs before clobbers.
      ++NumInputs;
      break;
    case isClobber:
      ++NumClobbers;
      break;
    }
    
    // Parse the id.  We accept [a-zA-Z0-9] currently.
    while (I != E && isalnum(*I)) ++I;
    
    // If we reached the end of the ID, we must have the end of the string or a
    // comma, which we skip now.
    if (I != E) {
      if (*I != ',') return false;
      ++I;
      if (I == E) return false;    // don't allow "xyz,"
    }
  }
  
  if (NumOutputs > 1) return false;  // Only one result allowed.
  
  if ((Ty->getReturnType() != Type::VoidTy) != NumOutputs)
    return false;   // NumOutputs = 1 iff has a result type.
  
  if (Ty->getNumParams() != NumInputs) return false;
  return true;
}
