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

std::vector<std::pair<InlineAsm::ConstraintPrefix, std::string> >
InlineAsm::ParseConstraints(const std::string &Constraints) {
  std::vector<std::pair<InlineAsm::ConstraintPrefix, std::string> > Result;
  
  // Scan the constraints string.
  for (std::string::const_iterator I = Constraints.begin(), 
       E = Constraints.end(); I != E; ) {
    if (*I == ',') { Result.clear(); break; } // Empty constraint like ",,"
    
    // Parse the prefix.
    ConstraintPrefix ConstraintType = isInput;
    
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
    
    if (I == E) { Result.clear(); break; }  // Just a prefix, like "==" or "~".
    
    std::string::const_iterator IdStart = I;
      
    // Parse the id.  We accept [a-zA-Z0-9] currently.
    while (I != E && isalnum(*I)) ++I;
    
    if (IdStart == I) {                    // Requires more than just a prefix
      Result.clear();
      break;
    }
    
    // Remember this constraint.
    Result.push_back(std::make_pair(ConstraintType, std::string(IdStart, I)));
    
    // If we reached the end of the ID, we must have the end of the string or a
    // comma, which we skip now.
    if (I != E) {
      if (*I != ',') { Result.clear(); break; }
      ++I;
      if (I == E) { Result.clear(); break; }    // don't allow "xyz,"
    }
  }
  
  return Result;
}


/// Verify - Verify that the specified constraint string is reasonable for the
/// specified function type, and otherwise validate the constraint string.
bool InlineAsm::Verify(const FunctionType *Ty, const std::string &ConstStr) {
  if (Ty->isVarArg()) return false;
  
  std::vector<std::pair<ConstraintPrefix, std::string> >
  Constraints = ParseConstraints(ConstStr);
  
  // Error parsing constraints.
  if (Constraints.empty() && !ConstStr.empty()) return false;
  
  unsigned NumOutputs = 0, NumInputs = 0, NumClobbers = 0;
  
  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    switch (Constraints[i].first) {
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
  }
    
  if (NumOutputs > 1) return false;  // Only one result allowed so far.
  
  if ((Ty->getReturnType() != Type::VoidTy) != NumOutputs)
    return false;   // NumOutputs = 1 iff has a result type.
  
  if (Ty->getNumParams() != NumInputs) return false;
  return true;
}
