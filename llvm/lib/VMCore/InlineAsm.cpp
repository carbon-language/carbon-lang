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

/// Parse - Analyze the specified string (e.g. "==&{eax}") and fill in the
/// fields in this structure.  If the constraint string is not understood,
/// return true, otherwise return false.
bool InlineAsm::ConstraintInfo::Parse(const std::string &Str) {
  std::string::const_iterator I = Str.begin(), E = Str.end();
  
  // Initialize
  Type = isInput;
  isEarlyClobber = false;
  isIndirectOutput =false;
  
  // Parse the prefix.
  if (*I == '~') {
    Type = isClobber;
    ++I;
  } else if (*I == '=') {
    ++I;
    Type = isOutput;
    if (I != E && *I == '=') {
      isIndirectOutput = true;
      ++I;
    }
  }
  
  if (I == E) return true;  // Just a prefix, like "==" or "~".
  
  // Parse the modifiers.
  bool DoneWithModifiers = false;
  while (!DoneWithModifiers) {
    switch (*I) {
    default:
      DoneWithModifiers = true;
      break;
    case '&':
      if (Type != isOutput ||      // Cannot early clobber anything but output.
          isEarlyClobber)          // Reject &&&&&&
        return true;
      isEarlyClobber = true;
      break;
    }
    
    if (!DoneWithModifiers) {
      ++I;
      if (I == E) return true;   // Just prefixes and modifiers!
    }
  }
  
  // Parse the various constraints.
  while (I != E) {
    if (*I == '{') {   // Physical register reference.
      // Find the end of the register name.
      std::string::const_iterator ConstraintEnd = std::find(I+1, E, '}');
      if (ConstraintEnd == E) return true;  // "{foo"
      Codes.push_back(std::string(I, ConstraintEnd+1));
      I = ConstraintEnd+1;
    } else if (isdigit(*I)) {
      // Maximal munch numbers.
      std::string::const_iterator NumStart = I;
      while (I != E && isdigit(*I))
        ++I;
      Codes.push_back(std::string(NumStart, I));
    } else {
      // Single letter constraint.
      Codes.push_back(std::string(I, I+1));
      ++I;
    }
  }

  return false;
}

std::vector<InlineAsm::ConstraintInfo>
InlineAsm::ParseConstraints(const std::string &Constraints) {
  std::vector<ConstraintInfo> Result;
  
  // Scan the constraints string.
  for (std::string::const_iterator I = Constraints.begin(), 
       E = Constraints.end(); I != E; ) {
    ConstraintInfo Info;

    // Find the end of this constraint.
    std::string::const_iterator ConstraintEnd = std::find(I, E, ',');

    if (ConstraintEnd == I ||  // Empty constraint like ",,"
        Info.Parse(std::string(I, ConstraintEnd))) {   // Erroneous constraint?
      Result.clear();
      break;
    }

    Result.push_back(Info);
    
    // ConstraintEnd may be either the next comma or the end of the string.  In
    // the former case, we skip the comma.
    I = ConstraintEnd;
    if (I != E) {
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
  
  std::vector<ConstraintInfo> Constraints = ParseConstraints(ConstStr);
  
  // Error parsing constraints.
  if (Constraints.empty() && !ConstStr.empty()) return false;
  
  unsigned NumOutputs = 0, NumInputs = 0, NumClobbers = 0;
  
  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    switch (Constraints[i].Type) {
    case InlineAsm::isOutput:
      if (!Constraints[i].isIndirectOutput) {
        if (NumInputs || NumClobbers) return false;  // outputs come first.
        ++NumOutputs;
        break;
      }
      // FALLTHROUGH for IndirectOutputs.
    case InlineAsm::isInput:
      if (NumClobbers) return false;               // inputs before clobbers.
      ++NumInputs;
      break;
    case InlineAsm::isClobber:
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
