//===- PassManager.cpp - LLVM Pass Infrastructure Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM Pass Manager infrastructure. 
//
//===----------------------------------------------------------------------===//


#include "llvm/PassManager.h"
#include "llvm/Function.h"
#include "llvm/Module.h"

using namespace llvm;

/// BasicBlockPassManager implementation

/// Add pass P into PassVector and return TRUE. If this pass is not
/// manageable by this manager then return FALSE.
bool
BasicBlockPassManager_New::addPass (Pass *P) {

  BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);
  if (!BP)
    return false;

  // TODO: Check if it suitable to manage P using this BasicBlockPassManager
  // or we need another instance of BasicBlockPassManager

  // Add pass
  PassVector.push_back(BP);
  return true;
}

/// Execute all of the passes scheduled for execution by invoking 
/// runOnBasicBlock method.  Keep track of whether any of the passes modifies 
/// the function, and if so, return true.
bool
BasicBlockPassManager_New::runOnFunction(Function &F) {

  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    for (std::vector<Pass *>::iterator itr = PassVector.begin(),
           e = PassVector.end(); itr != e; ++itr) {
      Pass *P = *itr;
      BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);
      Changed |= BP->runOnBasicBlock(*I);
    }
  return Changed;
}

