//===-- ModuloScheduling.cpp - Software Pipeling Approach - SMS -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// The is a software pipelining pass based on the Swing Modulo Scheduling
// algorithm (SMS).
//
//===----------------------------------------------------------------------===//

#include "ModuloSchedGraph.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"

namespace {
  
  class ModuloScheduling : public FunctionPass {
    
  public:
    virtual bool runOnFunction(Function &F);
  };

  RegisterOpt<ModuloScheduling> X("modulo-sched",
                                  "Modulo Scheduling/Software Pipelining");
}

/// Create Modulo Scheduling Pass
/// 
Pass *createModuloSchedPass() {
  return new ModuloScheduling(); 
}

/// ModuloScheduling::runOnFunction - main transformation entry point
///
bool ModuloScheduling::runOnFunction(Function &F) {
  bool Changed = false;
  return Changed;
}
