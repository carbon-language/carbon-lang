//===- ExtractFunction.cpp - Extract a function from Program --------------===//
//
// This file implements a method that extracts a function from program, cleans
// it up, and returns it as a new module.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Cloning.h"

/// extractFunctionFromModule - This method is used to extract the specified
/// (non-external) function from the current program, slim down the module, and
/// then return it.  This does not modify Program at all, it modifies a copy,
/// which it returns.
Module *BugDriver::extractFunctionFromModule(Function *F) const {
  Module *Result = CloneModule(Program);

  // Translate from the old module to the new copied module...
  F = Result->getFunction(F->getName(), F->getFunctionType());

  // In addition to just parsing the input from GCC, we also want to spiff it up
  // a little bit.  Do this now.
  //
  PassManager Passes;
  Passes.add(createFunctionExtractionPass(F));    // Extract the function
  Passes.add(createGlobalDCEPass());              // Delete unreachable globals
  Passes.add(createFunctionResolvingPass());      // Delete prototypes
  Passes.add(createDeadTypeEliminationPass());    // Remove dead types...
  Passes.run(*Result);
  return Result;
}
