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
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Type.h"
#include "llvm/Constant.h"

/// deleteInstructionFromProgram - This method clones the current Program and
/// deletes the specified instruction from the cloned module.  It then runs a
/// series of cleanup passes (ADCE and SimplifyCFG) to eliminate any code which
/// depends on the value.  The modified module is then returned.
///
Module *BugDriver::deleteInstructionFromProgram(Instruction *I,
                                                unsigned Simplification) const {
  Module *Result = CloneModule(Program);

  BasicBlock *PBB = I->getParent();
  Function *PF = PBB->getParent();

  Module::iterator RFI = Result->begin(); // Get iterator to corresponding fn
  std::advance(RFI, std::distance(Program->begin(), Module::iterator(PF)));

  Function::iterator RBI = RFI->begin();  // Get iterator to corresponding BB
  std::advance(RBI, std::distance(PF->begin(), Function::iterator(PBB)));

  BasicBlock::iterator RI = RBI->begin(); // Get iterator to corresponding inst
  std::advance(RI, std::distance(PBB->begin(), BasicBlock::iterator(I)));
  I = RI;                                 // Got the corresponding instruction!

  // If this instruction produces a value, replace any users with null values
  if (I->getType() != Type::VoidTy)
    I->replaceAllUsesWith(Constant::getNullValue(I->getType()));

  // Remove the instruction from the program.
  I->getParent()->getInstList().erase(I);

  // Spiff up the output a little bit.
  PassManager Passes;
  if (Simplification > 2)
    Passes.add(createAggressiveDCEPass());          // Remove dead code...
  //Passes.add(createInstructionCombiningPass());
  if (Simplification > 1)
    Passes.add(createDeadCodeEliminationPass());
  if (Simplification)
    Passes.add(createCFGSimplificationPass());      // Delete dead control flow

  Passes.add(createVerifierPass());
  Passes.run(*Result);
  return Result;
}

/// performFinalCleanups - This method clones the current Program and performs
/// a series of cleanups intended to get rid of extra cruft on the module
/// before handing it to the user...
///
Module *BugDriver::performFinalCleanups() const {
  PassManager CleanupPasses;
  CleanupPasses.add(createFunctionResolvingPass());
  CleanupPasses.add(createGlobalDCEPass());
  CleanupPasses.add(createVerifierPass());
  Module *M = CloneModule(Program);
  CleanupPasses.run(*M);
  return M;
}
