//===- ExtractFunction.cpp - Extract a function from Program --------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a method that extracts a function from program, cleans
// it up, and returns it as a new module.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "llvm/Constant.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Target/TargetData.h"
#include "Support/CommandLine.h"

bool DisableSimplifyCFG = false;

namespace {
  cl::opt<bool>
  NoADCE("disable-adce",
         cl::desc("Do not use the -adce pass to reduce testcases"));
  cl::opt<bool>
  NoDCE ("disable-dce",
         cl::desc("Do not use the -dce pass to reduce testcases"));
  cl::opt<bool, true>
  NoSCFG("disable-simplifycfg", cl::location(DisableSimplifyCFG),
         cl::desc("Do not use the -simplifycfg pass to reduce testcases"));
  cl::opt<bool>
  NoFinalCleanup("disable-final-cleanup",
                 cl::desc("Disable the final cleanup phase of narrowing"));
}

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
  // Make sure that the appropriate target data is always used...
  Passes.add(new TargetData("bugpoint", Result));

  if (Simplification > 2 && !NoADCE)
    Passes.add(createAggressiveDCEPass());          // Remove dead code...
  //Passes.add(createInstructionCombiningPass());
  if (Simplification > 1 && !NoDCE)
    Passes.add(createDeadCodeEliminationPass());
  if (Simplification && !DisableSimplifyCFG)
    Passes.add(createCFGSimplificationPass());      // Delete dead control flow

  Passes.add(createVerifierPass());
  Passes.run(*Result);
  return Result;
}

/// performFinalCleanups - This method clones the current Program and performs
/// a series of cleanups intended to get rid of extra cruft on the module
/// before handing it to the user...
///
Module *BugDriver::performFinalCleanups(Module *InM) const {
  Module *M = InM ? InM : CloneModule(Program);

  // Allow disabling these passes if they crash bugpoint.
  //
  // FIXME: This should eventually run these passes in a pass list to prevent
  // them from being able to crash bugpoint at all!
  //
  if (NoFinalCleanup) return M;

  // Make all functions external, so GlobalDCE doesn't delete them...
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);
  
  PassManager CleanupPasses;
  // Make sure that the appropriate target data is always used...
  CleanupPasses.add(new TargetData("bugpoint", M));
  CleanupPasses.add(createFunctionResolvingPass());
  CleanupPasses.add(createGlobalDCEPass());
  CleanupPasses.add(createDeadTypeEliminationPass());
  CleanupPasses.add(createDeadArgEliminationPass(InM == 0));
  CleanupPasses.add(createVerifierPass());
  CleanupPasses.run(*M);
  return M;
}
