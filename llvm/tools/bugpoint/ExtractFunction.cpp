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
#include "Support/FileUtilities.h"
using namespace llvm;

namespace llvm {
  bool DisableSimplifyCFG = false;
} // End llvm namespace

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

static const PassInfo *getPI(Pass *P) {
  const PassInfo *PI = P->getPassInfo();
  delete P;
  return PI;
}

/// performFinalCleanups - This method clones the current Program and performs
/// a series of cleanups intended to get rid of extra cruft on the module
/// before handing it to the user...
///
Module *BugDriver::performFinalCleanups(Module *M, bool MayModifySemantics) {
  // Make all functions external, so GlobalDCE doesn't delete them...
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);
  
  std::vector<const PassInfo*> CleanupPasses;
  CleanupPasses.push_back(getPI(createFunctionResolvingPass()));
  CleanupPasses.push_back(getPI(createGlobalDCEPass()));
  CleanupPasses.push_back(getPI(createDeadTypeEliminationPass()));

  if (MayModifySemantics)
    CleanupPasses.push_back(getPI(createDeadArgHackingPass()));
  else
    CleanupPasses.push_back(getPI(createDeadArgEliminationPass()));
  
  std::swap(Program, M);
  std::string Filename;
  bool Failed = runPasses(CleanupPasses, Filename);
  std::swap(Program, M);

  if (Failed) {
    std::cerr << "Final cleanups failed.  Sorry.  :(\n";
  } else {
    delete M;
    M = ParseInputFile(Filename);
    if (M == 0) {
      std::cerr << getToolName() << ": Error reading bytecode file '"
                << Filename << "'!\n";
      exit(1);
    }
    removeFile(Filename);
  }
  return M;
}
