//===- ExtractFunction.cpp - Extract a function from Program --------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements several methods that are used to extract functions,
// loops, or portions of a module from the rest of the module.
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
#include "Support/Debug.h"
#include "Support/FileUtilities.h"
using namespace llvm;

namespace llvm {
  bool DisableSimplifyCFG = false;
} // End llvm namespace

namespace {
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
Module *BugDriver::deleteInstructionFromProgram(const Instruction *I,
                                                unsigned Simplification) const {
  Module *Result = CloneModule(Program);

  const BasicBlock *PBB = I->getParent();
  const Function *PF = PBB->getParent();

  Module::iterator RFI = Result->begin(); // Get iterator to corresponding fn
  std::advance(RFI, std::distance(PF->getParent()->begin(),
                                  Module::const_iterator(PF)));

  Function::iterator RBI = RFI->begin();  // Get iterator to corresponding BB
  std::advance(RBI, std::distance(PF->begin(), Function::const_iterator(PBB)));

  BasicBlock::iterator RI = RBI->begin(); // Get iterator to corresponding inst
  std::advance(RI, std::distance(PBB->begin(), BasicBlock::const_iterator(I)));
  Instruction *TheInst = RI;              // Got the corresponding instruction!

  // If this instruction produces a value, replace any users with null values
  if (TheInst->getType() != Type::VoidTy)
    TheInst->replaceAllUsesWith(Constant::getNullValue(TheInst->getType()));

  // Remove the instruction from the program.
  TheInst->getParent()->getInstList().erase(TheInst);

  // Spiff up the output a little bit.
  PassManager Passes;
  // Make sure that the appropriate target data is always used...
  Passes.add(new TargetData("bugpoint", Result));

  /// FIXME: If this used runPasses() like the methods below, we could get rid
  /// of the -disable-* options!
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

  Module *New = runPassesOn(M, CleanupPasses);
  if (New == 0) {
    std::cerr << "Final cleanups failed.  Sorry. :(  Please report a bug!\n";
  }
  delete M;
  return New;
}


/// ExtractLoop - Given a module, extract up to one loop from it into a new
/// function.  This returns null if there are no extractable loops in the
/// program or if the loop extractor crashes.
Module *BugDriver::ExtractLoop(Module *M) {
  std::vector<const PassInfo*> LoopExtractPasses;
  LoopExtractPasses.push_back(getPI(createSingleLoopExtractorPass()));

  Module *NewM = runPassesOn(M, LoopExtractPasses);
  if (NewM == 0) {
    Module *Old = swapProgramIn(M);
    std::cout << "*** Loop extraction failed: ";
    EmitProgressBytecode("loopextraction", true);
    std::cout << "*** Sorry. :(  Please report a bug!\n";
    swapProgramIn(Old);
    return 0;
  }

  // Check to see if we created any new functions.  If not, no loops were
  // extracted and we should return null.
  if (M->size() == NewM->size()) {
    delete NewM;
    return 0;
  }
  
  return NewM;
}


// DeleteFunctionBody - "Remove" the function by deleting all of its basic
// blocks, making it external.
//
void llvm::DeleteFunctionBody(Function *F) {
  // delete the body of the function...
  F->deleteBody();
  assert(F->isExternal() && "This didn't make the function external!");
}

/// SplitFunctionsOutOfModule - Given a module and a list of functions in the
/// module, split the functions OUT of the specified module, and place them in
/// the new module.
///
/// FIXME: this could be made DRAMATICALLY more efficient for large programs if
/// we just MOVED functions from one module to the other, instead of cloning the
/// whole module, then proceeding to delete an entire module's worth of stuff.
///
Module *llvm::SplitFunctionsOutOfModule(Module *M,
                                        const std::vector<Function*> &F) {
  // Make sure functions & globals are all external so that linkage
  // between the two modules will work.
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);
  for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);

  Module *New = CloneModule(M);

  // Make sure global initializers exist only in the safe module (CBE->.so)
  for (Module::giterator I = New->gbegin(), E = New->gend(); I != E; ++I)
    I->setInitializer(0);  // Delete the initializer to make it external

  // Remove the Test functions from the Safe module
  for (unsigned i = 0, e = F.size(); i != e; ++i) {
    Function *TNOF = M->getFunction(F[i]->getName(), F[i]->getFunctionType());
    DEBUG(std::cerr << "Removing function " << F[i]->getName() << "\n");
    assert(TNOF && "Function doesn't exist in module!");
    DeleteFunctionBody(TNOF);       // Function is now external in this module!
  }

  // Remove the Safe functions from the Test module
  for (Module::iterator I = New->begin(), E = New->end(); I != E; ++I) {
    bool funcFound = false;
    for (std::vector<Function*>::const_iterator FI = F.begin(), Fe = F.end();
         FI != Fe; ++FI)
      if (I->getName() == (*FI)->getName() &&
          I->getType() == (*FI)->getType())
        funcFound = true;

    if (!funcFound)
      DeleteFunctionBody(I);
  }
  return New;
}
