//===-- ExtractFunction.cpp - Function extraction pass --------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass extracts
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO.h"
using namespace llvm;

namespace {
  class FunctionExtractorPass : public Pass {
    Function *Named;
    bool deleteFunc;
  public:
    /// FunctionExtractorPass - If deleteFn is true, this pass deletes as the
    /// specified function. Otherwise, it deletes as much of the module as
    /// possible, except for the function specified.
    ///
    FunctionExtractorPass(Function *F = 0, bool deleteFn = true) 
      : Named(F), deleteFunc(deleteFn) {}

    bool run(Module &M) {
      if (Named == 0) {
        Named = M.getMainFunction();
        if (Named == 0) return false;  // No function to extract
      }

      if (deleteFunc)
        return deleteFunction();
      else 
        return isolateFunction(M);
    }

    bool deleteFunction() {
      Named->setLinkage(GlobalValue::ExternalLinkage);
      Named->deleteBody();
      assert(Named->isExternal() && "This didn't make the function external!");
      return true;
    }

    bool isolateFunction(Module &M) {
      // Make sure our result is globally accessible...
      Named->setLinkage(GlobalValue::ExternalLinkage);

      // Mark all global variables internal
      for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
        if (!I->isExternal()) {
          I->setInitializer(0);  // Make all variables external
          I->setLinkage(GlobalValue::ExternalLinkage);
        }
      
      // All of the functions may be used by global variables or the named
      // function.  Loop through them and create a new, external functions that
      // can be "used", instead of ones with bodies.
      std::vector<Function*> NewFunctions;
      
      Function *Last = &M.back();  // Figure out where the last real fn is...
      
      for (Module::iterator I = M.begin(); ; ++I) {
        if (&*I != Named) {
          Function *New = new Function(I->getFunctionType(),
                                       GlobalValue::ExternalLinkage,
                                       I->getName());
          I->setName("");  // Remove Old name
          
          // If it's not the named function, delete the body of the function
          I->dropAllReferences();
          
          M.getFunctionList().push_back(New);
          NewFunctions.push_back(New);
        }
        
        if (&*I == Last) break;  // Stop after processing the last function
      }
      
      // Now that we have replacements all set up, loop through the module,
      // deleting the old functions, replacing them with the newly created
      // functions.
      if (!NewFunctions.empty()) {
        unsigned FuncNum = 0;
        Module::iterator I = M.begin();
        do {
          if (&*I != Named) {
            // Make everything that uses the old function use the new dummy fn
            I->replaceAllUsesWith(NewFunctions[FuncNum++]);
            
            Function *Old = I;
            ++I;  // Move the iterator to the new function
            
            // Delete the old function!
            M.getFunctionList().erase(Old);
            
          } else {
            ++I;  // Skip the function we are extracting
          }
        } while (&*I != NewFunctions[0]);
      }
      
      return true;
    }
  };

  RegisterPass<FunctionExtractorPass> X("extract", "Function Extractor");
}

Pass *llvm::createFunctionExtractionPass(Function *F, bool deleteFn) {
  return new FunctionExtractorPass(F, deleteFn);
}
