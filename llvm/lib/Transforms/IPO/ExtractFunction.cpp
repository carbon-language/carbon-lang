
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/IPO.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"

namespace llvm {

namespace {
  class FunctionExtractorPass : public Pass {
    Function *Named;
  public:
    FunctionExtractorPass(Function *F = 0) : Named(F) {}

    bool run(Module &M) {
      if (Named == 0) {
        Named = M.getMainFunction();
        if (Named == 0) return false;  // No function to extract
      }

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
      //
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

Pass *createFunctionExtractionPass(Function *F) {
  return new FunctionExtractorPass(F);
}

} // End llvm namespace
