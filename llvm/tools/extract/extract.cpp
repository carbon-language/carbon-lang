//===----------------------------------------------------------------------===//
// LLVM extract Utility
//
// This utility changes the input module to only contain a single function,
// which is primarily used for debugging transformations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO.h"
#include "Support/CommandLine.h"
#include <memory>

// InputFilename - The filename to read from.
static cl::opt<string>
InputFilename(cl::Positional, cl::desc("<input bytecode file>"),
              cl::init("-"), cl::value_desc("filename"));
              

// ExtractFunc - The function to extract from the module... defaults to main.
static cl::opt<string>
ExtractFunc("func", cl::desc("Specify function to extract"), cl::init("main"),
            cl::value_desc("function"));


struct FunctionExtractorPass : public Pass {
  bool run(Module &M) {
    // Mark all global variables to be internal
    for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
      I->setInternalLinkage(true);

    Function *Named = 0;

    // Loop over all of the functions in the module, dropping all references in
    // functions that are not the named function.
    for (Module::iterator I = M.begin(), E = M.end(); I != E;)
      // Check to see if this is the named function!
      if (!Named && I->getName() == ExtractFunc) {
        // Yes, it is.  Keep track of it...
        Named = I;

        // Make sure it's globally accessable...
        Named->setInternalLinkage(false);

        // Remove the named function from the module.
        M.getFunctionList().remove(I);
      } else {
        // Nope it's not the named function, delete the body of the function
        I->dropAllReferences();
        ++I;
      }

    // All of the functions that still have uses now must be used by global
    // variables or the named function.  Loop through them and create a new,
    // external function for the used ones... making all uses point to the new
    // functions.
    std::vector<Function*> NewFunctions;
    
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      if (!I->use_empty()) {
        Function *New = new Function(I->getFunctionType(), false, I->getName());
        I->replaceAllUsesWith(New);
        NewFunctions.push_back(New);
      }
    
    // Now the module only has unused functions with their references dropped.
    // Delete them all now!
    M.getFunctionList().clear();

    // Re-insert the named function...
    if (Named)
      M.getFunctionList().push_back(Named);
    else
      std::cerr << "Warning: Function '" << ExtractFunc << "' not found!\n";
    
    // Insert all of the function stubs...
    M.getFunctionList().insert(M.end(), NewFunctions.begin(),
                               NewFunctions.end());
    return true;
  }
};


static RegisterPass<FunctionExtractorPass> X("extract", "Function Extractor");


int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm extractor\n");

  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    std::cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  // In addition to just parsing the input from GCC, we also want to spiff it up
  // a little bit.  Do this now.
  //
  PassManager Passes;
  Passes.add(new FunctionExtractorPass());
  Passes.add(createGlobalDCEPass());              // Delete unreachable globals
  Passes.add(createConstantMergePass());          // Merge dup global constants
  Passes.add(createDeadTypeEliminationPass());    // Remove dead types...
  Passes.add(new WriteBytecodePass(&std::cout));  // Write bytecode to file...

  Passes.run(*M.get());
  return 0;
}
