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
#include "llvm/Transforms/IPO.h"
#include "Support/CommandLine.h"
#include <memory>

// InputFilename - The filename to read from.
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bytecode file>"),
              cl::init("-"), cl::value_desc("filename"));
              

// ExtractFunc - The function to extract from the module... defaults to main.
static cl::opt<std::string>
ExtractFunc("func", cl::desc("Specify function to extract"), cl::init("main"),
            cl::value_desc("function"));


struct FunctionExtractorPass : public Pass {
  bool run(Module &M) {
    // Mark all global variables to be internal
    for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
      if (!I->isExternal()) {
        I->setInitializer(0);  // Make all variables external
        I->setInternalLinkage(false); // Make sure it's not internal
      }

    Function *Named = 0;

    // Loop over all of the functions in the module, dropping all references in
    // functions that are not the named function.
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      // Check to see if this is the named function!
      if (I->getName() == ExtractFunc && !I->isExternal()) {
        if (Named) {                            // Two functions, same name?
          std::cerr << "extract ERROR: Two functions named: '" << ExtractFunc
                    << "' found!\n";
          exit(1);
        }

        // Yes, it is.  Keep track of it...
        Named = I;

        // Make sure it's globally accessable...
        Named->setInternalLinkage(false);
      }
    
    if (Named == 0) {
      std::cerr << "Warning: Function '" << ExtractFunc << "' not found!\n";
      return false;
    }
    
    // All of the functions may be used by global variables or the named
    // function.  Loop through them and create a new, external functions that
    // can be "used", instead of ones with bodies.
    //
    std::vector<Function*> NewFunctions;
    
    Function *Last = &M.back();  // Figure out where the last real fn is...

    for (Module::iterator I = M.begin(); ; ++I) {
      if (I->getName() != ExtractFunc) {
        Function *New = new Function(I->getFunctionType(), false, I->getName());
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
        if (I->getName() != ExtractFunc) {
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


static RegisterPass<FunctionExtractorPass> X("extract", "Function Extractor");


int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm extractor\n");

  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    std::cerr << argv[0] << ": bytecode didn't read correctly.\n";
    return 1;
  }

  // In addition to just parsing the input from GCC, we also want to spiff it up
  // a little bit.  Do this now.
  //
  PassManager Passes;
  Passes.add(new FunctionExtractorPass());
  Passes.add(createGlobalDCEPass());              // Delete unreachable globals
  Passes.add(createFunctionResolvingPass());      // Delete prototypes
  Passes.add(createConstantMergePass());          // Merge dup global constants
  Passes.add(createDeadTypeEliminationPass());    // Remove dead types...
  Passes.add(new WriteBytecodePass(&std::cout));  // Write bytecode to file...

  Passes.run(*M.get());
  return 0;
}
