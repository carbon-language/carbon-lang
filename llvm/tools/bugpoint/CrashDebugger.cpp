//===- CrashDebugger.cpp - Debug compilation crashes ----------------------===//
//
// This file defines the bugpoint internals that narrow down compilation crashes
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "SystemUtils.h"
#include "llvm/Module.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Pass.h"
#include <fstream>

/// debugCrash - This method is called when some pass crashes on input.  It
/// attempts to prune down the testcase to something reasonable, and figure
/// out exactly which pass is crashing.
///
bool BugDriver::debugCrash() {
  std::cout << "\n*** Debugging optimizer crash!\n";

  // Determine which pass causes the optimizer to crash... using binary search
  unsigned LastToPass = 0, LastToCrash = PassesToRun.size();
  while (LastToPass != LastToCrash) {
    unsigned Mid = (LastToCrash+LastToPass+1) / 2;
    std::vector<const PassInfo*> P(PassesToRun.begin(),
                                   PassesToRun.begin()+Mid);
    std::cout << "Checking to see if the first " << Mid << " passes crash: ";

    if (runPasses(P))
      LastToCrash = Mid-1;
    else
      LastToPass = Mid;
  }

  // Make sure something crashed.  :)
  if (LastToCrash >= PassesToRun.size()) {
    std::cerr << "ERROR: No passes crashed!\n";
    return true;
  }

  // Calculate which pass it is that crashes...
  const PassInfo *CrashingPass = PassesToRun[LastToCrash];
  
  std::cout << "\n*** Found crashing pass '-" << CrashingPass->getPassArgument()
            << "': " << CrashingPass->getPassName() << "\n";

  // Compile the program with just the passes that don't crash.
  if (LastToPass != 0) { // Don't bother doing this if the first pass crashes...
    std::vector<const PassInfo*> P(PassesToRun.begin(), 
                                   PassesToRun.begin()+LastToPass);
    std::string Filename;
    std::cout << "Running passes that don't crash to get input for pass: ";
    if (runPasses(P, Filename)) {
      std::cerr << "ERROR: Running the first " << LastToPass
                << " passes crashed this time!\n";
      return true;
    }

    // Assuming everything was successful, we now have a valid bytecode file in
    // OutputName.  Use it for "Program" Instead.
    delete Program;
    Program = ParseInputFile(Filename);

    // Delete the file now.
    removeFile(Filename);
  }

  return debugPassCrash(CrashingPass);
}

/// CountFunctions - return the number of non-external functions defined in the
/// module.
static unsigned CountFunctions(Module *M) {
  unsigned N = 0;
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isExternal())
      ++N;
  return N;
}

/// debugPassCrash - This method is called when the specified pass crashes on
/// Program as input.  It tries to reduce the testcase to something that still
/// crashes, but it smaller.
///
bool BugDriver::debugPassCrash(const PassInfo *Pass) {
  EmitProgressBytecode(Pass, "passinput");

  if (CountFunctions(Program) > 1) {
    // Attempt to reduce the input program down to a single function that still
    // crashes.  Do this by removing everything except for that one function...
    //
    std::cout << "\n*** Attempting to reduce the testcase to one function\n";

    for (Module::iterator I = Program->begin(), E = Program->end(); I != E; ++I)
      if (!I->isExternal()) {
        // Extract one function from the module...
        Module *M = extractFunctionFromModule(I);

        // Make the function the current program...
        std::swap(Program, M);
        
        // Find out if the pass still crashes on this pass...
        std::cout << "Checking function '" << I->getName() << "': ";
        if (runPass(Pass)) {
          // Yup, it does, we delete the old module, and continue trying to
          // reduce the testcase...
          delete M;

          EmitProgressBytecode(Pass, "reduced-"+I->getName());
          break;
        }
        
        // This pass didn't crash on this function, try the next one.
        delete Program;
        Program = M;
      }
  }

  if (CountFunctions(Program) > 1) {
    std::cout << "\n*** Couldn't reduce testcase to one function.\n"
	      << "    Attempting to remove individual functions.\n";
    std::cout << "XXX Individual function removal unimplemented!\n";
  }

  // Now that we have deleted the functions that are unneccesary for the
  // program, try to remove instructions and basic blocks that are not neccesary
  // to cause the crash.
  //
  
  return false;
}
