//===- CrashDebugger.cpp - Debug compilation crashes ----------------------===//
//
// This file defines the bugpoint internals that narrow down compilation crashes
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "SystemUtils.h"
#include "ListReducer.h"
#include "llvm/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Pass.h"
#include <fstream>
#include <set>

class DebugCrashes : public ListReducer<const PassInfo*> {
  BugDriver &BD;
public:
  DebugCrashes(BugDriver &bd) : BD(bd) {}

  // doTest - Return true iff running the "removed" passes succeeds, and running
  // the "Kept" passes fail when run on the output of the "removed" passes.  If
  // we return true, we update the current module of bugpoint.
  //
  virtual TestResult doTest(std::vector<const PassInfo*> &Removed,
                            std::vector<const PassInfo*> &Kept);
};

DebugCrashes::TestResult
DebugCrashes::doTest(std::vector<const PassInfo*> &Prefix,
                     std::vector<const PassInfo*> &Suffix) {
  std::string PrefixOutput;
  if (!Prefix.empty()) {
    std::cout << "Checking to see if these passes crash: "
              << getPassesString(Prefix) << ": ";
    if (BD.runPasses(Prefix, PrefixOutput))
      return KeepPrefix;
  }

  std::cout << "Checking to see if these passes crash: "
            << getPassesString(Suffix) << ": ";
  Module *OrigProgram = BD.Program;
  BD.Program = BD.ParseInputFile(PrefixOutput);
  if (BD.Program == 0) {
    std::cerr << BD.getToolName() << ": Error reading bytecode file '"
              << PrefixOutput << "'!\n";
    exit(1);
  }
  removeFile(PrefixOutput);

  if (BD.runPasses(Suffix)) {
    delete OrigProgram;            // The suffix crashes alone...
    return KeepSuffix;
  }

  // Nothing failed, restore state...
  delete BD.Program;
  BD.Program = OrigProgram;
  return NoFailure;
}

class ReduceCrashingFunctions : public ListReducer<Function*> {
  BugDriver &BD;
public:
  ReduceCrashingFunctions(BugDriver &bd) : BD(bd) {}

  virtual TestResult doTest(std::vector<Function*> &Prefix,
                            std::vector<Function*> &Kept) {
    if (TestFuncs(Kept))
      return KeepSuffix;
    if (!Prefix.empty() && TestFuncs(Prefix))
      return KeepPrefix;
    return NoFailure;
  }
  
  bool TestFuncs(std::vector<Function*> &Prefix);
};

bool ReduceCrashingFunctions::TestFuncs(std::vector<Function*> &Funcs) {
  // Clone the program to try hacking it appart...
  Module *M = CloneModule(BD.Program);
  
  // Convert list to set for fast lookup...
  std::set<Function*> Functions;
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i) {
    Function *CMF = M->getFunction(Funcs[i]->getName(), 
                                   Funcs[i]->getFunctionType());
    assert(CMF && "Function not in module?!");
    Functions.insert(CMF);
  }

  std::cout << "Checking for crash with only these functions:";
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i)
    std::cout << " " << Funcs[i]->getName();
  std::cout << ": ";

  // Loop over and delete any functions which we aren't supposed to be playing
  // with...
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isExternal() && !Functions.count(I))
      DeleteFunctionBody(I);

  // Try running the hacked up program...
  std::swap(BD.Program, M);
  if (BD.runPasses(BD.PassesToRun)) {
    delete M;         // It crashed, keep the trimmed version...

    // Make sure to use function pointers that point into the now-current
    // module.
    Funcs.assign(Functions.begin(), Functions.end());
    return true;
  }
  delete BD.Program;  // It didn't crash, revert...
  BD.Program = M;
  return false;
}


/// debugCrash - This method is called when some pass crashes on input.  It
/// attempts to prune down the testcase to something reasonable, and figure
/// out exactly which pass is crashing.
///
bool BugDriver::debugCrash() {
  bool AnyReduction = false;
  std::cout << "\n*** Debugging optimizer crash!\n";

  // Reduce the list of passes which causes the optimizer to crash...
  unsigned OldSize = PassesToRun.size();
  DebugCrashes(*this).reduceList(PassesToRun);

  std::cout << "\n*** Found crashing pass"
            << (PassesToRun.size() == 1 ? ": " : "es: ")
            << getPassesString(PassesToRun) << "\n";

  EmitProgressBytecode("passinput");
  
  // Now try to reduce the number of functions in the module to something small.
  std::vector<Function*> Functions;
  for (Module::iterator I = Program->begin(), E = Program->end(); I != E; ++I)
    if (!I->isExternal())
      Functions.push_back(I);

  if (Functions.size() > 1) {
    std::cout << "\n*** Attempting to reduce the number of functions "
      "in the testcase\n";

    OldSize = Functions.size();
    ReduceCrashingFunctions(*this).reduceList(Functions);

    if (Functions.size() < OldSize) {
      EmitProgressBytecode("reduced-function");
      AnyReduction = true;
    }
  }

  // FIXME: This should attempt to delete entire basic blocks at a time to speed
  // up convergence...

  // FIXME: This should use the list reducer to converge faster by deleting
  // larger chunks of instructions at a time!
  unsigned Simplification = 4;
  do {
    --Simplification;
    std::cout << "\n*** Attempting to reduce testcase by deleting instruc"
              << "tions: Simplification Level #" << Simplification << "\n";

    // Now that we have deleted the functions that are unneccesary for the
    // program, try to remove instructions that are not neccesary to cause the
    // crash.  To do this, we loop through all of the instructions in the
    // remaining functions, deleting them (replacing any values produced with
    // nulls), and then running ADCE and SimplifyCFG.  If the transformed input
    // still triggers failure, keep deleting until we cannot trigger failure
    // anymore.
    //
  TryAgain:
    
    // Loop over all of the (non-terminator) instructions remaining in the
    // function, attempting to delete them.
    for (Module::iterator FI = Program->begin(), E = Program->end();
         FI != E; ++FI)
      if (!FI->isExternal()) {
        for (Function::iterator BI = FI->begin(), E = FI->end(); BI != E; ++BI)
          for (BasicBlock::iterator I = BI->begin(), E = --BI->end();
               I != E; ++I) {
            Module *M = deleteInstructionFromProgram(I, Simplification);
            
            // Make the function the current program...
            std::swap(Program, M);
            
            // Find out if the pass still crashes on this pass...
            std::cout << "Checking instruction '" << I->getName() << "': ";
            if (runPasses(PassesToRun)) {
              // Yup, it does, we delete the old module, and continue trying to
              // reduce the testcase...
              delete M;
              AnyReduction = true;
              goto TryAgain;  // I wish I had a multi-level break here!
            }
            
            // This pass didn't crash without this instruction, try the next
            // one.
            delete Program;
            Program = M;
          }
      }
  } while (Simplification);

  // Try to clean up the testcase by running funcresolve and globaldce...
  if (AnyReduction) {
    std::cout << "\n*** Attempting to perform final cleanups: ";
    Module *M = performFinalCleanups();
    std::swap(Program, M);
            
    // Find out if the pass still crashes on the cleaned up program...
    if (runPasses(PassesToRun)) {
      // Yup, it does, keep the reduced version...
      delete M;
      AnyReduction = true;
    } else {
      delete Program;   // Otherwise, restore the original module...
      Program = M;
    }
  }

  if (AnyReduction)
    EmitProgressBytecode("reduced-simplified");

  return false;
}
