//===----------------------------------------------------------------------===//
// LLVM 'GCCAS' UTILITY 
//
//  This utility is designed to be used by the GCC frontend for creating
// bytecode files from it's intermediate llvm assembly.  The requirements for
// this utility are thus slightly different than that of the standard as util.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Transforms/CleanupGCCOutput.h"
#include "llvm/Transforms/LevelChange.h"
#include "llvm/Transforms/ConstantMerge.h"
#include "llvm/Transforms/ChangeAllocations.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <memory>
#include <fstream>
using std::cerr;

static cl::String InputFilename   ("", "Parse <arg> file, compile to bytecode",
                                   cl::Required, "");
static cl::String OutputFilename  ("o", "Override output filename");
static cl::Int    RunNPasses      ("stopAfterNPasses", "Only run the first N "
                                   "passes of gccas", cl::Hidden);
static cl::Flag   StopAtLevelRaise("stopraise", "Stop optimization before "
                                   "level raise", cl::Hidden);
static cl::Flag   Verify          ("verify", "Verify each pass result");

static inline void addPass(PassManager &PM, Pass *P) {
  static int NumPassesCreated = 0;
  
  // If we haven't already created the number of passes that was requested...
  if (RunNPasses == 0 || RunNPasses > NumPassesCreated) {
    // Add the pass to the pass manager...
    PM.add(P);

    // If we are verifying all of the intermediate steps, add the verifier...
    if (Verify) PM.add(createVerifierPass());

    // Keep track of how many passes we made for -stopAfterNPasses
    ++NumPassesCreated;
  }
}


void AddConfiguredTransformationPasses(PassManager &PM) {
  if (Verify) PM.add(createVerifierPass());

  addPass(PM, createFunctionResolvingPass());     // Resolve (...) functions
  addPass(PM, createConstantMergePass());         // Merge dup global constants
  addPass(PM, createDeadInstEliminationPass());   // Remove Dead code/vars
  addPass(PM, createRaiseAllocationsPass());      // call %malloc -> malloc inst
  addPass(PM, createCleanupGCCOutputPass());      // Fix gccisms
  addPass(PM, createIndVarSimplifyPass());        // Simplify indvars

  // Level raise is eternally buggy/in need of enhancements.  Allow
  // transformation to stop right before it runs.
  if (StopAtLevelRaise) return;

  addPass(PM, createRaisePointerReferencesPass());// Eliminate casts
  addPass(PM, createPromoteMemoryToRegister());   // Promote alloca's to regs
  addPass(PM, createReassociatePass());           // Reassociate expressions
  addPass(PM, createInstructionCombiningPass());  // Combine silly seq's
  addPass(PM, createDeadInstEliminationPass());   // Kill InstCombine remnants
  addPass(PM, createLICMPass());                  // Hoist loop invariants
  addPass(PM, createGCSEPass());                  // Remove common subexprs
  addPass(PM, createSCCPPass());                  // Constant prop with SCCP

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  addPass(PM, createInstructionCombiningPass());
  addPass(PM, createAggressiveDCEPass());          // SSA based 'Agressive DCE'
  addPass(PM, createCFGSimplificationPass());      // Merge & remove BBs
}


int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm .s -> .o assembler for GCC\n");

  std::auto_ptr<Module> M;
  try {
    // Parse the file now...
    M.reset(ParseAssemblyFile(InputFilename));
  } catch (const ParseException &E) {
    cerr << E.getMessage() << "\n";
    return 1;
  }

  if (M.get() == 0) {
    cerr << "assembly didn't read correctly.\n";
    return 1;
  }
  
  if (OutputFilename == "") {   // Didn't specify an output filename?
    std::string IFN = InputFilename;
    int Len = IFN.length();
    if (IFN[Len-2] == '.' && IFN[Len-1] == 's') {   // Source ends in .s?
      OutputFilename = std::string(IFN.begin(), IFN.end()-2);
    } else {
      OutputFilename = IFN;   // Append a .o to it
    }
    OutputFilename += ".o";
  }

  std::ofstream Out(OutputFilename.c_str(), std::ios::out);
  if (!Out.good()) {
    cerr << "Error opening " << OutputFilename << "!\n";
    return 1;
  }

  // Make sure that the Out file gets unlink'd from the disk if we get a SIGINT
  RemoveFileOnSignal(OutputFilename);

  // In addition to just parsing the input from GCC, we also want to spiff it up
  // a little bit.  Do this now.
  //
  PassManager Passes;

  // Add all of the transformation passes to the pass manager to do the cleanup
  // and optimization of the GCC output.
  //
  AddConfiguredTransformationPasses(Passes);

  // Write bytecode to file...
  Passes.add(new WriteBytecodePass(&Out));

  // Run our queue of passes all at once now, efficiently.
  Passes.run(*M.get());
  return 0;
}
