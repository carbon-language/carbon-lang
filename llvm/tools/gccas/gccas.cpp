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
#include "llvm/Transforms/RaisePointerReferences.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/LoadValueNumbering.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Target/TargetData.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <memory>
#include <fstream>

namespace {
  cl::opt<std::string>
  InputFilename(cl::Positional,cl::desc("<input llvm assembly>"),cl::init("-"));

  cl::opt<std::string> 
  OutputFilename("o", cl::desc("Override output filename"),
                 cl::value_desc("filename"));

  cl::opt<int>
  RunNPasses("stopAfterNPasses",
             cl::desc("Only run the first N passes of gccas"), cl::Hidden,
             cl::value_desc("# passes"));

  cl::opt<bool>   
  Verify("verify", cl::desc("Verify each pass result"));
}


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
  } else {
    delete P;             // We don't want this pass to run, just delete it now
  }
}


void AddConfiguredTransformationPasses(PassManager &PM) {
  if (Verify) PM.add(createVerifierPass());

  addPass(PM, createFunctionResolvingPass());    // Resolve (...) functions
  addPass(PM, createGlobalDCEPass());            // Kill unused uinit g-vars
  addPass(PM, createDeadTypeEliminationPass());  // Eliminate dead types
  addPass(PM, createConstantMergePass());        // Merge dup global constants
  addPass(PM, createCFGSimplificationPass());    // Merge & remove BBs
  addPass(PM, createVerifierPass());             // Verify that input is correct
  addPass(PM, createDeadInstEliminationPass());  // Remove Dead code/vars
  addPass(PM, createRaiseAllocationsPass());     // call %malloc -> malloc inst
  addPass(PM, createInstructionCombiningPass()); // Cleanup code for raise
  addPass(PM, createIndVarSimplifyPass());       // Simplify indvars
  addPass(PM, createRaisePointerReferencesPass());// Recover type information
  addPass(PM, createInstructionCombiningPass()); // Combine silly seq's
  addPass(PM, createPromoteMemoryToRegister());  // Promote alloca's to regs
  addPass(PM, createReassociatePass());          // Reassociate expressions
  //addPass(PM, createCorrelatedExpressionEliminationPass());// Kill corr branches
  addPass(PM, createInstructionCombiningPass()); // Combine silly seq's
  addPass(PM, createCFGSimplificationPass());    // Merge & remove BBs
  addPass(PM, createLICMPass());                 // Hoist loop invariants
  addPass(PM, createLoadValueNumberingPass());   // GVN for load instructions
  addPass(PM, createGCSEPass());                 // Remove common subexprs
  addPass(PM, createSCCPPass());                 // Constant prop with SCCP

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  addPass(PM, createInstructionCombiningPass());
  addPass(PM, createAggressiveDCEPass());        // SSA based 'Agressive DCE'
  addPass(PM, createCFGSimplificationPass());    // Merge & remove BBs
}


int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm .s -> .o assembler for GCC\n");

  std::auto_ptr<Module> M;
  try {
    // Parse the file now...
    M.reset(ParseAssemblyFile(InputFilename));
  } catch (const ParseException &E) {
    std::cerr << argv[0] << ": " << E.getMessage() << "\n";
    return 1;
  }

  if (M.get() == 0) {
    std::cerr << argv[0] << ": assembly didn't read correctly.\n";
    return 1;
  }

  std::ostream *Out = 0;
  if (OutputFilename == "") {   // Didn't specify an output filename?
    if (InputFilename == "-") {
      OutputFilename = "-";
    } else {
      std::string IFN = InputFilename;
      int Len = IFN.length();
      if (IFN[Len-2] == '.' && IFN[Len-1] == 's') {   // Source ends in .s?
        OutputFilename = std::string(IFN.begin(), IFN.end()-2);
      } else {
        OutputFilename = IFN;   // Append a .o to it
      }
      OutputFilename += ".o";
    }
  }

  if (OutputFilename == "-")
    Out = &std::cout;
  else {
    Out = new std::ofstream(OutputFilename.c_str(), std::ios::out);

    // Make sure that the Out file gets unlink'd from the disk if we get a
    // signal
    RemoveFileOnSignal(OutputFilename);
  }

  
  if (!Out->good()) {
    std::cerr << argv[0] << ": error opening " << OutputFilename << "!\n";
    return 1;
  }

  // In addition to just parsing the input from GCC, we also want to spiff it up
  // a little bit.  Do this now.
  //
  PassManager Passes;

  // Add an appropriate TargetData instance for this module...
  Passes.add(new TargetData("gccas", M.get()));

  // Add all of the transformation passes to the pass manager to do the cleanup
  // and optimization of the GCC output.
  //
  AddConfiguredTransformationPasses(Passes);

  // Write bytecode to file...
  Passes.add(new WriteBytecodePass(Out));

  // Run our queue of passes all at once now, efficiently.
  Passes.run(*M.get());

  if (Out != &std::cout) delete Out;
  return 0;
}
