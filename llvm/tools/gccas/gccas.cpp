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
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <memory>
#include <fstream>

cl::String InputFilename ("", "Parse <arg> file, compile to bytecode",
                          cl::Required, "");
cl::String OutputFilename("o", "Override output filename", cl::NoFlags, "");
cl::Flag   StopAtLevelRaise("stopraise", "Stop optimization before level raise",
                            cl::Hidden);

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm .s -> .o assembler for GCC\n");

  std::auto_ptr<Module> M;
  try {
    // Parse the file now...
    M.reset(ParseAssemblyFile(InputFilename));
  } catch (const ParseException &E) {
    cerr << E.getMessage() << endl;
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

  std::ofstream Out(OutputFilename.c_str(), ios::out);
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
  Passes.add(createFunctionResolvingPass());      // Resolve (...) functions
  Passes.add(createConstantMergePass());          // Merge dup global constants
  Passes.add(createDeadInstEliminationPass());    // Remove Dead code/vars
  Passes.add(createRaiseAllocationsPass());       // call %malloc -> malloc inst
  Passes.add(createCleanupGCCOutputPass());       // Fix gccisms
  Passes.add(createIndVarSimplifyPass());         // Simplify indvars
  if (!StopAtLevelRaise) {
    Passes.add(createRaisePointerReferencesPass()); // Eliminate casts
    Passes.add(createPromoteMemoryToRegister());    // Promote alloca's to regs
    Passes.add(createReassociatePass());            // Reassociate expressions
    Passes.add(createInstructionCombiningPass());   // Combine silly seq's
    Passes.add(createDeadInstEliminationPass());    // Kill InstCombine remnants
    Passes.add(createLICMPass());                   // Hoist loop invariants
    Passes.add(createGCSEPass());                   // Remove common subexprs
    Passes.add(createSCCPPass());                   // Constant prop with SCCP

    // Run instcombine after redundancy elimination to exploit opportunities
    // opened up by them.
    Passes.add(createInstructionCombiningPass());
    Passes.add(createAggressiveDCEPass());          // SSA based 'Agressive DCE'
    Passes.add(createCFGSimplificationPass());      // Merge & remove BBs
  }
  Passes.add(new WriteBytecodePass(&Out));        // Write bytecode to file...

  // Run our queue of passes all at once now, efficiently.
  Passes.run(M.get());
  return 0;
}
