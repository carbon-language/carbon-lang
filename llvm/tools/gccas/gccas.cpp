//===-- gccas.cpp - The "optimizing assembler" used by the GCC frontend ---===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This utility is designed to be used by the GCC frontend for creating bytecode
// files from its intermediate LLVM assembly.  The requirements for this utility
// are thus slightly different than that of the standard `as' util.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/LoadValueNumbering.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <memory>
#include <fstream>

using namespace llvm;

namespace {
  cl::opt<std::string>
  InputFilename(cl::Positional,cl::desc("<input llvm assembly>"),cl::init("-"));

  cl::opt<std::string> 
  OutputFilename("o", cl::desc("Override output filename"),
                 cl::value_desc("filename"));

  cl::opt<bool>   
  Verify("verify", cl::desc("Verify each pass result"));

  cl::opt<bool>
  DisableInline("disable-inlining", cl::desc("Do not run the inliner pass"));
}


static inline void addPass(PassManager &PM, Pass *P) {
  // Add the pass to the pass manager...
  PM.add(P);
  
  // If we are verifying all of the intermediate steps, add the verifier...
  if (Verify) PM.add(createVerifierPass());
}


void AddConfiguredTransformationPasses(PassManager &PM) {
  PM.add(createVerifierPass());                  // Verify that input is correct
  addPass(PM, createLowerSetJmpPass());          // Lower llvm.setjmp/.longjmp
  addPass(PM, createFunctionResolvingPass());    // Resolve (...) functions
  addPass(PM, createCFGSimplificationPass());    // Clean up disgusting code
  addPass(PM, createRaiseAllocationsPass());     // call %malloc -> malloc inst
  addPass(PM, createGlobalDCEPass());            // Remove unused globals
  addPass(PM, createIPConstantPropagationPass());// IP Constant Propagation
  addPass(PM, createDeadArgEliminationPass());   // Dead argument elimination

  addPass(PM, createPruneEHPass());              // Remove dead EH info

  if (!DisableInline)
    addPass(PM, createFunctionInliningPass());   // Inline small functions

  addPass(PM, createInstructionCombiningPass()); // Cleanup code for raise
  addPass(PM, createRaisePointerReferencesPass());// Recover type information
  addPass(PM, createTailDuplicationPass());      // Simplify cfg by copying code
  addPass(PM, createCFGSimplificationPass());    // Merge & remove BBs
  addPass(PM, createScalarReplAggregatesPass()); // Break up aggregate allocas
  addPass(PM, createInstructionCombiningPass()); // Combine silly seq's

  addPass(PM, createReassociatePass());          // Reassociate expressions
  addPass(PM, createInstructionCombiningPass()); // Combine silly seq's
  addPass(PM, createTailCallEliminationPass());  // Eliminate tail calls
  addPass(PM, createCFGSimplificationPass());    // Merge & remove BBs
  addPass(PM, createLICMPass());                 // Hoist loop invariants
  addPass(PM, createLoadValueNumberingPass());   // GVN for load instructions
  addPass(PM, createGCSEPass());                 // Remove common subexprs
  addPass(PM, createSCCPPass());                 // Constant prop with SCCP

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  addPass(PM, createInstructionCombiningPass());
  addPass(PM, createIndVarSimplifyPass());       // Canonicalize indvars
  addPass(PM, createAggressiveDCEPass());        // SSA based 'Aggressive DCE'
  addPass(PM, createCFGSimplificationPass());    // Merge & remove BBs
  addPass(PM, createDeadTypeEliminationPass());  // Eliminate dead types
  addPass(PM, createConstantMergePass());        // Merge dup global constants
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

    // Make sure that the Out file gets unlinked from the disk if we get a
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
