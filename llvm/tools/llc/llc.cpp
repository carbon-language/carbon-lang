//===-- llc.cpp - Implement the LLVM Compiler -----------------------------===//
//
// This is the llc compiler driver.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Reader.h"
#include "llvm/Target/Sparc.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation/TraceValues.h"
#include "llvm/Transforms/ChangeAllocations.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Transforms/ConstantMerge.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/PassManager.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <memory>
#include <fstream>
using std::string;

static cl::String InputFilename ("", "Input filename", cl::NoFlags, "-");
static cl::String OutputFilename("o", "Output filename", cl::NoFlags, "");
static cl::Flag   Force         ("f", "Overwrite output files");
static cl::Flag   DumpAsm       ("d", "Print bytecode before native code generation", cl::Hidden);

enum TraceLevel {
  TraceOff, TraceFunctions, TraceBasicBlocks
};

static cl::Enum<enum TraceLevel> TraceValues("trace", cl::NoFlags,
  "Trace values through functions or basic blocks",
  clEnumValN(TraceOff        , "off",        "Disable trace code"),
  clEnumValN(TraceFunctions  , "function",   "Trace each function"),
  clEnumValN(TraceBasicBlocks, "basicblock", "Trace each basic block"), 0);


// GetFileNameRoot - Helper function to get the basename of a filename...
static inline string GetFileNameRoot(const string &InputFilename) {
  string IFN = InputFilename;
  string outputFilename;
  int Len = IFN.length();
  if (IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c') {
    outputFilename = string(IFN.begin(), IFN.end()-3); // s/.bc/.s/
  } else {
    outputFilename = IFN;
  }
  return outputFilename;
}


//===---------------------------------------------------------------------===//
// Function main()
// 
// Entry point for the llc compiler.
//===---------------------------------------------------------------------===//

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");
  
  // Allocate a target... in the future this will be controllable on the
  // command line.
  std::auto_ptr<TargetMachine> target(allocateSparcTargetMachine());
  assert(target.get() && "Could not allocate target machine!");

  TargetMachine &Target = *target.get();
  
  // Load the module to be compiled...
  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  // Build up all of the passes that we want to do to the module...
  PassManager Passes;

  if (TraceValues != TraceOff) {   // If tracing enabled...
    // Insert trace code in all functions in the module
    if (TraceValues == TraceBasicBlocks)
      Passes.add(createTraceValuesPassForBasicBlocks());
    else if (TraceValues == TraceFunctions)
      Passes.add(createTraceValuesPassForFunction());
    else
      assert(0 && "Bad value for TraceValues!");

    // Eliminate duplication in constant pool
    Passes.add(createDynamicConstantMergePass());
  }
  
  // Decompose multi-dimensional refs into a sequence of 1D refs
  Passes.add(createDecomposeMultiDimRefsPass());
  
  // Write out the module with tracing code just before code generation
  if (TraceValues != TraceOff) {   // If tracing enabled...
    assert(InputFilename != "-" &&
           "files on stdin not supported with tracing");
    string traceFileName = GetFileNameRoot(InputFilename) + ".trace.bc";

    if (!Force && std::ifstream(OutputFilename.c_str())) {
      // If force is not specified, make sure not to overwrite a file!
      cerr << "Error opening '" << OutputFilename << "': File exists!\n"
           << "Use -f command line argument to force output\n";
      return 1;
    }
    
    std::ostream *os = new std::ofstream(traceFileName.c_str());
    if (!os->good()) {
      cerr << "Error opening " << traceFileName
           << "! SKIPPING OUTPUT OF TRACE CODE\n";
      delete os;
      return 1;
    }
    
    Passes.add(new WriteBytecodePass(os, true));
  }
  
  // Replace malloc and free instructions with library calls.
  // Do this after tracing until lli implements these lib calls.
  // For now, it will emulate malloc and free internally.
  Passes.add(createLowerAllocationsPass(Target.DataLayout));
  
  // If LLVM dumping after transformations is requested, add it to the pipeline
  if (DumpAsm)
    Passes.add(new PrintFunctionPass("Code after xformations: \n", &cerr));

  // Figure out where we are going to send the output...
  std::ostream *Out = 0;
  if (OutputFilename != "") {   // Specified an output filename?
    if (!Force && std::ifstream(OutputFilename.c_str())) {
      // If force is not specified, make sure not to overwrite a file!
      cerr << "Error opening '" << OutputFilename << "': File exists!\n"
           << "Use -f command line argument to force output\n";
      return 1;
    }
    Out = new std::ofstream(OutputFilename.c_str());

    // Make sure that the Out file gets unlink'd from the disk if we get a
    // SIGINT
    RemoveFileOnSignal(OutputFilename);
  } else {
    if (InputFilename == "-") {
      OutputFilename = "-";
      Out = &std::cout;
    } else {
      string OutputFilename = GetFileNameRoot(InputFilename); 
      OutputFilename += ".s";

      if (!Force && std::ifstream(OutputFilename.c_str())) {
        // If force is not specified, make sure not to overwrite a file!
        cerr << "Error opening '" << OutputFilename << "': File exists!\n"
             << "Use -f command line argument to force output\n";
        return 1;
      }

      Out = new std::ofstream(OutputFilename.c_str());
      if (!Out->good()) {
        cerr << "Error opening " << OutputFilename << "!\n";
        delete Out;
        return 1;
      }
      // Make sure that the Out file gets unlink'd from the disk if we get a
      // SIGINT
      RemoveFileOnSignal(OutputFilename);
    }
  }
  
  Target.addPassesToEmitAssembly(Passes, *Out);
  
  // Run our queue of passes all at once now, efficiently.
  Passes.run(M.get());

  if (Out != &std::cout) delete Out;

  return 0;
}
