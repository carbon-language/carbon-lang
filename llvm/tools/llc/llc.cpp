//===-- llc.cpp - Implement the LLVM Compiler -----------------------------===//
//
// This is the llc compiler driver.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Reader.h"
#include "llvm/Target/Sparc.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation/TraceValues.h"
#include "llvm/Transforms/LowerAllocations.h"
#include "llvm/Transforms/HoistPHIConstants.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Transforms/ConstantMerge.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include <memory>
#include <string>
#include <fstream>

cl::String InputFilename ("", "Input filename", cl::NoFlags, "-");
cl::String OutputFilename("o", "Output filename", cl::NoFlags, "");
cl::Flag   Force         ("f", "Overwrite output files");
cl::Flag   DumpAsm       ("d", "Print bytecode before native code generation",
                          cl::Hidden);
cl::Flag   DoNotEmitAssembly("noasm", "Do not emit assembly code", cl::Hidden);
cl::Flag   TraceBBValues ("trace",
                          "Trace values at basic block and method exits");
cl::Flag   TraceMethodValues("tracem", "Trace values only at method exits");


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
// GenerateCodeForTarget Pass
// 
// Native code generation for a specified target.
//===---------------------------------------------------------------------===//

class GenerateCodeForTarget : public Pass {
  TargetMachine &Target;
public:
  inline GenerateCodeForTarget(TargetMachine &T) : Target(T) {}

  // doPerMethodWork - This method does the actual work of generating code for
  // the specified method.
  //
  bool doPerMethodWork(Method *M) {
    if (!M->isExternal() && Target.compileMethod(M)) {
      cerr << "Error compiling " << InputFilename << "!\n";
      return true;
    }
    
    return false;
  }
};


//===---------------------------------------------------------------------===//
// EmitAssembly Pass
// 
// Write assembly code to specified output stream
//===---------------------------------------------------------------------===//

class EmitAssembly : public Pass {
  const TargetMachine &Target;   // Target to compile for
  ostream *Out;                  // Stream to print on
  bool DeleteStream;             // Delete stream in dtor?
public:
  inline EmitAssembly(const TargetMachine &T, ostream *O, bool D)
    : Target(T), Out(O), DeleteStream(D) {}


  virtual bool doPassFinalization(Module *M) {
    Target.emitAssembly(M, *Out);

    if (DeleteStream) delete Out;
    return false;
  }
};


//===---------------------------------------------------------------------===//
// Function main()
// 
// Entry point for the llc compiler.
//===---------------------------------------------------------------------===//

int main(int argc, char **argv) {
  int retCode = 0;
  cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");
  
  // Allocate a target... in the future this will be controllable on the
  // command line.
  auto_ptr<TargetMachine> target(allocateSparcTargetMachine());
  assert(target.get() && "Could not allocate target machine!");

  TargetMachine &Target = *target.get();
  
  // Load the module to be compiled...
  auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  // Build up all of the passes that we want to do to the module...
  vector<Pass*> Passes;

  // Hoist constants out of PHI nodes into predecessor BB's
  Passes.push_back(new HoistPHIConstants());

  if (TraceBBValues || TraceMethodValues) {   // If tracing enabled...
    // Insert trace code in all methods in the module
    Passes.push_back(new InsertTraceCode(TraceBBValues, 
                                         TraceBBValues ||TraceMethodValues));

    // Eliminate duplication in constant pool
    Passes.push_back(new DynamicConstantMerge());
      
    // Then write out the module with tracing code before code generation 
    assert(InputFilename != "-" &&
           "files on stdin not supported with tracing");
    string traceFileName = GetFileNameRoot(InputFilename) + ".trace.bc";
    ostream *os = new ofstream(traceFileName.c_str(), 
                               (Force ? 0 : ios::noreplace)|ios::out);
    if (!os->good()) {
      cerr << "Error opening " << traceFileName
           << "! SKIPPING OUTPUT OF TRACE CODE\n";
      delete os;
      return 1;
    }
    
    Passes.push_back(new WriteBytecodePass(os, true));
  }
  
  // Replace malloc and free instructions with library calls.
  // Do this after tracing until lli implements these lib calls.
  // For now, it will emulate malloc and free internally.
  Passes.push_back(new LowerAllocations(Target.DataLayout));
  
  // If LLVM dumping after transformations is requested, add it to the pipeline
  if (DumpAsm)
    Passes.push_back(new PrintModulePass("Code after xformations: \n",&cerr));

  // Generate Target code...
  Passes.push_back(new GenerateCodeForTarget(Target));

  if (!DoNotEmitAssembly) {                // If asm output is enabled...
    // Figure out where we are going to send the output...
    ostream *Out = 0;
    if (OutputFilename != "") {   // Specified an output filename?
      Out = new ofstream(OutputFilename.c_str(), 
                         (Force ? 0 : ios::noreplace)|ios::out);
    } else {
      if (InputFilename == "-") {
        OutputFilename = "-";
        Out = &cout;
      } else {
        string OutputFilename = GetFileNameRoot(InputFilename); 
        OutputFilename += ".s";
        Out = new ofstream(OutputFilename.c_str(), 
                           (Force ? 0 : ios::noreplace)|ios::out);
        if (!Out->good()) {
          cerr << "Error opening " << OutputFilename << "!\n";
          delete Out;
          return 1;
        }
      }
    }
    
    // Output assembly language to the .s file
    Passes.push_back(new EmitAssembly(Target, Out, Out != &cout));
  }
  
  // Run our queue of passes all at once now, efficiently.  This form of
  // runAllPasses frees the Pass objects after runAllPasses completes.
  Pass::runAllPassesAndFree(M.get(), Passes);

  return retCode;
}


