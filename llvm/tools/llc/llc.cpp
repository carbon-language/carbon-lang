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
#include "llvm/Transforms/HoistPHIConstants.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Transforms/ConstantMerge.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "Support/CommandLine.h"
#include <memory>
#include <string>
#include <fstream>
using std::string;

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

class GenerateCodeForTarget : public MethodPass {
  TargetMachine &Target;
public:
  inline GenerateCodeForTarget(TargetMachine &T) : Target(T) {}

  // runOnMethod - This method does the actual work of generating code for
  // the specified method.
  //
  bool runOnMethod(Method *M) {
    if (!M->isExternal() && Target.compileMethod(M)) {
      cerr << "Error compiling " << InputFilename << "!\n";
    }
    
    return true;
  }
};


//===---------------------------------------------------------------------===//
// EmitAssembly Pass
// 
// Write assembly code to specified output stream
//===---------------------------------------------------------------------===//

class EmitAssembly : public Pass {
  const TargetMachine &Target;   // Target to compile for
  std::ostream *Out;             // Stream to print on
  bool DeleteStream;             // Delete stream in dtor?
public:
  inline EmitAssembly(const TargetMachine &T, std::ostream *O, bool D)
    : Target(T), Out(O), DeleteStream(D) {}

  virtual bool run(Module *M) {
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

  // Hoist constants out of PHI nodes into predecessor BB's
  Passes.add(new HoistPHIConstants());

  if (TraceBBValues || TraceMethodValues) {   // If tracing enabled...
    // Insert trace code in all methods in the module
    Passes.add(new InsertTraceCode(TraceBBValues, 
                                   TraceBBValues ||TraceMethodValues));

    // Eliminate duplication in constant pool
    Passes.add(new DynamicConstantMerge());
      
    // Then write out the module with tracing code before code generation 
    assert(InputFilename != "-" &&
           "files on stdin not supported with tracing");
    string traceFileName = GetFileNameRoot(InputFilename) + ".trace.bc";

    if (!Force && !std::ifstream(OutputFilename.c_str())) {
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
  Passes.add(new LowerAllocations(Target.DataLayout));
  
  // If LLVM dumping after transformations is requested, add it to the pipeline
  if (DumpAsm)
    Passes.add(new PrintMethodPass("Code after xformations: \n",&cerr));

  // Generate Target code...
  Passes.add(new GenerateCodeForTarget(Target));

  if (!DoNotEmitAssembly) {                // If asm output is enabled...
    // Figure out where we are going to send the output...
    std::ostream *Out = 0;
    if (OutputFilename != "") {   // Specified an output filename?
      if (!Force && !std::ifstream(OutputFilename.c_str())) {
        // If force is not specified, make sure not to overwrite a file!
        cerr << "Error opening '" << OutputFilename << "': File exists!\n"
             << "Use -f command line argument to force output\n";
        return 1;
      }
      Out = new std::ofstream(OutputFilename.c_str());
    } else {
      if (InputFilename == "-") {
        OutputFilename = "-";
        Out = &std::cout;
      } else {
        string OutputFilename = GetFileNameRoot(InputFilename); 
        OutputFilename += ".s";

        if (!Force && !std::ifstream(OutputFilename.c_str())) {
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
      }
    }
    
    // Output assembly language to the .s file
    Passes.add(new EmitAssembly(Target, Out, Out != &std::cout));
  }
  
  // Run our queue of passes all at once now, efficiently.
  Passes.run(M.get());

  return 0;
}


