//===-- llc.cpp - Implement the LLVM Compiler ----------------------------===//
//
// This is the llc compiler driver.
//
//===---------------------------------------------------------------------===//

#include "llvm/Bytecode/Reader.h"
#include "llvm/Optimizations/Normalize.h"
#include "llvm/Target/Sparc.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include <memory>
#include <fstream>

cl::String InputFilename ("", "Input filename", cl::NoFlags, "-");
cl::String OutputFilename("o", "Output filename", cl::NoFlags, "");
cl::Flag   Force         ("f", "Overwrite output files", cl::NoFlags, false);


//-------------------------- Internal Functions -----------------------------//

static void NormalizeMethod(Method *M) {
  NormalizePhiConstantArgs(M);
}


//===---------------------------------------------------------------------===//
// Function main()
// 
// Entry point for the llc compiler.
//===---------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Parse command line options...
  cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");

  // Allocate a target... in the future this will be controllable on the
  // command line.
  auto_ptr<TargetMachine> Target(allocateSparcTargetMachine());

  // Load the module to be compiled...
  auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  // Loop over all of the methods in the module, compiling them.
  for (Module::const_iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI) {
    Method *Meth = *MI;
    
    NormalizeMethod(Meth);
    
    if (Target.get()->compileMethod(Meth)) {
      cerr << "Error compiling " << InputFilename << "!\n";
      return 1;
    }
  }
  
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
      string IFN = InputFilename;
      int Len = IFN.length();
      if (IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c') {
        OutputFilename = string(IFN.begin(), IFN.end()-3); // s/.bc/.s/
      } else {
        OutputFilename = IFN;   // Append a .s to it
      }
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

  // Emit the output...
  Target->emitAssembly(M.get(), *Out);

  if (Out != &cout) delete Out;
  return 0;
}


