//===----------------------------------------------------------------------===//
// LLVM 'DIS' UTILITY 
//
// This utility may be invoked in the following manner:
//  dis [options]      - Read LLVM bytecode from stdin, write assembly to stdout
//  dis [options] x.bc - Read LLVM bytecode from the x.bc file, write assembly
//                       to the x.ll file.
//  Options:
//      --help   - Output information about command line switches
//       -c      - Print C code instead of LLVM assembly
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Support/CFG.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include "llvm/Assembly/CWriter.h"
#include <fstream>
#include <iostream>
using std::cerr;

// OutputMode - The different orderings to print basic blocks in...
enum OutputMode {
  llvm = 0,           // Generate LLVM assembly (the default)
  c,                  // Generate C code
};

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bytecode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<enum OutputMode>
WriteMode(cl::desc("Specify the output format:"),
          cl::values(
                     clEnumVal(llvm, "Output LLVM assembly"),
                     clEnumVal(c   , "Output C code for program"),
                    0));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm .bc -> .ll disassembler\n");
  std::ostream *Out = &std::cout;  // Default to printing to stdout...

  Module *M = ParseBytecodeFile(InputFilename);
  if (M == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }
  
  if (OutputFilename != "") {   // Specified an output filename?
    if (!Force && std::ifstream(OutputFilename.c_str())) {
      // If force is not specified, make sure not to overwrite a file!
      cerr << "Error opening '" << OutputFilename
           << "': File exists! Sending to standard output.\n";
    } else {
      Out = new std::ofstream(OutputFilename.c_str());
    }
  } else {
    if (InputFilename == "-") {
      OutputFilename = "-";
    } else {
      std::string IFN = InputFilename;
      int Len = IFN.length();
      if (IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c') {
	// Source ends in .bc
	OutputFilename = std::string(IFN.begin(), IFN.end()-3);
      } else {
	OutputFilename = IFN;   // Append a .ll to it
      }
      if (WriteMode == c)
        OutputFilename += ".c";
      else
        OutputFilename += ".ll";

      if (!Force && std::ifstream(OutputFilename.c_str())) {
        // If force is not specified, make sure not to overwrite a file!
        cerr << "Error opening '" << OutputFilename
             << "': File exists! Sending to standard output.\n";
      } else {
        Out = new std::ofstream(OutputFilename.c_str());

        // Make sure that the Out file gets unlink'd from the disk if we get a
        // SIGINT
        RemoveFileOnSignal(OutputFilename);
      }
    }
  }

  if (!Out->good()) {
    cerr << "Error opening " << OutputFilename
	 << ": sending to stdout instead!\n";
    Out = &std::cout;
  }

  // All that dis does is write the assembly or C out to a file...
  //
  switch (WriteMode) {
  case llvm:
    (*Out) << M;           // Output LLVM assembly
    break;
  case c:
    WriteToC(M, *Out);     // Convert LLVM to C
    break;
  }
  delete M;

  if (Out != &std::cout) delete Out;
  return 0;
}

