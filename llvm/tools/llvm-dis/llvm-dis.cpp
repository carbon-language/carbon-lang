//===-- llvm-dis.cpp - The low-level LLVM disassembler --------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  llvm-dis [options]      - Read LLVM bytecode from stdin, write asm to stdout
//  llvm-dis [options] x.bc - Read LLVM bytecode from the x.bc file, write asm
//                            to the x.ll file.
//  Options:
//      --help   - Output information about command line switches
//       -c      - Print C code instead of LLVM assembly
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <fstream>
#include <memory>

// OutputMode - The different orderings to print basic blocks in...
enum OutputMode {
  LLVM = 0,           // Generate LLVM assembly (the default)
  c,                  // Generate C code
};

using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bytecode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"), 
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<enum OutputMode>
WriteMode(cl::desc("Specify the output format:"),
          cl::values(clEnumValN(LLVM, "llvm", "Output LLVM assembly"),
                     clEnumVal(c, "Output C code for program"),
                    0));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm .bc -> .ll disassembler\n");
  std::ostream *Out = &std::cout;  // Default to printing to stdout...
  std::string ErrorMessage;

  if (WriteMode == c) {
    std::cerr << "ERROR: llvm-dis no longer contains the C backend.  Use 'llc -march=c' instead!\n";
    exit(1);
  }

  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename, &ErrorMessage));
  if (M.get() == 0) {
    std::cerr << argv[0] << ": ";
    if (ErrorMessage.size())
      std::cerr << ErrorMessage << "\n";
    else
      std::cerr << "bytecode didn't read correctly.\n";
    return 1;
  }
  
  if (OutputFilename != "") {   // Specified an output filename?
    if (OutputFilename != "-") { // Not stdout?
      if (!Force && std::ifstream(OutputFilename.c_str())) {
        // If force is not specified, make sure not to overwrite a file!
        std::cerr << argv[0] << ": error opening '" << OutputFilename
                  << "': file exists! Sending to standard output.\n";
      } else {
        Out = new std::ofstream(OutputFilename.c_str());
      }
    }
  } else {
    if (InputFilename == "-") {
      OutputFilename = "-";
    } else {
      std::string IFN = InputFilename;
      int Len = IFN.length();
      if (IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c') {
	// Source ends in .bc
	OutputFilename = std::string(IFN.begin(), IFN.end()-3)+".ll";
      } else {
	OutputFilename = IFN+".ll";
      }

      if (!Force && std::ifstream(OutputFilename.c_str())) {
        // If force is not specified, make sure not to overwrite a file!
        std::cerr << argv[0] << ": error opening '" << OutputFilename
                  << "': file exists! Sending to standard output.\n";
      } else {
        Out = new std::ofstream(OutputFilename.c_str());

        // Make sure that the Out file gets unlinked from the disk if we get a
        // SIGINT
        RemoveFileOnSignal(OutputFilename);
      }
    }
  }

  if (!Out->good()) {
    std::cerr << argv[0] << ": error opening " << OutputFilename
              << ": sending to stdout instead!\n";
    Out = &std::cout;
  }

  // All that dis does is write the assembly or C out to a file...
  //
  PassManager Passes;
  Passes.add(new PrintModulePass(Out));
  Passes.run(*M.get());

  if (Out != &std::cout) {
    ((std::ofstream*)Out)->close();
    delete Out;
  }
  return 0;
}

