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

#include "llvm/Bytecode/Analyzer.h"
#include "Support/CommandLine.h"
#include "llvm/System/Signals.h"
#include <fstream>
#include <iostream>

using namespace llvm;

static cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input bytecode>"), cl::init("-"));

static cl::opt<std::string> 
  OutputFilename("o", cl::desc("Override output filename"), 
    cl::value_desc("filename"));

static cl::opt<bool> Force      ("f", cl::desc("Overwrite output files"));
static cl::opt<bool> Detailed   ("d", cl::desc("Detailed output"));

int 
main(int argc, char **argv) 
{
  cl::ParseCommandLineOptions(argc, argv, 
    " llvm-abcd Analysis of ByteCode Dumper\n");

  PrintStackTraceOnErrorSignal();

  std::ostream* Out = &std::cout;  // Default to printing to stdout...
  std::istream* In  = &std::cin;   // Default to reading stdin
  std::string ErrorMessage;
  BytecodeAnalysis bca;

  /// Analyze the bytecode file
  AnalyzeBytecodeFile(InputFilename, bca, &ErrorMessage);

  // If there was an error, print it and stop.
  if ( ErrorMessage.size() ) {
    std::cerr << argv[0] << ": " << ErrorMessage << "\n";
    return 1;
  }
  
  // Figure out where the output is going
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
	OutputFilename = std::string(IFN.begin(), IFN.end()-3)+".abc";
      } else {
	OutputFilename = IFN+".abc";
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

  // All that abcd does is write the gathered statistics to the output
  bca.dumpBytecode = true;
  PrintBytecodeAnalysis(bca,*Out);

  if (Out != &std::cout) {
    ((std::ofstream*)Out)->close();
    delete Out;
  }
  return 0;
}

// vim: sw=2
