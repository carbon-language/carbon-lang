//===-- llvm-abcd.cpp - Analysis of Byte Code Dumper ----------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This tool may be invoked in the following manner:
//  llvm-abcd [options]      - Read LLVM bytecode from stdin
//  llvm-abcd [options] x.bc - Read LLVM bytecode from the x.bc file
//
//  Options:
//      --help      - Output information about command line switches
//      --nodetails - Don't print out detailed informaton about individual 
//                    blocks and functions
//      --dump      - Dump low-level bytecode structure in readable format
//
// This tool provides analytical information about a bytecode file. It is
// intended as an aid to developers of bytecode reading and writing software. It
// produces on std::out a summary of the bytecode file that shows various 
// statistics about the contents of the file. By default this information is
// detailed and contains information about individual bytecode blocks and the
// functions in the module. To avoid this more detailed output, use the 
// -nodetails option to limit the output to just module level information.
// The tool is also able to print a bytecode file in a straight forward text 
// format that shows the containment and relationships of the information in 
// the bytecode file (-dump option). 
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Analyzer.h"
#include "Support/CommandLine.h"
#include "llvm/System/Signals.h"
#include <fstream>
#include <iostream>

using namespace llvm;

static cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input bytecode>"), cl::init("-"));

static cl::opt<bool> NoDetails ("nodetails", cl::desc("Skip detailed output"));
static cl::opt<bool> Dump      ("dump", cl::desc("Detailed output"));

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

  /// Determine what to generate
  bca.dumpBytecode = Dump;
  bca.detailedResults = !NoDetails;

  /// Analyze the bytecode file
  AnalyzeBytecodeFile(InputFilename, bca, &ErrorMessage);

  // If there was an error, print it and stop.
  if ( ErrorMessage.size() ) {
    std::cerr << argv[0] << ": " << ErrorMessage << "\n";
    return 1;
  }
  
  // All that abcd does is write the gathered statistics to the output
  PrintBytecodeAnalysis(bca,*Out);

  if (Out != &std::cout) {
    ((std::ofstream*)Out)->close();
    delete Out;
  }
  return 0;
}

// vim: sw=2
