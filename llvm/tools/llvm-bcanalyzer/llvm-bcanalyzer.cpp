//===-- llvm-bcanalyzer.cpp - Byte Code Analyzer --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool may be invoked in the following manner:
//  llvm-bcanalyzer [options]      - Read LLVM bytecode from stdin
//  llvm-bcanalyzer [options] x.bc - Read LLVM bytecode from the x.bc file
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

#include "llvm/Analysis/Verifier.h"
#include "llvm/Bytecode/Analyzer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/System/Signals.h"
#include <fstream>
#include <iostream>

using namespace llvm;

static cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input bytecode>"), cl::init("-"));

static cl::opt<std::string>
  OutputFilename("-o", cl::init("-"), cl::desc("<output file>"));

static cl::opt<bool> NoDetails ("nodetails", cl::desc("Skip detailed output"));
static cl::opt<bool> Dump      ("dump", cl::desc("Dump low level bytecode trace"));
static cl::opt<bool> Verify    ("verify", cl::desc("Progressively verify module"));

int
main(int argc, char **argv) {
  try {
    cl::ParseCommandLineOptions(argc, argv,
      " llvm-bcanalyzer Analysis of ByteCode Dumper\n");

    sys::PrintStackTraceOnErrorSignal();

    std::ostream* Out = &std::cout;  // Default to printing to stdout...
    std::istream* In  = &std::cin;   // Default to reading stdin
    std::string ErrorMessage;
    BytecodeAnalysis bca;

    /// Determine what to generate
    bca.detailedResults = !NoDetails;
    bca.progressiveVerify = Verify;

    /// Analyze the bytecode file
    Module* M = AnalyzeBytecodeFile(InputFilename, bca, &ErrorMessage, (Dump?Out:0));

    // All that bcanalyzer does is write the gathered statistics to the output
    PrintBytecodeAnalysis(bca,*Out);

    if ( M && Verify ) {
      std::string verificationMsg;
      if (verifyModule(*M, ReturnStatusAction, &verificationMsg))
        std::cerr << "Final Verification Message: " << verificationMsg << "\n";
    }

    if (Out != &std::cout) {
      ((std::ofstream*)Out)->close();
      delete Out;
    }
    return 0;
  } catch (const std::string& msg) {
    std::cerr << argv[0] << ": " << msg << "\n";
  } catch (...) {
    std::cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
  }
  return 1;
}

// vim: sw=2
