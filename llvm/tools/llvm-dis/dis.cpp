//===----------------------------------------------------------------------===//
// LLVM 'DIS' UTILITY 
//
// This utility may be invoked in the following manner:
//  dis [options]      - Read LLVM bytecode from stdin, write assembly to stdout
//  dis [options] x.bc - Read LLVM bytecode from the x.bc file, write assembly
//                       to the x.ll file.
//  Options:
//      --help   - Output information about command line switches
//       -dfo    - Print basic blocks in depth first order
//       -rdfo   - Print basic blocks in reverse depth first order
//       -po     - Print basic blocks in post order
//       -rpo    - Print basic blocks in reverse post order
//
// TODO: add -vcg which prints VCG compatible output.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Method.h"
#include "llvm/Support/CFG.h"
#include "Support/DepthFirstIterator.h"
#include "Support/PostOrderIterator.h"
#include "Support/CommandLine.h"
#include <fstream>
#include <iostream>
using std::cerr;

// OutputMode - The different orderings to print basic blocks in...
enum OutputMode {
  Default = 0,           // Method Order (list order)
  dfo,                   // Depth First ordering
  rdfo,                  // Reverse Depth First ordering
  po,                    // Post Order
  rpo,                   // Reverse Post Order
};

cl::String InputFilename ("", "Load <arg> file, print as assembly", 0, "-");
cl::String OutputFilename("o", "Override output filename", cl::NoFlags, "");
cl::Flag   Force         ("f", "Overwrite output files", cl::NoFlags, false);
cl::EnumFlags<enum OutputMode> WriteMode(cl::NoFlags,
  clEnumVal(Default, "Write basic blocks in bytecode order"),
  clEnumVal(dfo    , "Write basic blocks in depth first order"),
  clEnumVal(rdfo   , "Write basic blocks in reverse DFO"),
  clEnumVal(po     , "Write basic blocks in postorder"),
  clEnumVal(rpo    , "Write basic blocks in reverse postorder"),
 0);

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm .bc -> .ll disassembler\n");
  std::ostream *Out = &std::cout;  // Default to printing to stdout...

  Module *C = ParseBytecodeFile(InputFilename);
  if (C == 0) {
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
      OutputFilename += ".ll";

      if (!Force && std::ifstream(OutputFilename.c_str())) {
        // If force is not specified, make sure not to overwrite a file!
        cerr << "Error opening '" << OutputFilename
             << "': File exists! Sending to standard output.\n";
      } else {
        Out = new std::ofstream(OutputFilename.c_str());
      }
    }
  }

  if (!Out->good()) {
    cerr << "Error opening " << OutputFilename
	 << ": sending to stdout instead!\n";
    Out = &std::cout;
  }

  // All that dis does is write the assembly out to a file... which is exactly
  // what the writer library is supposed to do...
  //
  if (WriteMode == Default) {
    (*Out) << C;           // Print out in list order
  } else {
    // TODO: This does not print anything other than the basic blocks in the
    // methods... more should definately be printed.  It should be valid output
    // consumable by the assembler.
    //
    for (Module::iterator I = C->begin(), End = C->end(); I != End; ++I) {
      Method *M = *I;
      (*Out) << "-------------- Method: " << M->getName() << " -------------\n";

      switch (WriteMode) {
      case dfo:                   // Depth First ordering
	copy(df_begin(M), df_end(M),
	     std::ostream_iterator<BasicBlock*>(*Out, "\n"));
	break;
      case rdfo:            // Reverse Depth First ordering
	copy(df_begin(M, true), df_end(M),
	     std::ostream_iterator<BasicBlock*>(*Out, "\n"));
	break;
      case po:                    // Post Order
	copy(po_begin(M), po_end(M),
	     std::ostream_iterator<BasicBlock*>(*Out, "\n"));
	break;
      case rpo: {           // Reverse Post Order
#if 0  // FIXME, GCC 3.0.4 bug
	ReversePostOrderTraversal<Method*> RPOT(M());
	copy(RPOT.begin(), RPOT.end(),
	     std::ostream_iterator<BasicBlock*>(*Out, "\n"));
#endif
	break;
      }
      default:
	abort();
	break;
      }
    }
  }
  delete C;

  if (Out != &std::cout) delete Out;
  return 0;
}
