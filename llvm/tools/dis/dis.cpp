//===------------------------------------------------------------------------===
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
//===------------------------------------------------------------------------===

#include <iostream.h>
#include <fstream.h>
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Tools/CommandLine.h"
#include "llvm/Method.h"
#include "llvm/CFG.h"

int main(int argc, char **argv) {
  // WriteMode - The different orderings to print basic blocks in...
  enum {
    Default = 0,                  // Method Order (list order)
    DepthFirst,                   // Depth First ordering
    ReverseDepthFirst,            // Reverse Depth First ordering
    PostOrder,                    // Post Order
    ReversePostOrder              // Reverse Post Order
  } WriteMode = Default;

  ToolCommandLine Opts(argc, argv, false);

  // We only support the options that the system parser does... if it left any
  // then we don't know what to do.
  //
  if (argc > 1) {
    for (int i = 1; i < argc; i++) {
      if (string(argv[i]) == string("--help")) {
	cerr << argv[0] << " usage:\n"
	     << "\tx.bc        - Parse <x.bc> file and output to x.ll\n"
	     << "\tno .bc file - Parse stdin and write to stdout.\n"
	     << "\t-dfo        - Write basic blocks in depth first order.\n"
	     << "\t-rdfo       - Write basic blocks in reverse DFO.\n"
	     << "\t-po         - Write basic blocks in postorder.\n"
	     << "\t-rpo        - Write basic blocks in reverse postorder.\n"
	     << "\t--help      - Print this usage information\n\n";
	return 1;
      } else if (string(argv[i]) == string("-dfo")) {
	WriteMode = DepthFirst;
      } else if (string(argv[i]) == string("-rdfo")) {
	WriteMode = ReverseDepthFirst;
      } else if (string(argv[i]) == string("-po")) {
	WriteMode = PostOrder;
      } else if (string(argv[i]) == string("-rpo")) {
	WriteMode = ReversePostOrder;
      } else {
	cerr << argv[0] << ": argument not recognized: '" << argv[i] << "'!\n";
      }
    }
  }
  
  ostream *Out = &cout;  // Default to printing to stdout...

  Module *C = ParseBytecodeFile(Opts.getInputFilename());
  if (C == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }
  
  if (Opts.getOutputFilename() != "-") {
    Out = new ofstream(Opts.getOutputFilename().c_str(), 
                       (Opts.getForce() ? 0 : ios::noreplace)|ios::out);
    if (!Out->good()) {
      cerr << "Error opening " << Opts.getOutputFilename() 
           << ": sending to stdout instead!\n";
      Out = &cout;
      }
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
    for (Module::MethodListType::iterator I = C->getMethodList().begin(); 
	 I != C->getMethodList().end(); I++) {
      Method *M = *I;
      (*Out) << "-------------- Method: " << M->getName() << " -------------\n";

      switch (WriteMode) {
      case DepthFirst:                   // Depth First ordering
	copy(cfg::df_begin(M), cfg::df_end(M),
	     ostream_iterator<BasicBlock*>(*Out, "\n"));
	break;
      case ReverseDepthFirst:            // Reverse Depth First ordering
	copy(cfg::df_begin(M, true), cfg::df_end(M),
	     ostream_iterator<BasicBlock*>(*Out, "\n"));
	break;
      case PostOrder:                    // Post Order
	copy(cfg::po_begin(M), cfg::po_end(M),
	     ostream_iterator<BasicBlock*>(*Out, "\n"));
	break;
      case ReversePostOrder: {           // Reverse Post Order
	cfg::ReversePostOrderTraversal RPOT(M);
	copy(RPOT.begin(), RPOT.end(),
	     ostream_iterator<BasicBlock*>(*Out, "\n"));
	break;
      }
      default:
	abort();
	break;
      }
    }
  }
  delete C;

  if (Out != &cout) delete Out;
  return 0;
}
