//===------------------------------------------------------------------------===
// LLVM 'DIS' UTILITY 
//
// This utility may be invoked in the following manner:
//  dis --help     - Output information about command line switches
//  dis [options]      - Read LLVM bytecode from stdin, write assembly to stdout
//  dis [options] x.bc - Read LLVM bytecode from the x.bc file, write assembly
//                       to the x.ll file.
//
//===------------------------------------------------------------------------===

#include <iostream.h>
#include <fstream.h>
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Tools/CommandLine.h"

int main(int argc, char **argv) {
  ToolCommandLine Opts(argc, argv, false);

  // We only support the options that the system parser does... if it left any
  // then we don't know what to do.
  //
  if (argc > 1) {
    for (int i = 1; i < argc; i++) {
      if (string(argv[i]) != string("--help"))
	cerr << argv[0] << ": argument not recognized: '" << argv[i] << "'!\n";
    }
    
    cerr << argv[0] << " usage:\n"
	 << "  " << argv[0] << " --help  - Print this usage information\n" 
	 << "  " << argv[0] << " x.bc    - Parse <x.bc> file and output "
	 << "assembly to x.ll\n"
	 << "  " << argv[0] << "         - Parse stdin and write to stdout.\n";
    return 1;
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
  (*Out) << C;
  delete C;

  if (Out != &cout) delete Out;
  return 0;
}
