//===------------------------------------------------------------------------===
// LLVM 'AS' UTILITY 
//
//  This utility may be invoked in the following manner:
//   as --help     - Output information about command line switches
//   as [options]      - Read LLVM assembly from stdin, write bytecode to stdout
//   as [options] x.ll - Read LLVM assembly from the x.ll file, write bytecode
//                       to the x.bc file.
//
//===------------------------------------------------------------------------===

#include <iostream.h>
#include <fstream.h>
#include <string>
#include "llvm/Module.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Tools/CommandLine.h"


int main(int argc, char **argv) {
  ToolCommandLine Opts(argc, argv);
  bool DumpAsm = false;

  for (int i = 1; i < argc; i++) {
    if (string(argv[i]) == string("-d")) {
      argv[i] = 0; DumpAsm = true;
    }
  }

  bool PrintMessage = false;
  for (int i = 1; i < argc; i++) {
    if (argv[i] == 0) continue;

    if (string(argv[i]) == string("--help")) {
      PrintMessage = true;
    } else {
      cerr << argv[0] << ": argument not recognized: '" << argv[i] << "'!\n";
    }
  }

  if (PrintMessage) {
    cerr << argv[0] << " usage:\n"
         << "  " << argv[0] << " --help  - Print this usage information\n" 
         << "  " << argv[0] << " x.ll    - Parse <x.ll> file and output "
         << "bytecodes to x.bc\n"
         << "  " << argv[0] << "         - Parse stdin and write to stdout.\n";
    return 1;
  }

  ostream *Out = &cout;    // Default to output to stdout...
  try {
    // Parse the file now...
    Module *C = ParseAssemblyFile(Opts);
    if (C == 0) {
      cerr << "assembly didn't read correctly.\n";
      return 1;
    }
  
    if (DumpAsm) 
      cerr << "Here's the assembly:\n" << C;
  
    if (Opts.getOutputFilename() != "-") {
      Out = new ofstream(Opts.getOutputFilename().c_str(), 
			 (Opts.getForce() ? 0 : ios::noreplace)|ios::out);
      if (!Out->good()) {
        cerr << "Error opening " << Opts.getOutputFilename() << "!\n";
	delete C;
	return 1;
      }
    }
   
    WriteBytecodeToFile(C, *Out);

    delete C;
  } catch (const ParseException &E) {
    cerr << E.getMessage() << endl;
    return 1;
  }

  if (Out != &cout) delete Out;
  return 0;
}

