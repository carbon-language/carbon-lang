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
#include "llvm/Support/CommandLine.h"

cl::String InputFilename ("", "Parse <arg> file, compile to bytecode", 0, "-");
cl::String OutputFilename("o", "Override output filename", 0, "");
cl::Flag   Force         ("f", "Overwrite output files", 0, false);
cl::Flag   DumpAsm       ("d", "Print assembly as parsed", cl::Hidden, false);

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm .ll -> .bc assembler\n");

  ostream *Out = 0;
  try {
    // Parse the file now...
    Module *C = ParseAssemblyFile(InputFilename.getValue());
    if (C == 0) {
      cerr << "assembly didn't read correctly.\n";
      return 1;
    }
  
    if (DumpAsm.getValue())
      cerr << "Here's the assembly:\n" << C;

    if (OutputFilename.getValue() != "") {   // Specified an output filename?
      Out = new ofstream(OutputFilename.getValue().c_str(), 
			 (Force.getValue() ? 0 : ios::noreplace)|ios::out);
    } else {
      if (InputFilename.getValue() == "-") {
	OutputFilename.setValue("-");
	Out = &cout;
      } else {
	string IFN = InputFilename.getValue();
	int Len = IFN.length();
	if (IFN[Len-3] == '.' && IFN[Len-2] == 'l' && IFN[Len-1] == 'l') {
	  // Source ends in .ll
	  OutputFilename.setValue(string(IFN.begin(), IFN.end()-3));
        } else {
	  OutputFilename.setValue(IFN);   // Append a .bc to it
	}
	OutputFilename.setValue(OutputFilename.getValue() + ".bc");
	Out = new ofstream(OutputFilename.getValue().c_str(), 
			   (Force.getValue() ? 0 : ios::noreplace)|ios::out);
      }
  
      if (!Out->good()) {
        cerr << "Error opening " << OutputFilename.getValue() << "!\n";
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

