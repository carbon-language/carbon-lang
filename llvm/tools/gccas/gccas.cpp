//===------------------------------------------------------------------------===
// LLVM 'GCCAS' UTILITY 
//
//  This utility is designed to be used by the GCC frontend for creating
// bytecode files from it's intermediate llvm assembly.  The requirements for
// this utility are thus slightly different than that of the standard as util.
//
//===------------------------------------------------------------------------===

#include "llvm/Module.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Transforms/CleanupGCCOutput.h"
#include "llvm/Optimizations/DCE.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Support/CommandLine.h"
#include <memory>
#include <fstream>
#include <string>

cl::String InputFilename ("", "Parse <arg> file, compile to bytecode",
                          cl::Required, "");
cl::String OutputFilename("o", "Override output filename", cl::NoFlags, "");

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm .s -> .o assembler for GCC\n");

  ostream *Out = 0;
  std::auto_ptr<Module> M;
  try {
    // Parse the file now...
    M.reset(ParseAssemblyFile(InputFilename));
  } catch (const ParseException &E) {
    cerr << E.getMessage() << endl;
    return 1;
  }

  if (M.get() == 0) {
    cerr << "assembly didn't read correctly.\n";
    return 1;
  }
  
  if (OutputFilename == "") {   // Didn't specify an output filename?
    string IFN = InputFilename;
    int Len = IFN.length();
    if (IFN[Len-2] == '.' && IFN[Len-1] == 's') {   // Source ends in .s?
      OutputFilename = string(IFN.begin(), IFN.end()-2);
    } else {
      OutputFilename = IFN;   // Append a .o to it
    }
    OutputFilename += ".o";
  }

  Out = new ofstream(OutputFilename.c_str(), ios::out);
  if (!Out->good()) {
    cerr << "Error opening " << OutputFilename << "!\n";
    return 1;
  }

  // In addition to just parsing the input from GCC, we also want to spiff it up
  // a little bit.  Do this now.
  //
  vector<Pass*> Passes;
  Passes.push_back(new CleanupGCCOutput());
  Passes.push_back(new opt::DeadCodeElimination());

  // Run our queue of passes all at once now, efficiently.  This form of
  // runAllPasses frees the Pass objects after runAllPasses completes.
  //
  Pass::runAllPassesAndFree(M.get(), Passes);

  WriteBytecodeToFile(M.get(), *Out);
  return 0;
}

