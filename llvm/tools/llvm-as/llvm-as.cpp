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

#include "llvm/Module.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Analysis/Verifier.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <fstream>
#include <memory>

static cl::opt<std::string> 
InputFilename(cl::Positional, cl::desc("<input .llvm file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print assembly as parsed"), cl::Hidden);

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm .ll -> .bc assembler\n");

  std::ostream *Out = 0;
  try {
    // Parse the file now...
    std::auto_ptr<Module> M(ParseAssemblyFile(InputFilename));
    if (M.get() == 0) {
      std::cerr << argv[0] << ": assembly didn't read correctly.\n";
      return 1;
    }

    if (verifyModule(*M.get())) {
      std::cerr << argv[0]
                << ": assembly parsed, but does not verify as correct!\n";
      return 1;
    }

  
    if (DumpAsm) std::cerr << "Here's the assembly:\n" << M.get();

    if (OutputFilename != "") {   // Specified an output filename?
      if (OutputFilename != "-") {  // Not stdout?
        if (!Force && std::ifstream(OutputFilename.c_str())) {
          // If force is not specified, make sure not to overwrite a file!
          std::cerr << argv[0] << ": error opening '" << OutputFilename
                    << "': file exists!\n"
                    << "Use -f command line argument to force output\n";
          return 1;
        }
        Out = new std::ofstream(OutputFilename.c_str());
      } else {                      // Specified stdout
	Out = &std::cout;       
      }
    } else {
      if (InputFilename == "-") {
	OutputFilename = "-";
	Out = &std::cout;
      } else {
	std::string IFN = InputFilename;
	int Len = IFN.length();
	if (IFN[Len-3] == '.' && IFN[Len-2] == 'l' && IFN[Len-1] == 'l') {
	  // Source ends in .ll
	  OutputFilename = std::string(IFN.begin(), IFN.end()-3);
        } else {
	  OutputFilename = IFN;   // Append a .bc to it
	}
	OutputFilename += ".bc";

        if (!Force && std::ifstream(OutputFilename.c_str())) {
          // If force is not specified, make sure not to overwrite a file!
          std::cerr << argv[0] << ": error opening '" << OutputFilename
                    << "': file exists!\n"
                    << "Use -f command line argument to force output\n";
          return 1;
        }

	Out = new std::ofstream(OutputFilename.c_str());
        // Make sure that the Out file gets unlink'd from the disk if we get a
        // SIGINT
        RemoveFileOnSignal(OutputFilename);
      }
    }
  
    if (!Out->good()) {
      std::cerr << argv[0] << ": error opening " << OutputFilename << "!\n";
      return 1;
    }
   
    WriteBytecodeToFile(M.get(), *Out);
  } catch (const ParseException &E) {
    std::cerr << argv[0] << ": " << E.getMessage() << "\n";
    return 1;
  }

  if (Out != &std::cout) delete Out;
  return 0;
}

