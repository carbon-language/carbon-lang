//===----------------------------------------------------------------------===//
// LLVM 'LINK' UTILITY 
//
// This utility may be invoked in the following manner:
//  link a.bc b.bc c.bc -o x.bc
//
// Alternatively, this can be used as an 'ar' tool as well.  If invoked as
// either 'ar' or 'llvm-ar', it accepts a 'rc' parameter as well.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Linker.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Module.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <fstream>
#include <memory>
#include <sys/types.h>     // For FileExists
#include <sys/stat.h>
#include <iostream>

using std::cerr;

cl::StringList InputFilenames("", "Load <arg> files, linking them together", 
			      cl::OneOrMore);
cl::String OutputFilename("o", "Override output filename", cl::NoFlags, "-");
cl::Flag   Force         ("f", "Overwrite output files", cl::NoFlags, false);
cl::Flag   Verbose       ("v", "Print information about actions taken");
cl::Flag   DumpAsm       ("d", "Print assembly as linked", cl::Hidden, false);
cl::StringList LibPaths  ("L", "Specify a library search path", cl::ZeroOrMore);

// FileExists - Return true if the specified string is an openable file...
static inline bool FileExists(const std::string &FN) {
  struct stat StatBuf;
  return stat(FN.c_str(), &StatBuf) != -1;
}

// LoadFile - Read the specified bytecode file in and return it.  This routine
// searches the link path for the specified file to try to find it...
//
static inline std::auto_ptr<Module> LoadFile(const std::string &FN) {
  std::string Filename = FN;
  std::string ErrorMessage;

  unsigned NextLibPathIdx = 0;
  bool FoundAFile = false;

  while (1) {
    if (Verbose) cerr << "Loading '" << Filename << "'\n";
    if (FileExists(Filename)) FoundAFile = true;
    Module *Result = ParseBytecodeFile(Filename, &ErrorMessage);
    if (Result) return std::auto_ptr<Module>(Result);   // Load successful!

    if (Verbose) {
      cerr << "Error opening bytecode file: '" << Filename << "'";
      if (ErrorMessage.size()) cerr << ": " << ErrorMessage;
      cerr << "\n";
    }
    
    if (NextLibPathIdx == LibPaths.size()) break;
    Filename = LibPaths[NextLibPathIdx++] + "/" + FN;
  }

  if (FoundAFile)
    cerr << "Bytecode file '" << FN << "' corrupt!  "
         << "Use 'link -v ...' for more info.\n";
  else
    cerr << "Could not locate bytecode file: '" << FN << "'\n";
  return std::auto_ptr<Module>();
}




int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm linker\n",
			      cl::EnableSingleLetterArgValue |
			      cl::DisableSingleLetterArgGrouping);
  assert(InputFilenames.size() > 0 && "OneOrMore is not working");

  unsigned BaseArg = 0;
  std::string ErrorMessage;

  // TODO: TEST argv[0] for llvm-ar forms... for now, this is a huge hack.
  if (InputFilenames.size() >= 3 && InputFilenames[0] == "rc" &&
      OutputFilename == "-") {
    BaseArg = 2;
    OutputFilename = InputFilenames[1];
  }

  std::auto_ptr<Module> Composite(LoadFile(InputFilenames[BaseArg]));
  if (Composite.get() == 0) return 1;

  for (unsigned i = BaseArg+1; i < InputFilenames.size(); ++i) {
    std::auto_ptr<Module> M(LoadFile(InputFilenames[i]));
    if (M.get() == 0) return 1;

    if (Verbose) cerr << "Linking in '" << InputFilenames[i] << "'\n";

    if (LinkModules(Composite.get(), M.get(), &ErrorMessage)) {
      cerr << "Error linking in '" << InputFilenames[i] << "': "
	   << ErrorMessage << "\n";
      return 1;
    }
  }

  if (DumpAsm) cerr << "Here's the assembly:\n" << Composite.get();

  std::ostream *Out = &std::cout;  // Default to printing to stdout...
  if (OutputFilename != "-") {
    if (!Force && std::ifstream(OutputFilename.c_str())) {
      // If force is not specified, make sure not to overwrite a file!
      cerr << "Error opening '" << OutputFilename << "': File exists!\n"
           << "Use -f command line argument to force output\n";
      return 1;
    }
    Out = new std::ofstream(OutputFilename.c_str());
    if (!Out->good()) {
      cerr << "Error opening '" << OutputFilename << "'!\n";
      return 1;
    }

    // Make sure that the Out file gets unlink'd from the disk if we get a
    // SIGINT
    RemoveFileOnSignal(OutputFilename);
  }

  if (Verbose) cerr << "Writing bytecode...\n";
  WriteBytecodeToFile(Composite.get(), *Out);

  if (Out != &std::cout) delete Out;
  return 0;
}
