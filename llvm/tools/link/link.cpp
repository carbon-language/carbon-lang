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

#include "llvm/Transforms/Linker.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "Support/CommandLine.h"
#include <fstream>
#include <memory>
#include <sys/types.h>     // For FileExists
#include <sys/stat.h>


cl::StringList InputFilenames("", "Load <arg> files, linking them together", 
			      cl::OneOrMore);
cl::String OutputFilename("o", "Override output filename", cl::NoFlags, "-");
cl::Flag   Force         ("f", "Overwrite output files", cl::NoFlags, false);
cl::Flag   Verbose       ("v", "Print information about actions taken");
cl::Flag   DumpAsm       ("d", "Print assembly as linked", cl::Hidden, false);
cl::StringList LibPaths  ("L", "Specify a library search path", cl::ZeroOrMore);


// FileExists - Return true if the specified string is an openable file...
static inline bool FileExists(const string &FN) {
  struct stat StatBuf;
  return stat(FN.c_str(), &StatBuf) != -1;
}

// LoadFile - Read the specified bytecode file in and return it.  This routine
// searches the link path for the specified file to try to find it...
//
static inline std::auto_ptr<Module> LoadFile(const string &FN) {
  string Filename = FN;
  string ErrorMessage;

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
      cerr << endl;
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
  string ErrorMessage;

  // TODO: TEST argv[0] for llvm-ar forms... for now, this is a huge hack.
  if (InputFilenames.size() >= 3 && InputFilenames[0] == "rc" &&
      OutputFilename == "-") {
    BaseArg = 2;
    OutputFilename = InputFilenames[1];
  }

  std::auto_ptr<Module> Composite(LoadFile(InputFilenames[BaseArg]));
  if (Composite.get() == 0) return 1;

  for (unsigned i = BaseArg+1; i < InputFilenames.size(); ++i) {
    auto_ptr<Module> M(LoadFile(InputFilenames[i]));
    if (M.get() == 0) return 1;

    if (Verbose) cerr << "Linking in '" << InputFilenames[i] << "'\n";

    if (LinkModules(Composite.get(), M.get(), &ErrorMessage)) {
      cerr << "Error linking in '" << InputFilenames[i] << "': "
	   << ErrorMessage << endl;
      return 1;
    }
  }

  if (DumpAsm)
    cerr << "Here's the assembly:\n" << Composite.get();

  ostream *Out = &cout;  // Default to printing to stdout...
  if (OutputFilename != "-") {
    Out = new ofstream(OutputFilename.c_str(), 
		       (Force ? 0 : ios::noreplace)|ios::out);
    if (!Out->good()) {
      cerr << "Error opening '" << OutputFilename << "'!\n";
      return 1;
    }
  }

  if (Verbose) cerr << "Writing bytecode...\n";
  WriteBytecodeToFile(Composite.get(), *Out);

  if (Out != &cout) delete Out;
  return 0;
}
