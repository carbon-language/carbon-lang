//===----------------------------------------------------------------------===//
// LLVM 'LINK' UTILITY 
//
// This utility may be invoked in the following manner:
//  link a.bc b.bc c.bc -o x.bc
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Transforms/Utils/Linker.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <fstream>
#include <memory>
#include <sys/types.h>     // For FileExists
#include <sys/stat.h>

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
               cl::desc("<input bytecode files>"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"), cl::init("-"),
               cl::value_desc("filename"));

static cl::opt<bool> Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
Verbose("v", cl::desc("Print information about actions taken"));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print assembly as linked"), cl::Hidden);

static cl::list<std::string>
LibPaths("L", cl::desc("Specify a library search path"), cl::ZeroOrMore,
         cl::value_desc("directory"), cl::Prefix);

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
    if (Verbose) std::cerr << "Loading '" << Filename << "'\n";
    if (FileExists(Filename)) FoundAFile = true;
    Module *Result = ParseBytecodeFile(Filename, &ErrorMessage);
    if (Result) return std::auto_ptr<Module>(Result);   // Load successful!

    if (Verbose) {
      std::cerr << "Error opening bytecode file: '" << Filename << "'";
      if (ErrorMessage.size()) std::cerr << ": " << ErrorMessage;
      std::cerr << "\n";
    }
    
    if (NextLibPathIdx == LibPaths.size()) break;
    Filename = LibPaths[NextLibPathIdx++] + "/" + FN;
  }

  if (FoundAFile)
    std::cerr << "Bytecode file '" << FN << "' corrupt!  "
              << "Use 'link -v ...' for more info.\n";
  else
    std::cerr << "Could not locate bytecode file: '" << FN << "'\n";
  return std::auto_ptr<Module>();
}




int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm linker\n");
  assert(InputFilenames.size() > 0 && "OneOrMore is not working");

  unsigned BaseArg = 0;
  std::string ErrorMessage;

  std::auto_ptr<Module> Composite(LoadFile(InputFilenames[BaseArg]));
  if (Composite.get() == 0) return 1;

  for (unsigned i = BaseArg+1; i < InputFilenames.size(); ++i) {
    std::auto_ptr<Module> M(LoadFile(InputFilenames[i]));
    if (M.get() == 0) return 1;

    if (Verbose) std::cerr << "Linking in '" << InputFilenames[i] << "'\n";

    if (LinkModules(Composite.get(), M.get(), &ErrorMessage)) {
      std::cerr << argv[0] << ": error linking in '" << InputFilenames[i]
                << "': " << ErrorMessage << "\n";
      return 1;
    }
  }

  if (DumpAsm) std::cerr << "Here's the assembly:\n" << Composite.get();

  std::ostream *Out = &std::cout;  // Default to printing to stdout...
  if (OutputFilename != "-") {
    if (!Force && std::ifstream(OutputFilename.c_str())) {
      // If force is not specified, make sure not to overwrite a file!
      std::cerr << argv[0] << ": error opening '" << OutputFilename
                << "': file exists!\n"
                << "Use -f command line argument to force output\n";
      return 1;
    }
    Out = new std::ofstream(OutputFilename.c_str());
    if (!Out->good()) {
      std::cerr << argv[0] << ": error opening '" << OutputFilename << "'!\n";
      return 1;
    }

    // Make sure that the Out file gets unlink'd from the disk if we get a
    // SIGINT
    RemoveFileOnSignal(OutputFilename);
  }

  if (verifyModule(*Composite.get())) {
    std::cerr << argv[0] << ": linked module is broken!\n";
    return 1;
  }

  if (Verbose) std::cerr << "Writing bytecode...\n";
  WriteBytecodeToFile(Composite.get(), *Out);

  if (Out != &std::cout) delete Out;
  return 0;
}
