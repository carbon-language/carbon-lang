//===----------------------------------------------------------------------===//
// LLVM 'LINK' UTILITY 
//
// This utility may be invoked in the following manner:
//  link a.bc b.bc c.bc -o x.bc
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Linker.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include <fstream.h>
#include <memory>


cl::StringList InputFilenames("", "Load <arg> files, linking them together", 
			      cl::OneOrMore);
cl::String OutputFilename("o", "Override output filename", cl::NoFlags, "-");
cl::Flag   Force         ("f", "Overwrite output files", cl::NoFlags, false);
cl::Flag   DumpAsm       ("d", "Print assembly as linked", cl::Hidden, false);


int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm linker\n");
  assert(InputFilenames.size() > 0 && "OneOrMore is not working");

  std::auto_ptr<Module> Composite(ParseBytecodeFile(InputFilenames[0]));
  if (Composite.get() == 0) {
    cerr << "Error opening bytecode file: '" << InputFilenames[0] << "'\n";
    return 1;
  }

  for (unsigned i = 1; i < InputFilenames.size(); ++i) {
    auto_ptr<Module> M(ParseBytecodeFile(InputFilenames[i]));
    if (M.get() == 0) {
      cerr << "Error opening bytecode file: '" << InputFilenames[i] << "'\n";
      return 1;
    }
    
    string ErrorMessage;
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

  WriteBytecodeToFile(Composite.get(), *Out);

  if (Out != &cout) delete Out;
  return 0;
}
