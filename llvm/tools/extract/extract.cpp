//===- extract.cpp - LLVM function extraction utility ---------------------===//
//
// This utility changes the input module to only contain a single function,
// which is primarily used for debugging transformations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Transforms/IPO.h"
#include "Support/CommandLine.h"
#include <memory>

// InputFilename - The filename to read from.
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bytecode file>"),
              cl::init("-"), cl::value_desc("filename"));
              

// ExtractFunc - The function to extract from the module... defaults to main.
static cl::opt<std::string>
ExtractFunc("func", cl::desc("Specify function to extract"), cl::init("main"),
            cl::value_desc("function"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm extractor\n");

  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    std::cerr << argv[0] << ": bytecode didn't read correctly.\n";
    return 1;
  }

  // Figure out which function we should extract
  Function *F = M.get()->getNamedFunction(ExtractFunc);
  if (F == 0) {
    std::cerr << argv[0] << ": program doesn't contain function named '"
              << ExtractFunc << "'!\n";
    return 1;
  }

  // In addition to just parsing the input from GCC, we also want to spiff it up
  // a little bit.  Do this now.
  //
  PassManager Passes;
  Passes.add(createFunctionExtractionPass(F));    // Extract the function
  Passes.add(createGlobalDCEPass());              // Delete unreachable globals
  Passes.add(createFunctionResolvingPass());      // Delete prototypes
  Passes.add(createDeadTypeEliminationPass());    // Remove dead types...
  Passes.add(new WriteBytecodePass(&std::cout));  // Write bytecode to file...

  Passes.run(*M.get());
  return 0;
}
