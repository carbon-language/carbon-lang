//===- extract.cpp - LLVM function extraction utility ---------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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
#include "llvm/Target/TargetData.h"
#include "Support/CommandLine.h"
#include "llvm/System/Signals.h"
#include <memory>
#include <fstream>
using namespace llvm;

// InputFilename - The filename to read from.
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bytecode file>"),
              cl::init("-"), cl::value_desc("filename"));
              
static cl::opt<std::string>
OutputFilename("o", cl::desc("Specify output filename"), 
               cl::value_desc("filename"), cl::init("-"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
DeleteFn("delete", cl::desc("Delete specified function from Module"));

// ExtractFunc - The function to extract from the module... defaults to main.
static cl::opt<std::string>
ExtractFunc("func", cl::desc("Specify function to extract"), cl::init("main"),
            cl::value_desc("function"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm extractor\n");
  PrintStackTraceOnErrorSignal();

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

  // In addition to deleting all other functions, we also want to spiff it up a
  // little bit.  Do this now.
  //
  PassManager Passes;
  Passes.add(new TargetData("extract", M.get())); // Use correct TargetData
  // Either isolate the function or delete it from the Module
  Passes.add(createFunctionExtractionPass(F, DeleteFn));
  Passes.add(createGlobalDCEPass());              // Delete unreachable globals
  Passes.add(createFunctionResolvingPass());      // Delete prototypes
  Passes.add(createDeadTypeEliminationPass());    // Remove dead types...

  std::ostream *Out = 0;

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

  Passes.add(new WriteBytecodePass(Out));  // Write bytecode to file...
  Passes.run(*M.get());

  if (Out != &std::cout)
    delete Out;
  return 0;
}
