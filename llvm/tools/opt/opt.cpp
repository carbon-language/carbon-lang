//===----------------------------------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// LLVM Modular Optimizer Utility: opt
//
// Optimizations may be specified an arbitrary number of times on the command
// line, they are run in the order specified.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/Support/PassNameParser.h"
#include "Support/Signals.h"
#include <fstream>
#include <memory>
#include <algorithm>

using namespace llvm;

// The OptimizationList is automatically populated with registered Passes by the
// PassNameParser.
//
static cl::list<const PassInfo*, bool,
                FilteredPassNameParser<PassInfo::Optimization> >
OptimizationList(cl::desc("Optimizations available:"));


// Other command line options...
//
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bytecode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
PrintEachXForm("p", cl::desc("Print module after each transformation"));

static cl::opt<bool>
NoOutput("disable-output",
         cl::desc("Do not write result bytecode file"), cl::Hidden);

static cl::opt<bool>
NoVerify("disable-verify", cl::desc("Do not verify result module"), cl::Hidden);

static cl::opt<bool>
Quiet("q", cl::desc("Don't print 'program modified' message"));

static cl::alias
QuietA("quiet", cl::desc("Alias for -q"), cl::aliasopt(Quiet));


//===----------------------------------------------------------------------===//
// main for opt
//
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
			      " llvm .bc -> .bc modular optimizer\n");

  // Allocate a full target machine description only if necessary...
  // FIXME: The choice of target should be controllable on the command line.
  std::auto_ptr<TargetMachine> target;

  TargetMachine* TM = NULL;
  std::string ErrorMessage;

  // Load the input module...
  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename, &ErrorMessage));
  if (M.get() == 0) {
    std::cerr << argv[0] << ": ";
    if (ErrorMessage.size())
      std::cerr << ErrorMessage << "\n";
    else
      std::cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  // Figure out what stream we are supposed to write to...
  std::ostream *Out = &std::cout;  // Default to printing to stdout...
  if (OutputFilename != "") {
    if (!Force && std::ifstream(OutputFilename.c_str())) {
      // If force is not specified, make sure not to overwrite a file!
      std::cerr << argv[0] << ": error opening '" << OutputFilename
                << "': file exists!\n"
                << "Use -f command line argument to force output\n";
      return 1;
    }
    Out = new std::ofstream(OutputFilename.c_str());

    if (!Out->good()) {
      std::cerr << argv[0] << ": error opening " << OutputFilename << "!\n";
      return 1;
    }

    // Make sure that the Output file gets unlinked from the disk if we get a
    // SIGINT
    RemoveFileOnSignal(OutputFilename);
  }

  // Create a PassManager to hold and optimize the collection of passes we are
  // about to build...
  //
  PassManager Passes;

  // Add an appropriate TargetData instance for this module...
  Passes.add(new TargetData("opt", M.get()));

  // Create a new optimization pass for each one specified on the command line
  for (unsigned i = 0; i < OptimizationList.size(); ++i) {
    const PassInfo *Opt = OptimizationList[i];
    
    if (Opt->getNormalCtor())
      Passes.add(Opt->getNormalCtor()());
    else if (Opt->getTargetCtor()) {
#if 0
      if (target.get() == NULL)
        target.reset(allocateSparcTargetMachine()); // FIXME: target option
#endif
      assert(target.get() && "Could not allocate target machine!");
      Passes.add(Opt->getTargetCtor()(*target.get()));
    } else
      std::cerr << argv[0] << ": cannot create pass: " << Opt->getPassName()
                << "\n";

    if (PrintEachXForm)
      Passes.add(new PrintModulePass(&std::cerr));
  }

  // Check that the module is well formed on completion of optimization
  if (!NoVerify)
    Passes.add(createVerifierPass());

  // Write bytecode out to disk or cout as the last step...
  if (!NoOutput)
    Passes.add(new WriteBytecodePass(Out, Out != &std::cout));

  // Now that we have all of the passes ready, run them.
  if (Passes.run(*M.get()) && !Quiet)
    std::cerr << "Program modified.\n";

  return 0;
}
