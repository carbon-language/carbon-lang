//===- analyze.cpp - The LLVM analyze utility -----------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This utility is designed to print out the results of running various analysis
// passes on a program.  This is useful for understanding a program, or for 
// debugging an analysis pass.
//
//  analyze --help           - Output information about command line switches
//  analyze --quiet          - Do not print analysis name before output
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/PassNameParser.h"
#include "Support/Timer.h"
#include <algorithm>

using namespace llvm;

struct ModulePassPrinter : public Pass {
  const PassInfo *PassToPrint;
  ModulePassPrinter(const PassInfo *PI) : PassToPrint(PI) {}

  virtual bool run(Module &M) {
    std::cout << "Printing analysis '" << PassToPrint->getPassName() << "':\n";
    getAnalysisID<Pass>(PassToPrint).print(std::cout, &M);
    
    // Get and print pass...
    return false;
  }
  
  virtual const char *getPassName() const { return "'Pass' Printer"; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint);
    AU.setPreservesAll();
  }
};

struct FunctionPassPrinter : public FunctionPass {
  const PassInfo *PassToPrint;
  FunctionPassPrinter(const PassInfo *PI) : PassToPrint(PI) {}

  virtual bool runOnFunction(Function &F) {
    std::cout << "Printing analysis '" << PassToPrint->getPassName()
              << "' for function '" << F.getName() << "':\n";
    getAnalysisID<Pass>(PassToPrint).print(std::cout, F.getParent());

    // Get and print pass...
    return false;
  }

  virtual const char *getPassName() const { return "FunctionPass Printer"; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint);
    AU.setPreservesAll();
  }
};

struct BasicBlockPassPrinter : public BasicBlockPass {
  const PassInfo *PassToPrint;
  BasicBlockPassPrinter(const PassInfo *PI) : PassToPrint(PI) {}

  virtual bool runOnBasicBlock(BasicBlock &BB) {
    std::cout << "Printing Analysis info for BasicBlock '" << BB.getName()
              << "': Pass " << PassToPrint->getPassName() << ":\n";
    getAnalysisID<Pass>(PassToPrint).print(std::cout, BB.getParent()->getParent());

    // Get and print pass...
    return false;
  }

  virtual const char *getPassName() const { return "BasicBlockPass Printer"; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint);
    AU.setPreservesAll();
  }
};



namespace {
  cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"),
                cl::value_desc("filename"));

  cl::opt<bool> Quiet("q", cl::desc("Don't print analysis pass names"));
  cl::alias    QuietA("quiet", cl::desc("Alias for -q"),
                      cl::aliasopt(Quiet));

  cl::opt<bool> NoVerify("disable-verify", cl::Hidden,
                         cl::desc("Do not verify input module"));

  // The AnalysesList is automatically populated with registered Passes by the
  // PassNameParser.
  //
  cl::list<const PassInfo*, bool, FilteredPassNameParser<PassInfo::Analysis> >
  AnalysesList(cl::desc("Analyses available:"));

  Timer BytecodeLoadTimer("Bytecode Loader");
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm analysis printer tool\n");

  Module *CurMod = 0;
  try {
#if 0
    TimeRegion RegionTimer(BytecodeLoadTimer);
#endif
    CurMod = ParseBytecodeFile(InputFilename);
    if (!CurMod && !(CurMod = ParseAssemblyFile(InputFilename))){
      std::cerr << argv[0] << ": input file didn't read correctly.\n";
      return 1;
    }
  } catch (const ParseException &E) {
    std::cerr << argv[0] << ": " << E.getMessage() << "\n";
    return 1;
  }

  // Create a PassManager to hold and optimize the collection of passes we are
  // about to build...
  //
  PassManager Passes;

  // Add an appropriate TargetData instance for this module...
  Passes.add(new TargetData("analyze", CurMod));

  // Make sure the input LLVM is well formed.
  if (!NoVerify)
    Passes.add(createVerifierPass());

  // Create a new optimization pass for each one specified on the command line
  for (unsigned i = 0; i < AnalysesList.size(); ++i) {
    const PassInfo *Analysis = AnalysesList[i];
    
    if (Analysis->getNormalCtor()) {
      Pass *P = Analysis->getNormalCtor()();
      Passes.add(P);

      if (BasicBlockPass *BBP = dynamic_cast<BasicBlockPass*>(P))
        Passes.add(new BasicBlockPassPrinter(Analysis));
      else if (FunctionPass *FP = dynamic_cast<FunctionPass*>(P))
        Passes.add(new FunctionPassPrinter(Analysis));
      else
        Passes.add(new ModulePassPrinter(Analysis));

    } else
      std::cerr << argv[0] << ": cannot create pass: "
                << Analysis->getPassName() << "\n";
  }

  Passes.run(*CurMod);

  delete CurMod;
  return 0;
}
