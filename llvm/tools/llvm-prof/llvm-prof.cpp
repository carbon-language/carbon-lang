//===- llvm-prof.cpp - Read in and process llvmprof.out data files --------===//
// 
//                      The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This tools is meant for use with the various LLVM profiling instrumentation
// passes.  It reads in the data file produced by executing an instrumented
// program, and outputs a nice report.
//
//===----------------------------------------------------------------------===//

#include "ProfileInfo.h"
#include "llvm/Module.h"
#include "llvm/Assembly/AsmAnnotationWriter.h"
#include "llvm/Bytecode/Reader.h"
#include "Support/CommandLine.h"
#include <iostream>
#include <cstdio>
#include <map>
#include <set>

namespace {
  cl::opt<std::string> 
  BytecodeFile(cl::Positional, cl::desc("<program bytecode file>"),
               cl::Required);

  cl::opt<std::string> 
  ProfileDataFile(cl::Positional, cl::desc("<llvmprof.out file>"),
                  cl::Optional, cl::init("llvmprof.out"));

  cl::opt<bool>
  PrintAnnotatedLLVM("annotated-llvm",
                     cl::desc("Print LLVM code with frequency annotations"));
  cl::alias PrintAnnotated2("A", cl::desc("Alias for --annotated-llvm"),
                            cl::aliasopt(PrintAnnotatedLLVM));
}

// PairSecondSort - A sorting predicate to sort by the second element of a pair.
template<class T>
struct PairSecondSortReverse
  : public std::binary_function<std::pair<T, unsigned>,
                                std::pair<T, unsigned>, bool> {
  bool operator()(const std::pair<T, unsigned> &LHS,
                  const std::pair<T, unsigned> &RHS) const {
    return LHS.second > RHS.second;
  }
};

namespace {
  class ProfileAnnotator : public AssemblyAnnotationWriter {
    std::map<const Function  *, unsigned> &FuncFreqs;
    std::map<const BasicBlock*, unsigned> &BlockFreqs;
  public:
    ProfileAnnotator(std::map<const Function  *, unsigned> &FF,
                     std::map<const BasicBlock*, unsigned> &BF)
      : FuncFreqs(FF), BlockFreqs(BF) {}

    virtual void emitFunctionAnnot(const Function *F, std::ostream &OS) {
      OS << ";;; %" << F->getName() << " called " << FuncFreqs[F]
         << " times.\n;;;\n";
    }
    virtual void emitBasicBlockAnnot(const BasicBlock *BB, std::ostream &OS) {
      if (unsigned Count = BlockFreqs[BB])
        OS << ";;; Executed " << Count << " times.\n";
      else
        OS << ";;; Never executed!\n";
    }
  };
}


int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm profile dump decoder\n");

  // Read in the bytecode file...
  std::string ErrorMessage;
  Module *M = ParseBytecodeFile(BytecodeFile, &ErrorMessage);
  if (M == 0) {
    std::cerr << argv[0] << ": " << BytecodeFile << ": " << ErrorMessage
              << "\n";
    return 1;
  }

  // Read the profiling information
  ProfileInfo PI(argv[0], ProfileDataFile, *M);

  std::map<const Function  *, unsigned> FuncFreqs;
  std::map<const BasicBlock*, unsigned> BlockFreqs;

  // Output a report.  Eventually, there will be multiple reports selectable on
  // the command line, for now, just keep things simple.

  // Emit the most frequent function table...
  std::vector<std::pair<Function*, unsigned> > FunctionCounts;
  PI.getFunctionCounts(FunctionCounts);
  FuncFreqs.insert(FunctionCounts.begin(), FunctionCounts.end());

  // Sort by the frequency, backwards.
  std::sort(FunctionCounts.begin(), FunctionCounts.end(),
            PairSecondSortReverse<Function*>());

  unsigned TotalExecutions = 0;
  for (unsigned i = 0, e = FunctionCounts.size(); i != e; ++i)
    TotalExecutions += FunctionCounts[i].second;
  
  std::cout << "===" << std::string(73, '-') << "===\n"
            << "LLVM profiling output for execution";
  if (PI.getNumExecutions() != 1) std::cout << "s";
  std::cout << ":\n";
  
  for (unsigned i = 0, e = PI.getNumExecutions(); i != e; ++i) {
    std::cout << "  ";
    if (e != 1) std::cout << i+1 << ". ";
    std::cout << PI.getExecution(i) << "\n";
  }
  
  std::cout << "\n===" << std::string(73, '-') << "===\n";
  std::cout << "Function execution frequencies:\n\n";

  // Print out the function frequencies...
  printf(" ##   Frequency\n");
  for (unsigned i = 0, e = FunctionCounts.size(); i != e; ++i) {
    if (FunctionCounts[i].second == 0) {
      printf("\n  NOTE: %d function%s never executed!\n",
             e-i, e-i-1 ? "s were" : " was");
      break;
    }

    printf("%3d. %5d/%d %s\n", i+1, FunctionCounts[i].second, TotalExecutions,
           FunctionCounts[i].first->getName().c_str());
  }

  std::set<Function*> FunctionsToPrint;

  // If we have block count information, print out the LLVM module with
  // frequency annotations.
  if (PI.hasAccurateBlockCounts()) {
    std::vector<std::pair<BasicBlock*, unsigned> > Counts;
    PI.getBlockCounts(Counts);

    TotalExecutions = 0;
    for (unsigned i = 0, e = Counts.size(); i != e; ++i)
      TotalExecutions += Counts[i].second;

    // Sort by the frequency, backwards.
    std::sort(Counts.begin(), Counts.end(),
              PairSecondSortReverse<BasicBlock*>());
    
    std::cout << "\n===" << std::string(73, '-') << "===\n";
    std::cout << "Top 20 most frequently executed basic blocks:\n\n";

    // Print out the function frequencies...
    printf(" ##   Frequency\n");
    unsigned BlocksToPrint = Counts.size();
    if (BlocksToPrint > 20) BlocksToPrint = 20;
    for (unsigned i = 0; i != BlocksToPrint; ++i) {
      Function *F = Counts[i].first->getParent();
      printf("%3d. %5d/%d %s() - %s\n", i+1, Counts[i].second, TotalExecutions,
             F->getName().c_str(), Counts[i].first->getName().c_str());
      FunctionsToPrint.insert(F);
    }

    BlockFreqs.insert(Counts.begin(), Counts.end());
  }
  
  if (PrintAnnotatedLLVM) {
    std::cout << "\n===" << std::string(73, '-') << "===\n";
    std::cout << "Annotated LLVM code for the module:\n\n";
    
    if (FunctionsToPrint.empty())
      for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
        FunctionsToPrint.insert(I);
    
    ProfileAnnotator PA(FuncFreqs, BlockFreqs);

    for (std::set<Function*>::iterator I = FunctionsToPrint.begin(),
           E = FunctionsToPrint.end(); I != E; ++I)
      (*I)->print(std::cout, &PA);
  }

  return 0;
}
