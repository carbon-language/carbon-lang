//===- llvm-prof.cpp - Read in and process llvmprof.out data files --------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tools is meant for use with the various LLVM profiling instrumentation
// passes.  It reads in the data file produced by executing an instrumented
// program, and outputs a nice report.
//
//===----------------------------------------------------------------------===//

#include "llvm/InstrTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Assembly/AsmAnnotationWriter.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Analysis/ProfileInfoLoader.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>
#include <set>

using namespace llvm;

namespace {
  cl::opt<std::string>
  BitcodeFile(cl::Positional, cl::desc("<program bitcode file>"),
              cl::Required);

  cl::opt<std::string>
  ProfileDataFile(cl::Positional, cl::desc("<llvmprof.out file>"),
                  cl::Optional, cl::init("llvmprof.out"));

  cl::opt<bool>
  PrintAnnotatedLLVM("annotated-llvm",
                     cl::desc("Print LLVM code with frequency annotations"));
  cl::alias PrintAnnotated2("A", cl::desc("Alias for --annotated-llvm"),
                            cl::aliasopt(PrintAnnotatedLLVM));
  cl::opt<bool>
  PrintAllCode("print-all-code",
               cl::desc("Print annotated code for the entire program"));
}

// PairSecondSort - A sorting predicate to sort by the second element of a pair.
template<class T>
struct PairSecondSortReverse
  : public std::binary_function<std::pair<T, double>,
                                std::pair<T, double>, bool> {
  bool operator()(const std::pair<T, double> &LHS,
                  const std::pair<T, double> &RHS) const {
    return LHS.second > RHS.second;
  }
};

static double ignoreMissing(double w) {
  if (w == ProfileInfo::MissingValue) return 0;
  return w;
}

namespace {
  class ProfileAnnotator : public AssemblyAnnotationWriter {
    ProfileInfo &PI;
  public:
    ProfileAnnotator(ProfileInfo& pi) : PI(pi) {}

    virtual void emitFunctionAnnot(const Function *F, raw_ostream &OS) {
      double w = PI.getExecutionCount(F);
      if (w != ProfileInfo::MissingValue) {
        OS << ";;; %" << F->getName() << " called "<<(unsigned)w
           <<" times.\n;;;\n";
      }
    }
    virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                          raw_ostream &OS) {
      double w = PI.getExecutionCount(BB);
      if (w != ProfileInfo::MissingValue) {
        if (w != 0) {
          OS << "\t;;; Basic block executed " << (unsigned)w << " times.\n";
        } else {
          OS << "\t;;; Never executed!\n";
        }
      }
    }

    virtual void emitBasicBlockEndAnnot(const BasicBlock *BB, raw_ostream &OS) {
      // Figure out how many times each successor executed.
      std::vector<std::pair<ProfileInfo::Edge, double> > SuccCounts;

      const TerminatorInst *TI = BB->getTerminator();
      for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s) {
        BasicBlock* Succ = TI->getSuccessor(s);
        double w = ignoreMissing(PI.getEdgeWeight(std::make_pair(BB, Succ)));
        if (w != 0)
          SuccCounts.push_back(std::make_pair(std::make_pair(BB, Succ), w));
      }
      if (!SuccCounts.empty()) {
        OS << "\t;;; Out-edge counts:";
        for (unsigned i = 0, e = SuccCounts.size(); i != e; ++i)
          OS << " [" << (SuccCounts[i]).second << " -> "
             << (SuccCounts[i]).first.second->getName() << "]";
        OS << "\n";
      }
    }
  };
}

namespace {
  /// ProfileInfoPrinterPass - Helper pass to dump the profile information for
  /// a module.
  //
  // FIXME: This should move elsewhere.
  class ProfileInfoPrinterPass : public ModulePass {
    ProfileInfoLoader &PIL;
  public:
    static char ID; // Class identification, replacement for typeinfo.
    explicit ProfileInfoPrinterPass(ProfileInfoLoader &_PIL) 
      : ModulePass(&ID), PIL(_PIL) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<ProfileInfo>();
    }

    bool runOnModule(Module &M);
  };
}

char ProfileInfoPrinterPass::ID = 0;

bool ProfileInfoPrinterPass::runOnModule(Module &M) {
  ProfileInfo &PI = getAnalysis<ProfileInfo>();
  std::map<const Function  *, unsigned> FuncFreqs;
  std::map<const BasicBlock*, unsigned> BlockFreqs;
  std::map<ProfileInfo::Edge, unsigned> EdgeFreqs;

  // Output a report. Eventually, there will be multiple reports selectable on
  // the command line, for now, just keep things simple.

  // Emit the most frequent function table...
  std::vector<std::pair<Function*, double> > FunctionCounts;
  std::vector<std::pair<BasicBlock*, double> > Counts;
  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
    if (FI->isDeclaration()) continue;
    double w = ignoreMissing(PI.getExecutionCount(FI));
    FunctionCounts.push_back(std::make_pair(FI, w));
    for (Function::iterator BB = FI->begin(), BBE = FI->end(); 
         BB != BBE; ++BB) {
      double w = ignoreMissing(PI.getExecutionCount(BB));
      Counts.push_back(std::make_pair(BB, w));
    }
  }

  // Sort by the frequency, backwards.
  sort(FunctionCounts.begin(), FunctionCounts.end(),
            PairSecondSortReverse<Function*>());

  double TotalExecutions = 0;
  for (unsigned i = 0, e = FunctionCounts.size(); i != e; ++i)
    TotalExecutions += FunctionCounts[i].second;

  std::cout << "===" << std::string(73, '-') << "===\n"
            << "LLVM profiling output for execution";
  if (PIL.getNumExecutions() != 1) std::cout << "s";
  std::cout << ":\n";

  for (unsigned i = 0, e = PIL.getNumExecutions(); i != e; ++i) {
    std::cout << "  ";
    if (e != 1) std::cout << i+1 << ". ";
    std::cout << PIL.getExecution(i) << "\n";
  }

  std::cout << "\n===" << std::string(73, '-') << "===\n";
  std::cout << "Function execution frequencies:\n\n";

  // Print out the function frequencies...
  std::cout << " ##   Frequency\n";
  for (unsigned i = 0, e = FunctionCounts.size(); i != e; ++i) {
    if (FunctionCounts[i].second == 0) {
      std::cout << "\n  NOTE: " << e-i << " function" <<
             (e-i-1 ? "s were" : " was") << " never executed!\n";
      break;
    }

    std::cout << std::setw(3) << i+1 << ". " 
      << std::setw(5) << FunctionCounts[i].second << "/"
      << TotalExecutions << " "
      << FunctionCounts[i].first->getNameStr() << "\n";
  }

  std::set<Function*> FunctionsToPrint;

  TotalExecutions = 0;
  for (unsigned i = 0, e = Counts.size(); i != e; ++i)
    TotalExecutions += Counts[i].second;
  
  // Sort by the frequency, backwards.
  sort(Counts.begin(), Counts.end(),
       PairSecondSortReverse<BasicBlock*>());
  
  std::cout << "\n===" << std::string(73, '-') << "===\n";
  std::cout << "Top 20 most frequently executed basic blocks:\n\n";
  
  // Print out the function frequencies...
  std::cout <<" ##      %% \tFrequency\n";
  unsigned BlocksToPrint = Counts.size();
  if (BlocksToPrint > 20) BlocksToPrint = 20;
  for (unsigned i = 0; i != BlocksToPrint; ++i) {
    if (Counts[i].second == 0) break;
    Function *F = Counts[i].first->getParent();
    std::cout << std::setw(3) << i+1 << ". " 
              << std::setw(5) << std::setprecision(3) 
              << Counts[i].second/(double)TotalExecutions*100 << "% "
              << std::setw(5) << Counts[i].second << "/"
              << TotalExecutions << "\t"
              << F->getNameStr() << "() - "
              << Counts[i].first->getNameStr() << "\n";
    FunctionsToPrint.insert(F);
  }

  if (PrintAnnotatedLLVM || PrintAllCode) {
    std::cout << "\n===" << std::string(73, '-') << "===\n";
    std::cout << "Annotated LLVM code for the module:\n\n";
  
    ProfileAnnotator PA(PI);

    if (FunctionsToPrint.empty() || PrintAllCode)
      M.print(outs(), &PA);
    else
      // Print just a subset of the functions.
      for (std::set<Function*>::iterator I = FunctionsToPrint.begin(),
             E = FunctionsToPrint.end(); I != E; ++I)
        (*I)->print(outs(), &PA);
  }

  return false;
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  try {
    cl::ParseCommandLineOptions(argc, argv, "llvm profile dump decoder\n");

    // Read in the bitcode file...
    std::string ErrorMessage;
    Module *M = 0;
    if (MemoryBuffer *Buffer = MemoryBuffer::getFileOrSTDIN(BitcodeFile,
                                                            &ErrorMessage)) {
      M = ParseBitcodeFile(Buffer, Context, &ErrorMessage);
      delete Buffer;
    }
    if (M == 0) {
      errs() << argv[0] << ": " << BitcodeFile << ": "
        << ErrorMessage << "\n";
      return 1;
    }

    // Read the profiling information. This is redundant since we load it again
    // using the standard profile info provider pass, but for now this gives us
    // access to additional information not exposed via the ProfileInfo
    // interface.
    ProfileInfoLoader PIL(argv[0], ProfileDataFile, *M);

    // Run the printer pass.
    PassManager PassMgr;
    PassMgr.add(createProfileLoaderPass(ProfileDataFile));
    PassMgr.add(new ProfileInfoPrinterPass(PIL));
    PassMgr.run(*M);

    return 0;
  } catch (const std::string& msg) {
    errs() << argv[0] << ": " << msg << "\n";
  } catch (...) {
    errs() << argv[0] << ": Unexpected unknown exception occurred.\n";
  }
  
  return 1;
}
