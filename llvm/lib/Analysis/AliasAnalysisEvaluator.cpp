//===- AliasAnalysisEvaluator.cpp - Alias Analysis Accuracy Evaluator -----===//
//
// This file implements a simple N^2 alias analysis accuracy evaluator.
// Basically, for each function in the program, it simply queries to see how the
// alias analysis implementation answers alias queries between each pair of
// pointers in the function.
//
// This is inspired and adapted from code by: Naveen Neelakantam, Francesco
// Spadini, and Wojciech Stryjewski.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Assembly/Writer.h"
#include "Support/CommandLine.h"

namespace {
  cl::opt<bool> PrintNo  ("print-no-aliases", cl::ReallyHidden);
  cl::opt<bool> PrintMay ("print-may-aliases", cl::ReallyHidden);
  cl::opt<bool> PrintMust("print-must-aliases", cl::ReallyHidden);

  class AAEval : public FunctionPass {
    unsigned No, May, Must;

  public:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<AliasAnalysis>();
      AU.setPreservesAll();
    }
    
    bool doInitialization(Module &M) { No = May = Must = 0; return false; }
    bool runOnFunction(Function &F);
    bool doFinalization(Module &M);
  };

  RegisterOpt<AAEval>
  X("aa-eval", "Exhaustive Alias Analysis Precision Evaluator");
}

static inline void PrintResults(const char *Msg, bool P, Value *V1, Value *V2) {
  if (P) {
    std::cerr << "  " << Msg << ":\t";
    WriteAsOperand(std::cerr, V1) << ", ";
    WriteAsOperand(std::cerr, V2) << "\n";
  }
}

bool AAEval::runOnFunction(Function &F) {
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  
  std::vector<Value *> Pointers;

  for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
    if (isa<PointerType>(I->getType()))    // Add all pointer arguments
      Pointers.push_back(I);

  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
    if (isa<PointerType>((*I)->getType())) // Add all pointer instructions
      Pointers.push_back(*I);

  if (PrintNo || PrintMay || PrintMust)
    std::cerr << "Function: " << F.getName() << "\n";

  // iterate over the worklist, and run the full (n^2)/2 disambiguations
  for (std::vector<Value *>::iterator I1 = Pointers.begin(), E = Pointers.end();
       I1 != E; ++I1)
    for (std::vector<Value *>::iterator I2 = Pointers.begin(); I2 != I1; ++I2)
      switch (AA.alias(*I1, 0, *I2, 0)) {
      case AliasAnalysis::NoAlias:
        PrintResults("No", PrintNo, *I1, *I2);
        ++No; break;
      case AliasAnalysis::MayAlias:
        PrintResults("May", PrintMay, *I1, *I2);
        ++May; break;
      case AliasAnalysis::MustAlias:
        PrintResults("Must", PrintMust, *I1, *I2);
        ++Must; break;
      default:
        std::cerr << "Unknown alias query result!\n";
      }

  return false;
}

bool AAEval::doFinalization(Module &M) {
  unsigned Sum = No+May+Must;
  std::cerr << "===== Alias Analysis Evaluator Report =====\n";
  if (Sum == 0) {
    std::cerr << "  Alias Analysis Evaluator Summary: No pointers!\n";
    return false;
  }

  std::cerr << "  " << Sum << " Total Alias Queries Performed\n";
  std::cerr << "  " << No << " no alias responses (" << No*100/Sum << "%)\n";
  std::cerr << "  " << May << " may alias responses (" << May*100/Sum << "%)\n";
  std::cerr << "  " << Must << " must alias responses (" <<Must*100/Sum<<"%)\n";
  std::cerr << "  Alias Analysis Evaluator Summary: " << No*100/Sum << "%/"
            << May*100/Sum << "%/" << Must*100/Sum<<"%\n";
  return false;
}
