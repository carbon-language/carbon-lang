//===- AliasAnalysisEvaluator.cpp - Alias Analysis Accuracy Evaluator -----===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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

#include "llvm/Function.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/InstIterator.h"
#include "Support/CommandLine.h"
#include <set>
using namespace llvm;

namespace {
  cl::opt<bool> PrintNoAlias("print-no-aliases", cl::ReallyHidden);
  cl::opt<bool> PrintMayAlias("print-may-aliases", cl::ReallyHidden);
  cl::opt<bool> PrintMustAlias("print-must-aliases", cl::ReallyHidden);

  cl::opt<bool> PrintNoModRef("print-no-modref", cl::ReallyHidden);
  cl::opt<bool> PrintMod("print-mod", cl::ReallyHidden);
  cl::opt<bool> PrintRef("print-ref", cl::ReallyHidden);
  cl::opt<bool> PrintModRef("print-modref", cl::ReallyHidden);

  class AAEval : public FunctionPass {
    unsigned NoAlias, MayAlias, MustAlias;
    unsigned NoModRef, Mod, Ref, ModRef;

  public:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<AliasAnalysis>();
      AU.setPreservesAll();
    }
    
    bool doInitialization(Module &M) { 
      NoAlias = MayAlias = MustAlias = 0; 
      NoModRef = Mod = Ref = ModRef = 0;
      return false; 
    }

    bool runOnFunction(Function &F);
    bool doFinalization(Module &M);
  };

  RegisterOpt<AAEval>
  X("aa-eval", "Exhaustive Alias Analysis Precision Evaluator");
}

static inline void PrintResults(const char *Msg, bool P, Value *V1, Value *V2,
                                Module *M) {
  if (P) {
    std::cerr << "  " << Msg << ":\t";
    WriteAsOperand(std::cerr, V1, true, true, M) << ", ";
    WriteAsOperand(std::cerr, V2, true, true, M) << "\n";
  }
}

bool AAEval::runOnFunction(Function &F) {
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  
  std::set<Value *> Pointers;
  std::set<CallSite> CallSites;

  for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
    if (isa<PointerType>(I->getType()))    // Add all pointer arguments
      Pointers.insert(I);

  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    if (isa<PointerType>((*I)->getType())) // Add all pointer instructions
      Pointers.insert(*I);
    for (User::op_iterator OI = (*I)->op_begin(); OI != (*I)->op_end(); ++OI)
      if (isa<PointerType>((*OI)->getType()))
        Pointers.insert(*OI);
  }

  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    CallSite CS = CallSite::get(*I);
    if (CS.getInstruction()) CallSites.insert(CS);
  }

  if (PrintNoAlias || PrintMayAlias || PrintMustAlias ||
      PrintNoModRef || PrintMod || PrintRef || PrintModRef)
    std::cerr << "Function: " << F.getName() << "\n";

  // iterate over the worklist, and run the full (n^2)/2 disambiguations
  for (std::set<Value *>::iterator I1 = Pointers.begin(), E = Pointers.end();
       I1 != E; ++I1)
    for (std::set<Value *>::iterator I2 = Pointers.begin(); I2 != I1; ++I2)
      switch (AA.alias(*I1, 0, *I2, 0)) {
      case AliasAnalysis::NoAlias:
        PrintResults("NoAlias", PrintNoAlias, *I1, *I2, F.getParent());
        ++NoAlias; break;
      case AliasAnalysis::MayAlias:
        PrintResults("MayAlias", PrintMayAlias, *I1, *I2, F.getParent());
        ++MayAlias; break;
      case AliasAnalysis::MustAlias:
        PrintResults("MustAlias", PrintMustAlias, *I1, *I2, F.getParent());
        ++MustAlias; break;
      default:
        std::cerr << "Unknown alias query result!\n";
      }

  // Mod/ref alias analysis: compare all pairs of calls and values
  for (std::set<Value *>::iterator V = Pointers.begin(), Ve = Pointers.end();
       V != Ve; ++V)
    for (std::set<CallSite>::iterator C = CallSites.begin(), 
           Ce = CallSites.end(); C != Ce; ++C) {
      Instruction *I = C->getInstruction();
      switch (AA.getModRefInfo(*C, *V, (*V)->getType()->getPrimitiveSize())) {
      case AliasAnalysis::NoModRef:
        PrintResults("NoModRef", PrintNoModRef, I, *V, F.getParent());
        ++NoModRef; break;
      case AliasAnalysis::Mod:
        PrintResults("Mod", PrintMod, I, *V, F.getParent());
        ++Mod; break;
      case AliasAnalysis::Ref:
        PrintResults("Ref", PrintRef, I, *V, F.getParent());
        ++Ref; break;
      case AliasAnalysis::ModRef:
        PrintResults("ModRef", PrintModRef, I, *V, F.getParent());
        ++ModRef; break;
      default:
        std::cerr << "Unknown alias query result!\n";
      }
    }

  return false;
}

bool AAEval::doFinalization(Module &M) {
  unsigned AliasSum = NoAlias + MayAlias + MustAlias;
  std::cerr << "===== Alias Analysis Evaluator Report =====\n";
  if (AliasSum == 0) {
    std::cerr << "  Alias Analysis Evaluator Summary: No pointers!\n";
  } else { 
    std::cerr << "  " << AliasSum << " Total Alias Queries Performed\n";
    std::cerr << "  " << NoAlias << " no alias responses (" 
              << NoAlias*100/AliasSum << "%)\n";
    std::cerr << "  " << MayAlias << " may alias responses (" 
              << MayAlias*100/AliasSum << "%)\n";
    std::cerr << "  " << MustAlias << " must alias responses (" 
              << MustAlias*100/AliasSum <<"%)\n";
    std::cerr << "  Alias Analysis Evaluator Pointer Alias Summary: " 
              << NoAlias*100/AliasSum  << "%/" << MayAlias*100/AliasSum << "%/" 
              << MustAlias*100/AliasSum << "%\n";
  }

  // Display the summary for mod/ref analysis
  unsigned ModRefSum = NoModRef + Mod + Ref + ModRef;
  if (ModRefSum == 0) {
    std::cerr << "  Alias Analysis Mod/Ref Evaluator Summary: no mod/ref!\n";
  } else {
    std::cerr << "  " << ModRefSum << " Total ModRef Queries Performed\n";
    std::cerr << "  " << NoModRef << " no mod/ref responses (" 
              << NoModRef*100/ModRefSum << "%)\n";
    std::cerr << "  " << Mod << " mod responses (" 
              << Mod*100/ModRefSum << "%)\n";
    std::cerr << "  " << Ref << " ref responses (" 
              << Ref*100/ModRefSum <<"%)\n";
    std::cerr << "  " << ModRef << " mod & ref responses (" 
              << ModRef*100/ModRefSum <<"%)\n";
    std::cerr << "  Alias Analysis Evaluator Mod/Ref Summary: " 
              << NoModRef*100/ModRefSum  << "%/" << Mod*100/ModRefSum << "%/" 
              << Ref*100/ModRefSum << "%/" << ModRef*100/ModRefSum << "%\n";
  }

  return false;
}
