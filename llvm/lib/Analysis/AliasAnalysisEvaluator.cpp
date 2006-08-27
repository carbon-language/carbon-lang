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

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>
#include <set>

using namespace llvm;

namespace {
  cl::opt<bool> PrintAll("print-all-alias-modref-info", cl::ReallyHidden);

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

      if (PrintAll) {
        PrintNoAlias = PrintMayAlias = PrintMustAlias = true;
        PrintNoModRef = PrintMod = PrintRef = PrintModRef = true;
      }
      return false;
    }

    bool runOnFunction(Function &F);
    bool doFinalization(Module &M);
  };

  RegisterPass<AAEval>
  X("aa-eval", "Exhaustive Alias Analysis Precision Evaluator");
}

FunctionPass *llvm::createAAEvalPass() { return new AAEval(); }

static inline void PrintResults(const char *Msg, bool P, Value *V1, Value *V2,
                                Module *M) {
  if (P) {
    std::cerr << "  " << Msg << ":\t";
    WriteAsOperand(std::cerr, V1, true, true, M) << ", ";
    WriteAsOperand(std::cerr, V2, true, true, M) << "\n";
  }
}

static inline void
PrintModRefResults(const char *Msg, bool P, Instruction *I, Value *Ptr,
                   Module *M) {
  if (P) {
    std::cerr << "  " << Msg << ":  Ptr: ";
    WriteAsOperand(std::cerr, Ptr, true, true, M);
    std::cerr << "\t<->" << *I;
  }
}

bool AAEval::runOnFunction(Function &F) {
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();

  const TargetData &TD = AA.getTargetData();

  std::set<Value *> Pointers;
  std::set<CallSite> CallSites;

  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I)
    if (isa<PointerType>(I->getType()))    // Add all pointer arguments
      Pointers.insert(I);

  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    if (isa<PointerType>(I->getType())) // Add all pointer instructions
      Pointers.insert(&*I);
    Instruction &Inst = *I;
    User::op_iterator OI = Inst.op_begin();
    if ((isa<InvokeInst>(Inst) || isa<CallInst>(Inst)) &&
        isa<Function>(Inst.getOperand(0)))
      ++OI;  // Skip actual functions for direct function calls.
    for (; OI != Inst.op_end(); ++OI)
      if (isa<PointerType>((*OI)->getType()) && !isa<ConstantPointerNull>(*OI))
        Pointers.insert(*OI);

    CallSite CS = CallSite::get(&*I);
    if (CS.getInstruction()) CallSites.insert(CS);
  }

  if (PrintNoAlias || PrintMayAlias || PrintMustAlias ||
      PrintNoModRef || PrintMod || PrintRef || PrintModRef)
    std::cerr << "Function: " << F.getName() << ": " << Pointers.size()
              << " pointers, " << CallSites.size() << " call sites\n";

  // iterate over the worklist, and run the full (n^2)/2 disambiguations
  for (std::set<Value *>::iterator I1 = Pointers.begin(), E = Pointers.end();
       I1 != E; ++I1) {
    unsigned I1Size = 0;
    const Type *I1ElTy = cast<PointerType>((*I1)->getType())->getElementType();
    if (I1ElTy->isSized()) I1Size = TD.getTypeSize(I1ElTy);

    for (std::set<Value *>::iterator I2 = Pointers.begin(); I2 != I1; ++I2) {
      unsigned I2Size = 0;
      const Type *I2ElTy =cast<PointerType>((*I2)->getType())->getElementType();
      if (I2ElTy->isSized()) I2Size = TD.getTypeSize(I2ElTy);

      switch (AA.alias(*I1, I1Size, *I2, I2Size)) {
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
    }
  }

  // Mod/ref alias analysis: compare all pairs of calls and values
  for (std::set<CallSite>::iterator C = CallSites.begin(),
         Ce = CallSites.end(); C != Ce; ++C) {
    Instruction *I = C->getInstruction();

    for (std::set<Value *>::iterator V = Pointers.begin(), Ve = Pointers.end();
         V != Ve; ++V) {
      unsigned Size = 0;
      const Type *ElTy = cast<PointerType>((*V)->getType())->getElementType();
      if (ElTy->isSized()) Size = TD.getTypeSize(ElTy);

      switch (AA.getModRefInfo(*C, *V, Size)) {
      case AliasAnalysis::NoModRef:
        PrintModRefResults("NoModRef", PrintNoModRef, I, *V, F.getParent());
        ++NoModRef; break;
      case AliasAnalysis::Mod:
        PrintModRefResults("     Mod", PrintMod, I, *V, F.getParent());
        ++Mod; break;
      case AliasAnalysis::Ref:
        PrintModRefResults("     Ref", PrintRef, I, *V, F.getParent());
        ++Ref; break;
      case AliasAnalysis::ModRef:
        PrintModRefResults("  ModRef", PrintModRef, I, *V, F.getParent());
        ++ModRef; break;
      default:
        std::cerr << "Unknown alias query result!\n";
      }
    }
  }

  return false;
}

static void PrintPercent(unsigned Num, unsigned Sum) {
  std::cerr << "(" << Num*100ULL/Sum << "."
            << ((Num*1000ULL/Sum) % 10) << "%)\n";
}

bool AAEval::doFinalization(Module &M) {
  unsigned AliasSum = NoAlias + MayAlias + MustAlias;
  std::cerr << "===== Alias Analysis Evaluator Report =====\n";
  if (AliasSum == 0) {
    std::cerr << "  Alias Analysis Evaluator Summary: No pointers!\n";
  } else {
    std::cerr << "  " << AliasSum << " Total Alias Queries Performed\n";
    std::cerr << "  " << NoAlias << " no alias responses ";
    PrintPercent(NoAlias, AliasSum);
    std::cerr << "  " << MayAlias << " may alias responses ";
    PrintPercent(MayAlias, AliasSum);
    std::cerr << "  " << MustAlias << " must alias responses ";
    PrintPercent(MustAlias, AliasSum);
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
    std::cerr << "  " << NoModRef << " no mod/ref responses ";
    PrintPercent(NoModRef, ModRefSum);
    std::cerr << "  " << Mod << " mod responses ";
    PrintPercent(Mod, ModRefSum);
    std::cerr << "  " << Ref << " ref responses ";
    PrintPercent(Ref, ModRefSum);
    std::cerr << "  " << ModRef << " mod & ref responses ";
    PrintPercent(ModRef, ModRefSum);
    std::cerr << "  Alias Analysis Evaluator Mod/Ref Summary: "
              << NoModRef*100/ModRefSum  << "%/" << Mod*100/ModRefSum << "%/"
              << Ref*100/ModRefSum << "%/" << ModRef*100/ModRefSum << "%\n";
  }

  return false;
}
