//===---- Delinearization.cpp - MultiDimensional Index Delinearization ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements an analysis pass that tries to delinearize all GEP
// instructions in all loops using the SCEV analysis functionality. This pass is
// only used for testing purposes: if your pass needs delinearization, please
// use the on-demand SCEVAddRecExpr::delinearize() function.
//
//===----------------------------------------------------------------------===//

#define DL_NAME "delinearize"
#define DEBUG_TYPE DL_NAME
#include "llvm/IR/Constants.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

class Delinearization : public FunctionPass {
  Delinearization(const Delinearization &); // do not implement
protected:
  Function *F;
  LoopInfo *LI;
  ScalarEvolution *SE;

public:
  static char ID; // Pass identification, replacement for typeid

  Delinearization() : FunctionPass(ID) {
    initializeDelinearizationPass(*PassRegistry::getPassRegistry());
  }
  virtual bool runOnFunction(Function &F);
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void print(raw_ostream &O, const Module *M = 0) const;
};

} // end anonymous namespace

void Delinearization::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<LoopInfo>();
  AU.addRequired<ScalarEvolution>();
}

bool Delinearization::runOnFunction(Function &F) {
  this->F = &F;
  SE = &getAnalysis<ScalarEvolution>();
  LI = &getAnalysis<LoopInfo>();
  return false;
}

static Value *getPointerOperand(Instruction &Inst) {
  if (LoadInst *Load = dyn_cast<LoadInst>(&Inst))
    return Load->getPointerOperand();
  else if (StoreInst *Store = dyn_cast<StoreInst>(&Inst))
    return Store->getPointerOperand();
  else if (GetElementPtrInst *Gep = dyn_cast<GetElementPtrInst>(&Inst))
    return Gep->getPointerOperand();
  return NULL;
}

void Delinearization::print(raw_ostream &O, const Module *) const {
  O << "Delinearization on function " << F->getName() << ":\n";
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    Instruction *Inst = &(*I);

    // Only analyze loads and stores.
    if (!isa<StoreInst>(Inst) && !isa<LoadInst>(Inst) &&
        !isa<GetElementPtrInst>(Inst))
      continue;

    const BasicBlock *BB = Inst->getParent();
    // Delinearize the memory access as analyzed in all the surrounding loops.
    // Do not analyze memory accesses outside loops.
    for (Loop *L = LI->getLoopFor(BB); L != NULL; L = L->getParentLoop()) {
      const SCEV *AccessFn = SE->getSCEVAtScope(getPointerOperand(*Inst), L);
      const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(AccessFn);

      // Do not try to delinearize memory accesses that are not AddRecs.
      if (!AR)
        break;

      O << "AddRec: " << *AR << "\n";

      SmallVector<const SCEV *, 3> Subscripts, Sizes;
      const SCEV *Res = AR->delinearize(*SE, Subscripts, Sizes);
      int Size = Subscripts.size();
      if (Res == AR || Size == 0) {
        O << "failed to delinearize\n";
        continue;
      }
      O << "Base offset: " << *Res << "\n";
      O << "ArrayDecl[UnknownSize]";
      for (int i = 0; i < Size - 1; i++)
        O << "[" << *Sizes[i] << "]";
      O << " with elements of " << *Sizes[Size - 1] << " bytes.\n";

      O << "ArrayRef";
      for (int i = 0; i < Size; i++)
        O << "[" << *Subscripts[i] << "]";
      O << "\n";
    }
  }
}

char Delinearization::ID = 0;
static const char delinearization_name[] = "Delinearization";
INITIALIZE_PASS_BEGIN(Delinearization, DL_NAME, delinearization_name, true,
                      true)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_END(Delinearization, DL_NAME, delinearization_name, true, true)

FunctionPass *llvm::createDelinearizationPass() { return new Delinearization; }
