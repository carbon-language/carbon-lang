//===- LoopDependenceAnalysis.cpp - LDA Implementation ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the (beginning) of an implementation of a loop dependence analysis
// framework, which is used to detect dependences in memory accesses in loops.
//
// Please note that this is work in progress and the interface is subject to
// change.
//
// TODO: adapt as implementation progresses.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lda"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopDependenceAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Instructions.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;

LoopPass *llvm::createLoopDependenceAnalysisPass() {
  return new LoopDependenceAnalysis();
}

static RegisterPass<LoopDependenceAnalysis>
R("lda", "Loop Dependence Analysis", false, true);
char LoopDependenceAnalysis::ID = 0;

//===----------------------------------------------------------------------===//
//                             Utility Functions
//===----------------------------------------------------------------------===//

static inline bool IsMemRefInstr(const Value *V) {
  const Instruction *I = dyn_cast<const Instruction>(V);
  return I && (I->mayReadFromMemory() || I->mayWriteToMemory());
}

static void GetMemRefInstrs(const Loop *L,
                            SmallVectorImpl<Instruction*> &Memrefs) {
  for (Loop::block_iterator b = L->block_begin(), be = L->block_end();
      b != be; ++b)
    for (BasicBlock::iterator i = (*b)->begin(), ie = (*b)->end();
        i != ie; ++i)
      if (IsMemRefInstr(i))
        Memrefs.push_back(i);
}

static bool IsLoadOrStoreInst(Value *I) {
  return isa<LoadInst>(I) || isa<StoreInst>(I);
}

static Value *GetPointerOperand(Value *I) {
  if (LoadInst *i = dyn_cast<LoadInst>(I))
    return i->getPointerOperand();
  if (StoreInst *i = dyn_cast<StoreInst>(I))
    return i->getPointerOperand();
  llvm_unreachable("Value is no load or store instruction!");
  // Never reached.
  return 0;
}

//===----------------------------------------------------------------------===//
//                             Dependence Testing
//===----------------------------------------------------------------------===//

bool LoopDependenceAnalysis::isDependencePair(const Value *A,
                                              const Value *B) const {
  return IsMemRefInstr(A) &&
         IsMemRefInstr(B) &&
         (cast<const Instruction>(A)->mayWriteToMemory() ||
          cast<const Instruction>(B)->mayWriteToMemory());
}

bool LoopDependenceAnalysis::findOrInsertDependencePair(Value *X,
                                                        Value *Y,
                                                        DependencePair *&P) {
  void *insertPos = 0;
  FoldingSetNodeID id;
  id.AddPointer(X);
  id.AddPointer(Y);

  P = Pairs.FindNodeOrInsertPos(id, insertPos);
  if (P) return true;

  P = PairAllocator.Allocate<DependencePair>();
  new (P) DependencePair(id, X, Y);
  Pairs.InsertNode(P, insertPos);
  return false;
}

void LoopDependenceAnalysis::analysePair(DependencePair *P) const {
  DOUT << "Analysing:\n" << *P->A << "\n" << *P->B << "\n";

  // Our default answer: we don't know anything, i.e. we failed to analyse this
  // pair to get a more specific answer (dependent, independent).
  P->Result = Unknown;

  // We only analyse loads and stores but no possible memory accesses by e.g.
  // free, call, or invoke instructions.
  if (!IsLoadOrStoreInst(P->A) || !IsLoadOrStoreInst(P->B)) {
    DOUT << "--> [?] no load/store\n";
    return;
  }

  Value *aptr = GetPointerOperand(P->A);
  Value *bptr = GetPointerOperand(P->B);
  const Value *aobj = aptr->getUnderlyingObject();
  const Value *bobj = bptr->getUnderlyingObject();
  AliasAnalysis::AliasResult alias = AA->alias(
      aobj, AA->getTargetData().getTypeStoreSize(aobj->getType()),
      bobj, AA->getTargetData().getTypeStoreSize(bobj->getType()));

  // We can not analyse objects if we do not know about their aliasing.
  if (alias == AliasAnalysis::MayAlias) {
    DOUT << "---> [?] may alias\n";
    return;
  }

  // If the objects noalias, they are distinct, accesses are independent.
  if (alias == AliasAnalysis::NoAlias) {
    DOUT << "---> [I] no alias\n";
    P->Result = Independent;
    return;
  }

  // TODO: the underlying objects MustAlias, test for dependence

  DOUT << "---> [?] cannot analyse\n";
  return;
}

bool LoopDependenceAnalysis::depends(Value *A, Value *B) {
  assert(isDependencePair(A, B) && "Values form no dependence pair!");

  DependencePair *p;
  if (!findOrInsertDependencePair(A, B, p)) {
    // The pair is not cached, so analyse it.
    analysePair(p);
  }
  return p->Result != Independent;
}

//===----------------------------------------------------------------------===//
//                   LoopDependenceAnalysis Implementation
//===----------------------------------------------------------------------===//

bool LoopDependenceAnalysis::runOnLoop(Loop *L, LPPassManager &) {
  this->L = L;
  AA = &getAnalysis<AliasAnalysis>();
  SE = &getAnalysis<ScalarEvolution>();
  return false;
}

void LoopDependenceAnalysis::releaseMemory() {
  Pairs.clear();
  PairAllocator.Reset();
}

void LoopDependenceAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequiredTransitive<ScalarEvolution>();
}

static void PrintLoopInfo(raw_ostream &OS,
                          LoopDependenceAnalysis *LDA, const Loop *L) {
  if (!L->empty()) return; // ignore non-innermost loops

  SmallVector<Instruction*, 8> memrefs;
  GetMemRefInstrs(L, memrefs);

  OS << "Loop at depth " << L->getLoopDepth() << ", header block: ";
  WriteAsOperand(OS, L->getHeader(), false);
  OS << "\n";

  OS << "  Load/store instructions: " << memrefs.size() << "\n";
  for (SmallVector<Instruction*, 8>::const_iterator x = memrefs.begin(),
      end = memrefs.end(); x != end; ++x)
    OS << "\t" << (x - memrefs.begin()) << ": " << **x << "\n";

  OS << "  Pairwise dependence results:\n";
  for (SmallVector<Instruction*, 8>::const_iterator x = memrefs.begin(),
      end = memrefs.end(); x != end; ++x)
    for (SmallVector<Instruction*, 8>::const_iterator y = x + 1;
        y != end; ++y)
      if (LDA->isDependencePair(*x, *y))
        OS << "\t" << (x - memrefs.begin()) << "," << (y - memrefs.begin())
           << ": " << (LDA->depends(*x, *y) ? "dependent" : "independent")
           << "\n";
}

void LoopDependenceAnalysis::print(raw_ostream &OS, const Module*) const {
  // TODO: doc why const_cast is safe
  PrintLoopInfo(OS, const_cast<LoopDependenceAnalysis*>(this), this->L);
}

void LoopDependenceAnalysis::print(std::ostream &OS, const Module *M) const {
  raw_os_ostream os(OS);
  print(os, M);
}
