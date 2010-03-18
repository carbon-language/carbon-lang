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
// TODO: document lingo (pair, subscript, index)
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lda"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopDependenceAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Instructions.h"
#include "llvm/Operator.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;

STATISTIC(NumAnswered,    "Number of dependence queries answered");
STATISTIC(NumAnalysed,    "Number of distinct dependence pairs analysed");
STATISTIC(NumDependent,   "Number of pairs with dependent accesses");
STATISTIC(NumIndependent, "Number of pairs with independent accesses");
STATISTIC(NumUnknown,     "Number of pairs with unknown accesses");

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

static AliasAnalysis::AliasResult UnderlyingObjectsAlias(AliasAnalysis *AA,
                                                         const Value *A,
                                                         const Value *B) {
  const Value *aObj = A->getUnderlyingObject();
  const Value *bObj = B->getUnderlyingObject();
  return AA->alias(aObj, AA->getTypeStoreSize(aObj->getType()),
                   bObj, AA->getTypeStoreSize(bObj->getType()));
}

static inline const SCEV *GetZeroSCEV(ScalarEvolution *SE) {
  return SE->getConstant(Type::getInt32Ty(SE->getContext()), 0L);
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

bool LoopDependenceAnalysis::findOrInsertDependencePair(Value *A,
                                                        Value *B,
                                                        DependencePair *&P) {
  void *insertPos = 0;
  FoldingSetNodeID id;
  id.AddPointer(A);
  id.AddPointer(B);

  P = Pairs.FindNodeOrInsertPos(id, insertPos);
  if (P) return true;

  P = new (PairAllocator) DependencePair(id, A, B);
  Pairs.InsertNode(P, insertPos);
  return false;
}

void LoopDependenceAnalysis::getLoops(const SCEV *S,
                                      DenseSet<const Loop*>* Loops) const {
  // Refactor this into an SCEVVisitor, if efficiency becomes a concern.
  for (const Loop *L = this->L; L != 0; L = L->getParentLoop())
    if (!S->isLoopInvariant(L))
      Loops->insert(L);
}

bool LoopDependenceAnalysis::isLoopInvariant(const SCEV *S) const {
  DenseSet<const Loop*> loops;
  getLoops(S, &loops);
  return loops.empty();
}

bool LoopDependenceAnalysis::isAffine(const SCEV *S) const {
  const SCEVAddRecExpr *rec = dyn_cast<SCEVAddRecExpr>(S);
  return isLoopInvariant(S) || (rec && rec->isAffine());
}

bool LoopDependenceAnalysis::isZIVPair(const SCEV *A, const SCEV *B) const {
  return isLoopInvariant(A) && isLoopInvariant(B);
}

bool LoopDependenceAnalysis::isSIVPair(const SCEV *A, const SCEV *B) const {
  DenseSet<const Loop*> loops;
  getLoops(A, &loops);
  getLoops(B, &loops);
  return loops.size() == 1;
}

LoopDependenceAnalysis::DependenceResult
LoopDependenceAnalysis::analyseZIV(const SCEV *A,
                                   const SCEV *B,
                                   Subscript *S) const {
  assert(isZIVPair(A, B) && "Attempted to ZIV-test non-ZIV SCEVs!");
  return A == B ? Dependent : Independent;
}

LoopDependenceAnalysis::DependenceResult
LoopDependenceAnalysis::analyseSIV(const SCEV *A,
                                   const SCEV *B,
                                   Subscript *S) const {
  return Unknown; // TODO: Implement.
}

LoopDependenceAnalysis::DependenceResult
LoopDependenceAnalysis::analyseMIV(const SCEV *A,
                                   const SCEV *B,
                                   Subscript *S) const {
  return Unknown; // TODO: Implement.
}

LoopDependenceAnalysis::DependenceResult
LoopDependenceAnalysis::analyseSubscript(const SCEV *A,
                                         const SCEV *B,
                                         Subscript *S) const {
  DEBUG(dbgs() << "  Testing subscript: " << *A << ", " << *B << "\n");

  if (A == B) {
    DEBUG(dbgs() << "  -> [D] same SCEV\n");
    return Dependent;
  }

  if (!isAffine(A) || !isAffine(B)) {
    DEBUG(dbgs() << "  -> [?] not affine\n");
    return Unknown;
  }

  if (isZIVPair(A, B))
    return analyseZIV(A, B, S);

  if (isSIVPair(A, B))
    return analyseSIV(A, B, S);

  return analyseMIV(A, B, S);
}

LoopDependenceAnalysis::DependenceResult
LoopDependenceAnalysis::analysePair(DependencePair *P) const {
  DEBUG(dbgs() << "Analysing:\n" << *P->A << "\n" << *P->B << "\n");

  // We only analyse loads and stores but no possible memory accesses by e.g.
  // free, call, or invoke instructions.
  if (!IsLoadOrStoreInst(P->A) || !IsLoadOrStoreInst(P->B)) {
    DEBUG(dbgs() << "--> [?] no load/store\n");
    return Unknown;
  }

  Value *aPtr = GetPointerOperand(P->A);
  Value *bPtr = GetPointerOperand(P->B);

  switch (UnderlyingObjectsAlias(AA, aPtr, bPtr)) {
  case AliasAnalysis::MayAlias:
    // We can not analyse objects if we do not know about their aliasing.
    DEBUG(dbgs() << "---> [?] may alias\n");
    return Unknown;

  case AliasAnalysis::NoAlias:
    // If the objects noalias, they are distinct, accesses are independent.
    DEBUG(dbgs() << "---> [I] no alias\n");
    return Independent;

  case AliasAnalysis::MustAlias:
    break; // The underlying objects alias, test accesses for dependence.
  }

  const GEPOperator *aGEP = dyn_cast<GEPOperator>(aPtr);
  const GEPOperator *bGEP = dyn_cast<GEPOperator>(bPtr);

  if (!aGEP || !bGEP)
    return Unknown;

  // FIXME: Is filtering coupled subscripts necessary?

  // Collect GEP operand pairs (FIXME: use GetGEPOperands from BasicAA), adding
  // trailing zeroes to the smaller GEP, if needed.
  typedef SmallVector<std::pair<const SCEV*, const SCEV*>, 4> GEPOpdPairsTy;
  GEPOpdPairsTy opds;
  for(GEPOperator::const_op_iterator aIdx = aGEP->idx_begin(),
                                     aEnd = aGEP->idx_end(),
                                     bIdx = bGEP->idx_begin(),
                                     bEnd = bGEP->idx_end();
      aIdx != aEnd && bIdx != bEnd;
      aIdx += (aIdx != aEnd), bIdx += (bIdx != bEnd)) {
    const SCEV* aSCEV = (aIdx != aEnd) ? SE->getSCEV(*aIdx) : GetZeroSCEV(SE);
    const SCEV* bSCEV = (bIdx != bEnd) ? SE->getSCEV(*bIdx) : GetZeroSCEV(SE);
    opds.push_back(std::make_pair(aSCEV, bSCEV));
  }

  if (!opds.empty() && opds[0].first != opds[0].second) {
    // We cannot (yet) handle arbitrary GEP pointer offsets. By limiting
    //
    // TODO: this could be relaxed by adding the size of the underlying object
    // to the first subscript. If we have e.g. (GEP x,0,i; GEP x,2,-i) and we
    // know that x is a [100 x i8]*, we could modify the first subscript to be
    // (i, 200-i) instead of (i, -i).
    return Unknown;
  }

  // Now analyse the collected operand pairs (skipping the GEP ptr offsets).
  for (GEPOpdPairsTy::const_iterator i = opds.begin() + 1, end = opds.end();
       i != end; ++i) {
    Subscript subscript;
    DependenceResult result = analyseSubscript(i->first, i->second, &subscript);
    if (result != Dependent) {
      // We either proved independence or failed to analyse this subscript.
      // Further subscripts will not improve the situation, so abort early.
      return result;
    }
    P->Subscripts.push_back(subscript);
  }
  // We successfully analysed all subscripts but failed to prove independence.
  return Dependent;
}

bool LoopDependenceAnalysis::depends(Value *A, Value *B) {
  assert(isDependencePair(A, B) && "Values form no dependence pair!");
  ++NumAnswered;

  DependencePair *p;
  if (!findOrInsertDependencePair(A, B, p)) {
    // The pair is not cached, so analyse it.
    ++NumAnalysed;
    switch (p->Result = analysePair(p)) {
    case Dependent:   ++NumDependent;   break;
    case Independent: ++NumIndependent; break;
    case Unknown:     ++NumUnknown;     break;
    }
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
