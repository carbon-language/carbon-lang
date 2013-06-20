//===- SLPVectorizer.cpp - A bottom up SLP Vectorizer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass implements the Bottom Up SLP vectorizer. It detects consecutive
// stores that can be put together into vector-stores. Next, it attempts to
// construct vectorizable tree using the use-def chains. If a profitable tree
// was found, the SLP vectorizer performs vectorization on the tree.
//
// The pass is inspired by the work described in the paper:
//  "Loop-Aware SLP in GCC" by Ira Rosen, Dorit Nuzman, Ayal Zaks.
//
//===----------------------------------------------------------------------===//
#define SV_NAME "slp-vectorizer"
#define DEBUG_TYPE "SLP"

#include "VecUtils.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

using namespace llvm;

static cl::opt<int>
SLPCostThreshold("slp-threshold", cl::init(0), cl::Hidden,
                 cl::desc("Only vectorize trees if the gain is above this "
                          "number. (gain = -cost of vectorization)"));
namespace {

/// The SLPVectorizer Pass.
struct SLPVectorizer : public FunctionPass {
  typedef MapVector<Value*, BoUpSLP::StoreList> StoreListMap;

  /// Pass identification, replacement for typeid
  static char ID;

  explicit SLPVectorizer() : FunctionPass(ID) {
    initializeSLPVectorizerPass(*PassRegistry::getPassRegistry());
  }

  ScalarEvolution *SE;
  DataLayout *DL;
  TargetTransformInfo *TTI;
  AliasAnalysis *AA;
  LoopInfo *LI;

  virtual bool runOnFunction(Function &F) {
    SE = &getAnalysis<ScalarEvolution>();
    DL = getAnalysisIfAvailable<DataLayout>();
    TTI = &getAnalysis<TargetTransformInfo>();
    AA = &getAnalysis<AliasAnalysis>();
    LI = &getAnalysis<LoopInfo>();

    StoreRefs.clear();
    bool Changed = false;

    // Must have DataLayout. We can't require it because some tests run w/o
    // triple.
    if (!DL)
      return false;

    DEBUG(dbgs()<<"SLP: Analyzing blocks in " << F.getName() << ".\n");

    for (Function::iterator it = F.begin(), e = F.end(); it != e; ++it) {
      BasicBlock *BB = it;
      bool BBChanged = false;

      // Use the bollom up slp vectorizer to construct chains that start with
      // he store instructions.
      BoUpSLP R(BB, SE, DL, TTI, AA, LI->getLoopFor(BB));

      // Vectorize trees that end at reductions.
      BBChanged |= vectorizeChainsInBlock(BB, R);

      // Vectorize trees that end at stores.
      if (unsigned count = collectStores(BB, R)) {
        (void)count;
        DEBUG(dbgs()<<"SLP: Found " << count << " stores to vectorize.\n");
        BBChanged |= vectorizeStoreChains(R);
      }

      // Try to hoist some of the scalarization code to the preheader.
      if (BBChanged) {
        hoistGatherSequence(LI, BB, R);
        Changed |= vectorizeUsingGatherHints(R.getGatherSeqInstructions());
      }

      Changed |= BBChanged;
    }

    if (Changed) {
      DEBUG(dbgs()<<"SLP: vectorized \""<<F.getName()<<"\"\n");
      DEBUG(verifyFunction(F));
    }
    return Changed;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<AliasAnalysis>();
    AU.addRequired<TargetTransformInfo>();
    AU.addRequired<LoopInfo>();
  }

private:

  /// \brief Collect memory references and sort them according to their base
  /// object. We sort the stores to their base objects to reduce the cost of the
  /// quadratic search on the stores. TODO: We can further reduce this cost
  /// if we flush the chain creation every time we run into a memory barrier.
  unsigned collectStores(BasicBlock *BB, BoUpSLP &R);

  /// \brief Try to vectorize a chain that starts at two arithmetic instrs.
  bool tryToVectorizePair(Value *A, Value *B,  BoUpSLP &R);

  /// \brief Try to vectorize a list of operands. If \p NeedExtracts is true
  /// then we calculate the cost of extracting the scalars from the vector.
  /// \returns true if a value was vectorized.
  bool tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R, bool NeedExtracts);

  /// \brief Try to vectorize a chain that may start at the operands of \V;
  bool tryToVectorize(BinaryOperator *V,  BoUpSLP &R);

  /// \brief Vectorize the stores that were collected in StoreRefs.
  bool vectorizeStoreChains(BoUpSLP &R);

  /// \brief Try to hoist gather sequences outside of the loop in cases where
  /// all of the sources are loop invariant.
  void hoistGatherSequence(LoopInfo *LI, BasicBlock *BB, BoUpSLP &R);

  /// \brief Try to vectorize additional sequences in different basic blocks
  /// based on values that we gathered in previous blocks. The list \p Gathers
  /// holds the gather InsertElement instructions that were generated during
  /// vectorization.
  /// \returns True if some code was vectorized.
  bool vectorizeUsingGatherHints(BoUpSLP::InstrList &Gathers);

  /// \brief Scan the basic block and look for patterns that are likely to start
  /// a vectorization chain.
  bool vectorizeChainsInBlock(BasicBlock *BB, BoUpSLP &R);

private:
  StoreListMap StoreRefs;
};

unsigned SLPVectorizer::collectStores(BasicBlock *BB, BoUpSLP &R) {
  unsigned count = 0;
  StoreRefs.clear();
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    StoreInst *SI = dyn_cast<StoreInst>(it);
    if (!SI)
      continue;

    // Check that the pointer points to scalars.
    Type *Ty = SI->getValueOperand()->getType();
    if (Ty->isAggregateType() || Ty->isVectorTy())
      return 0;

    // Find the base of the GEP.
    Value *Ptr = SI->getPointerOperand();
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr))
      Ptr = GEP->getPointerOperand();

    // Save the store locations.
    StoreRefs[Ptr].push_back(SI);
    count++;
  }
  return count;
}

bool SLPVectorizer::tryToVectorizePair(Value *A, Value *B,  BoUpSLP &R) {
  if (!A || !B) return false;
  Value *VL[] = { A, B };
  return tryToVectorizeList(VL, R, true);
}

bool SLPVectorizer::tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R,
                                       bool NeedExtracts) {
  if (VL.size() < 2)
    return false;

  DEBUG(dbgs()<<"SLP: Vectorizing a list of length = " << VL.size() << ".\n");

  // Check that all of the parts are scalar instructions of the same type.
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0) return 0;

  unsigned Opcode0 = I0->getOpcode();

  for (int i = 0, e = VL.size(); i < e; ++i) {
    Type *Ty = VL[i]->getType();
    if (Ty->isAggregateType() || Ty->isVectorTy())
      return 0;
    Instruction *Inst = dyn_cast<Instruction>(VL[i]);
    if (!Inst || Inst->getOpcode() != Opcode0)
      return 0;
  }

  int Cost = R.getTreeCost(VL);
  int ExtrCost =  NeedExtracts ? R.getScalarizationCost(VL) : 0;
  DEBUG(dbgs()<<"SLP: Cost of pair:" << Cost <<
        " Cost of extract:" << ExtrCost << ".\n");
  if ((Cost+ExtrCost) >= -SLPCostThreshold) return false;
  DEBUG(dbgs()<<"SLP: Vectorizing pair.\n");
  R.vectorizeArith(VL);
  return true;
}

bool SLPVectorizer::tryToVectorize(BinaryOperator *V,  BoUpSLP &R) {
  if (!V) return false;
  // Try to vectorize V.
  if (tryToVectorizePair(V->getOperand(0), V->getOperand(1), R))
    return true;

  BinaryOperator *A = dyn_cast<BinaryOperator>(V->getOperand(0));
  BinaryOperator *B = dyn_cast<BinaryOperator>(V->getOperand(1));
  // Try to skip B.
  if (B && B->hasOneUse()) {
    BinaryOperator *B0 = dyn_cast<BinaryOperator>(B->getOperand(0));
    BinaryOperator *B1 = dyn_cast<BinaryOperator>(B->getOperand(1));
    if (tryToVectorizePair(A, B0, R)) {
      B->moveBefore(V);
      return true;
    }
    if (tryToVectorizePair(A, B1, R)) {
      B->moveBefore(V);
      return true;
    }
  }

  // Try to skip A.
  if (A && A->hasOneUse()) {
    BinaryOperator *A0 = dyn_cast<BinaryOperator>(A->getOperand(0));
    BinaryOperator *A1 = dyn_cast<BinaryOperator>(A->getOperand(1));
    if (tryToVectorizePair(A0, B, R)) {
      A->moveBefore(V);
      return true;
    }
    if (tryToVectorizePair(A1, B, R)) {
      A->moveBefore(V);
      return true;
    }
  }
  return 0;
}

bool SLPVectorizer::vectorizeChainsInBlock(BasicBlock *BB, BoUpSLP &R) {
  bool Changed = false;
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    if (isa<DbgInfoIntrinsic>(it)) continue;

    // Try to vectorize reductions that use PHINodes.
    if (PHINode *P = dyn_cast<PHINode>(it)) {
      // Check that the PHI is a reduction PHI.
      if (P->getNumIncomingValues() != 2) return Changed;
      Value *Rdx = (P->getIncomingBlock(0) == BB ? P->getIncomingValue(0) :
                    (P->getIncomingBlock(1) == BB ? P->getIncomingValue(1) :
                     0));
      // Check if this is a Binary Operator.
      BinaryOperator *BI = dyn_cast_or_null<BinaryOperator>(Rdx);
      if (!BI)
        continue;

      Value *Inst = BI->getOperand(0);
      if (Inst == P) Inst = BI->getOperand(1);
      Changed |= tryToVectorize(dyn_cast<BinaryOperator>(Inst), R);
      continue;
    }

    // Try to vectorize trees that start at compare instructions.
    if (CmpInst *CI = dyn_cast<CmpInst>(it)) {
      if (tryToVectorizePair(CI->getOperand(0), CI->getOperand(1), R)) {
        Changed |= true;
        continue;
      }
      for (int i = 0; i < 2; ++i)
        if (BinaryOperator *BI = dyn_cast<BinaryOperator>(CI->getOperand(i)))
          Changed |= tryToVectorizePair(BI->getOperand(0), BI->getOperand(1), R);
      continue;
    }
  }

  // Scan the PHINodes in our successors in search for pairing hints.
  for (succ_iterator it = succ_begin(BB), e = succ_end(BB); it != e; ++it) {
    BasicBlock *Succ = *it;
    SmallVector<Value*, 4> Incoming;

    // Collect the incoming values from the PHIs.
    for (BasicBlock::iterator instr = Succ->begin(), ie = Succ->end();
         instr != ie; ++instr) {
      PHINode *P = dyn_cast<PHINode>(instr);

      if (!P)
        break;

      Value *V = P->getIncomingValueForBlock(BB);
      if (Instruction *I = dyn_cast<Instruction>(V))
        if (I->getParent() == BB)
          Incoming.push_back(I);
    }

    if (Incoming.size() > 1)
      Changed |= tryToVectorizeList(Incoming, R, true);
  }
  
  return Changed;
}

bool SLPVectorizer::vectorizeStoreChains(BoUpSLP &R) {
  bool Changed = false;
  // Attempt to sort and vectorize each of the store-groups.
  for (StoreListMap::iterator it = StoreRefs.begin(), e = StoreRefs.end();
       it != e; ++it) {
    if (it->second.size() < 2)
      continue;

    DEBUG(dbgs()<<"SLP: Analyzing a store chain of length " <<
          it->second.size() << ".\n");

    Changed |= R.vectorizeStores(it->second, -SLPCostThreshold);
  }
  return Changed;
}

bool SLPVectorizer::vectorizeUsingGatherHints(BoUpSLP::InstrList &Gathers) {
  SmallVector<Value*, 4> Seq;
  bool Changed = false;
  for (int i = 0, e = Gathers.size(); i < e; ++i) {
    InsertElementInst *IEI = dyn_cast_or_null<InsertElementInst>(Gathers[i]);

    if (IEI) {
      if (Instruction *I = dyn_cast<Instruction>(IEI->getOperand(1)))
        Seq.push_back(I);
    } else {

      if (!Seq.size())
        continue;

      Instruction *I = cast<Instruction>(Seq[0]);
      BasicBlock *BB = I->getParent();

      DEBUG(dbgs()<<"SLP: Inspecting a gather list of size " << Seq.size() <<
            " in " << BB->getName() << ".\n");

      // Check if the gathered values have multiple uses. If they only have one
      // user then we know that the insert/extract pair will go away.
      bool HasMultipleUsers = false;
      for (int i=0; e = Seq.size(), i < e; ++i) {
        if (!Seq[i]->hasOneUse()) {
          HasMultipleUsers = true;
          break;
        }
      }

      BoUpSLP BO(BB, SE, DL, TTI, AA, LI->getLoopFor(BB));

      if (tryToVectorizeList(Seq, BO, HasMultipleUsers)) {
        DEBUG(dbgs()<<"SLP: Vectorized a gather list of len " << Seq.size() <<
              " in " << BB->getName() << ".\n");
        Changed = true;
      }

      Seq.clear();
    }
  }

  return Changed;
}

void SLPVectorizer::hoistGatherSequence(LoopInfo *LI, BasicBlock *BB,
                                        BoUpSLP &R) {
  // Check if this block is inside a loop.
  Loop *L = LI->getLoopFor(BB);
  if (!L)
    return;

  // Check if it has a preheader.
  BasicBlock *PreHeader = L->getLoopPreheader();
  if (!PreHeader)
    return;

  // Mark the insertion point for the block.
  Instruction *Location = PreHeader->getTerminator();

  BoUpSLP::InstrList &Gathers = R.getGatherSeqInstructions();
  for (BoUpSLP::InstrList::iterator it = Gathers.begin(), e = Gathers.end();
       it != e; ++it) {
    InsertElementInst *Insert = dyn_cast_or_null<InsertElementInst>(*it);

    // The InsertElement sequence can be simplified into a constant.
    // Also Ignore NULL pointers because they are only here to separate
    // sequences.
    if (!Insert)
      continue;

    // If the vector or the element that we insert into it are
    // instructions that are defined in this basic block then we can't
    // hoist this instruction.
    Instruction *CurrVec = dyn_cast<Instruction>(Insert->getOperand(0));
    Instruction *NewElem = dyn_cast<Instruction>(Insert->getOperand(1));
    if (CurrVec && L->contains(CurrVec)) continue;
    if (NewElem && L->contains(NewElem)) continue;

    // We can hoist this instruction. Move it to the pre-header.
    Insert->moveBefore(Location);
  }
}

} // end anonymous namespace

char SLPVectorizer::ID = 0;
static const char lv_name[] = "SLP Vectorizer";
INITIALIZE_PASS_BEGIN(SLPVectorizer, SV_NAME, lv_name, false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_AG_DEPENDENCY(TargetTransformInfo)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(SLPVectorizer, SV_NAME, lv_name, false, false)

namespace llvm {
  Pass *createSLPVectorizerPass() {
    return new SLPVectorizer();
  }
}

