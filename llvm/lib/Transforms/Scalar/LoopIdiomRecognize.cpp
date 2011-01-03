//===-- LoopIdiomRecognize.cpp - Loop idiom recognition -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements an idiom recognizer that transforms simple loops into a
// non-loop form.  In cases that this kicks in, it can be a significant
// performance win.
//
//===----------------------------------------------------------------------===//
//
// TODO List:
//
// Future loop memory idioms to recognize:
//   memcmp, memmove, strlen, etc.
// Future floating point idioms to recognize in -ffast-math mode:
//   fpowi
// Future integer operation idioms to recognize:
//   ctpop, ctlz, cttz
//
// Beware that isel's default lowering for ctpop is highly inefficient for
// i64 and larger types when i64 is legal and the value has few bits set.  It
// would be good to enhance isel to emit a loop for ctpop in this case.
//
// We should enhance the memset/memcpy recognition to handle multiple stores in
// the loop.  This would handle things like:
//   void foo(_Complex float *P)
//     for (i) { __real__(*P) = 0;  __imag__(*P) = 0; }
// this is also "Example 2" from http://blog.regehr.org/archives/320
//
// This could recognize common matrix multiplies and dot product idioms and
// replace them with calls to BLAS (if linked in??).
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-idiom"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumMemSet, "Number of memset's formed from loop stores");
STATISTIC(NumMemCpy, "Number of memcpy's formed from loop load+stores");

namespace {
  class LoopIdiomRecognize : public LoopPass {
    Loop *CurLoop;
    const TargetData *TD;
    DominatorTree *DT;
    ScalarEvolution *SE;
  public:
    static char ID;
    explicit LoopIdiomRecognize() : LoopPass(ID) {
      initializeLoopIdiomRecognizePass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM);
    bool runOnLoopBlock(BasicBlock *BB, const SCEV *BECount,
                        SmallVectorImpl<BasicBlock*> &ExitBlocks);

    bool processLoopStore(StoreInst *SI, const SCEV *BECount);
    
    bool processLoopStoreOfSplatValue(StoreInst *SI, unsigned StoreSize,
                                      Value *SplatValue,
                                      const SCEVAddRecExpr *Ev,
                                      const SCEV *BECount);
    bool processLoopStoreOfLoopLoad(StoreInst *SI, unsigned StoreSize,
                                    const SCEVAddRecExpr *StoreEv,
                                    const SCEVAddRecExpr *LoadEv,
                                    const SCEV *BECount);
      
    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<AliasAnalysis>();
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<ScalarEvolution>();
      AU.addPreserved<DominatorTree>();
      AU.addRequired<DominatorTree>();
    }
  };
}

char LoopIdiomRecognize::ID = 0;
INITIALIZE_PASS_BEGIN(LoopIdiomRecognize, "loop-idiom", "Recognize loop idioms",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(LoopIdiomRecognize, "loop-idiom", "Recognize loop idioms",
                    false, false)

Pass *llvm::createLoopIdiomPass() { return new LoopIdiomRecognize(); }

/// DeleteDeadInstruction - Delete this instruction.  Before we do, go through
/// and zero out all the operands of this instruction.  If any of them become
/// dead, delete them and the computation tree that feeds them.
///
static void DeleteDeadInstruction(Instruction *I, ScalarEvolution &SE) {
  SmallVector<Instruction*, 32> NowDeadInsts;
  
  NowDeadInsts.push_back(I);
  
  // Before we touch this instruction, remove it from SE!
  do {
    Instruction *DeadInst = NowDeadInsts.pop_back_val();
    
    // This instruction is dead, zap it, in stages.  Start by removing it from
    // SCEV.
    SE.forgetValue(DeadInst);
    
    for (unsigned op = 0, e = DeadInst->getNumOperands(); op != e; ++op) {
      Value *Op = DeadInst->getOperand(op);
      DeadInst->setOperand(op, 0);
      
      // If this operand just became dead, add it to the NowDeadInsts list.
      if (!Op->use_empty()) continue;
      
      if (Instruction *OpI = dyn_cast<Instruction>(Op))
        if (isInstructionTriviallyDead(OpI))
          NowDeadInsts.push_back(OpI);
    }
    
    DeadInst->eraseFromParent();
    
  } while (!NowDeadInsts.empty());
}

bool LoopIdiomRecognize::runOnLoop(Loop *L, LPPassManager &LPM) {
  CurLoop = L;
  
  // The trip count of the loop must be analyzable.
  SE = &getAnalysis<ScalarEvolution>();
  if (!SE->hasLoopInvariantBackedgeTakenCount(L))
    return false;
  const SCEV *BECount = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BECount)) return false;
  
  // If this loop executes exactly one time, then it should be peeled, not
  // optimized by this pass.
  if (const SCEVConstant *BECst = dyn_cast<SCEVConstant>(BECount))
    if (BECst->getValue()->getValue() == 0)
      return false;
  
  // We require target data for now.
  TD = getAnalysisIfAvailable<TargetData>();
  if (TD == 0) return false;

  DT = &getAnalysis<DominatorTree>();
  LoopInfo &LI = getAnalysis<LoopInfo>();
  
  SmallVector<BasicBlock*, 8> ExitBlocks;
  CurLoop->getUniqueExitBlocks(ExitBlocks);

  DEBUG(dbgs() << "loop-idiom Scanning: F["
               << L->getHeader()->getParent()->getName()
               << "] Loop %" << L->getHeader()->getName() << "\n");
  
  bool MadeChange = false;
  // Scan all the blocks in the loop that are not in subloops.
  for (Loop::block_iterator BI = L->block_begin(), E = L->block_end(); BI != E;
       ++BI) {
    // Ignore blocks in subloops.
    if (LI.getLoopFor(*BI) != CurLoop)
      continue;
    
    MadeChange |= runOnLoopBlock(*BI, BECount, ExitBlocks);
  }
  return MadeChange;
}

/// runOnLoopBlock - Process the specified block, which lives in a counted loop
/// with the specified backedge count.  This block is known to be in the current
/// loop and not in any subloops.
bool LoopIdiomRecognize::runOnLoopBlock(BasicBlock *BB, const SCEV *BECount,
                                     SmallVectorImpl<BasicBlock*> &ExitBlocks) {
  // We can only promote stores in this block if they are unconditionally
  // executed in the loop.  For a block to be unconditionally executed, it has
  // to dominate all the exit blocks of the loop.  Verify this now.
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (!DT->dominates(BB, ExitBlocks[i]))
      return false;
  
  bool MadeChange = false;
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
    // Look for store instructions, which may be memsets.
    StoreInst *SI = dyn_cast<StoreInst>(I++);
    if (SI == 0 || SI->isVolatile()) continue;
    
    WeakVH InstPtr(SI);
    if (!processLoopStore(SI, BECount)) continue;
    
    MadeChange = true;
    
    // If processing the store invalidated our iterator, start over from the
    // head of the loop.
    if (InstPtr == 0)
      I = BB->begin();
  }
  
  return MadeChange;
}


/// scanBlock - Look over a block to see if we can promote anything out of it.
bool LoopIdiomRecognize::processLoopStore(StoreInst *SI, const SCEV *BECount) {
  Value *StoredVal = SI->getValueOperand();
  Value *StorePtr = SI->getPointerOperand();
  
  // Reject stores that are so large that they overflow an unsigned.
  uint64_t SizeInBits = TD->getTypeSizeInBits(StoredVal->getType());
  if ((SizeInBits & 7) || (SizeInBits >> 32) != 0)
    return false;
  
  // See if the pointer expression is an AddRec like {base,+,1} on the current
  // loop, which indicates a strided store.  If we have something else, it's a
  // random store we can't handle.
  const SCEVAddRecExpr *StoreEv =
    dyn_cast<SCEVAddRecExpr>(SE->getSCEV(StorePtr));
  if (StoreEv == 0 || StoreEv->getLoop() != CurLoop || !StoreEv->isAffine())
    return false;

  // Check to see if the stride matches the size of the store.  If so, then we
  // know that every byte is touched in the loop.
  unsigned StoreSize = (unsigned)SizeInBits >> 3; 
  const SCEVConstant *Stride = dyn_cast<SCEVConstant>(StoreEv->getOperand(1));
  
  // TODO: Could also handle negative stride here someday, that will require the
  // validity check in mayLoopAccessLocation to be updated though.
  if (Stride == 0 || StoreSize != Stride->getValue()->getValue())
    return false;
  
  // If the stored value is a byte-wise value (like i32 -1), then it may be
  // turned into a memset of i8 -1, assuming that all the consequtive bytes
  // are stored.  A store of i32 0x01020304 can never be turned into a memset.
  if (Value *SplatValue = isBytewiseValue(StoredVal))
    if (processLoopStoreOfSplatValue(SI, StoreSize, SplatValue, StoreEv,
                                     BECount))
      return true;

  // If the stored value is a strided load in the same loop with the same stride
  // this this may be transformable into a memcpy.  This kicks in for stuff like
  //   for (i) A[i] = B[i];
  if (LoadInst *LI = dyn_cast<LoadInst>(StoredVal)) {
    const SCEVAddRecExpr *LoadEv =
      dyn_cast<SCEVAddRecExpr>(SE->getSCEV(LI->getOperand(0)));
    if (LoadEv && LoadEv->getLoop() == CurLoop && LoadEv->isAffine() &&
        StoreEv->getOperand(1) == LoadEv->getOperand(1) && !LI->isVolatile())
      if (processLoopStoreOfLoopLoad(SI, StoreSize, StoreEv, LoadEv, BECount))
        return true;
  }
  //errs() << "UNHANDLED strided store: " << *StoreEv << " - " << *SI << "\n";

  return false;
}

/// mayLoopAccessLocation - Return true if the specified loop might access the
/// specified pointer location, which is a loop-strided access.  The 'Access'
/// argument specifies what the verboten forms of access are (read or write).
static bool mayLoopAccessLocation(Value *Ptr,AliasAnalysis::ModRefResult Access,
                                  Loop *L, const SCEV *BECount,
                                  unsigned StoreSize, AliasAnalysis &AA,
                                  StoreInst *IgnoredStore) {
  // Get the location that may be stored across the loop.  Since the access is
  // strided positively through memory, we say that the modified location starts
  // at the pointer and has infinite size.
  uint64_t AccessSize = AliasAnalysis::UnknownSize;

  // If the loop iterates a fixed number of times, we can refine the access size
  // to be exactly the size of the memset, which is (BECount+1)*StoreSize
  if (const SCEVConstant *BECst = dyn_cast<SCEVConstant>(BECount))
    AccessSize = (BECst->getValue()->getZExtValue()+1)*StoreSize;
  
  // TODO: For this to be really effective, we have to dive into the pointer
  // operand in the store.  Store to &A[i] of 100 will always return may alias
  // with store of &A[100], we need to StoreLoc to be "A" with size of 100,
  // which will then no-alias a store to &A[100].
  AliasAnalysis::Location StoreLoc(Ptr, AccessSize);

  for (Loop::block_iterator BI = L->block_begin(), E = L->block_end(); BI != E;
       ++BI)
    for (BasicBlock::iterator I = (*BI)->begin(), E = (*BI)->end(); I != E; ++I)
      if (&*I != IgnoredStore &&
          (AA.getModRefInfo(I, StoreLoc) & Access))
        return true;

  return false;
}

/// processLoopStoreOfSplatValue - We see a strided store of a memsetable value.
/// If we can transform this into a memset in the loop preheader, do so.
bool LoopIdiomRecognize::
processLoopStoreOfSplatValue(StoreInst *SI, unsigned StoreSize,
                             Value *SplatValue,
                             const SCEVAddRecExpr *Ev, const SCEV *BECount) {
  // Verify that the stored value is loop invariant.  If not, we can't promote
  // the memset.
  if (!CurLoop->isLoopInvariant(SplatValue))
    return false;
  
  // Okay, we have a strided store "p[i]" of a splattable value.  We can turn
  // this into a memset in the loop preheader now if we want.  However, this
  // would be unsafe to do if there is anything else in the loop that may read
  // or write to the aliased location.  Check for an alias.
  if (mayLoopAccessLocation(SI->getPointerOperand(), AliasAnalysis::ModRef,
                            CurLoop, BECount,
                            StoreSize, getAnalysis<AliasAnalysis>(), SI))
    return false;
  
  // Okay, everything looks good, insert the memset.
  BasicBlock *Preheader = CurLoop->getLoopPreheader();
  
  IRBuilder<> Builder(Preheader->getTerminator());
  
  // The trip count of the loop and the base pointer of the addrec SCEV is
  // guaranteed to be loop invariant, which means that it should dominate the
  // header.  Just insert code for it in the preheader.
  SCEVExpander Expander(*SE);
  
  unsigned AddrSpace = SI->getPointerAddressSpace();
  Value *BasePtr = 
    Expander.expandCodeFor(Ev->getStart(), Builder.getInt8PtrTy(AddrSpace),
                           Preheader->getTerminator());
  
  // The # stored bytes is (BECount+1)*Size.  Expand the trip count out to
  // pointer size if it isn't already.
  const Type *IntPtr = TD->getIntPtrType(SI->getContext());
  unsigned BESize = SE->getTypeSizeInBits(BECount->getType());
  if (BESize < TD->getPointerSizeInBits())
    BECount = SE->getZeroExtendExpr(BECount, IntPtr);
  else if (BESize > TD->getPointerSizeInBits())
    BECount = SE->getTruncateExpr(BECount, IntPtr);
  
  const SCEV *NumBytesS = SE->getAddExpr(BECount, SE->getConstant(IntPtr, 1),
                                         true, true /*nooverflow*/);
  if (StoreSize != 1)
    NumBytesS = SE->getMulExpr(NumBytesS, SE->getConstant(IntPtr, StoreSize),
                               true, true /*nooverflow*/);
  
  Value *NumBytes = 
    Expander.expandCodeFor(NumBytesS, IntPtr, Preheader->getTerminator());
  
  Value *NewCall =
    Builder.CreateMemSet(BasePtr, SplatValue, NumBytes, SI->getAlignment());
  
  DEBUG(dbgs() << "  Formed memset: " << *NewCall << "\n"
               << "    from store to: " << *Ev << " at: " << *SI << "\n");
  (void)NewCall;
  
  // Okay, the memset has been formed.  Zap the original store and anything that
  // feeds into it.
  DeleteDeadInstruction(SI, *SE);
  ++NumMemSet;
  return true;
}

/// processLoopStoreOfLoopLoad - We see a strided store whose value is a
/// same-strided load.
bool LoopIdiomRecognize::
processLoopStoreOfLoopLoad(StoreInst *SI, unsigned StoreSize,
                           const SCEVAddRecExpr *StoreEv,
                           const SCEVAddRecExpr *LoadEv,
                           const SCEV *BECount) {
  LoadInst *LI = cast<LoadInst>(SI->getValueOperand());
  
  // Okay, we have a strided store "p[i]" of a loaded value.  We can turn
  // this into a memcpy in the loop preheader now if we want.  However, this
  // would be unsafe to do if there is anything else in the loop that may read
  // or write to the stored location (including the load feeding the stores).
  // Check for an alias.
  if (mayLoopAccessLocation(SI->getPointerOperand(), AliasAnalysis::ModRef,
                            CurLoop, BECount, StoreSize,
                            getAnalysis<AliasAnalysis>(), SI))
    return false;

  // For a memcpy, we have to make sure that the input array is not being
  // mutated by the loop.
  if (mayLoopAccessLocation(LI->getPointerOperand(), AliasAnalysis::Mod,
                            CurLoop, BECount, StoreSize,
                            getAnalysis<AliasAnalysis>(), SI))
    return false;
  
  // Okay, everything looks good, insert the memcpy.
  BasicBlock *Preheader = CurLoop->getLoopPreheader();
  
  IRBuilder<> Builder(Preheader->getTerminator());
  
  // The trip count of the loop and the base pointer of the addrec SCEV is
  // guaranteed to be loop invariant, which means that it should dominate the
  // header.  Just insert code for it in the preheader.
  SCEVExpander Expander(*SE);

  Value *LoadBasePtr = 
    Expander.expandCodeFor(LoadEv->getStart(),
                           Builder.getInt8PtrTy(LI->getPointerAddressSpace()),
                           Preheader->getTerminator());
  Value *StoreBasePtr = 
    Expander.expandCodeFor(StoreEv->getStart(),
                           Builder.getInt8PtrTy(SI->getPointerAddressSpace()),
                           Preheader->getTerminator());
  
  // The # stored bytes is (BECount+1)*Size.  Expand the trip count out to
  // pointer size if it isn't already.
  const Type *IntPtr = TD->getIntPtrType(SI->getContext());
  unsigned BESize = SE->getTypeSizeInBits(BECount->getType());
  if (BESize < TD->getPointerSizeInBits())
    BECount = SE->getZeroExtendExpr(BECount, IntPtr);
  else if (BESize > TD->getPointerSizeInBits())
    BECount = SE->getTruncateExpr(BECount, IntPtr);
  
  const SCEV *NumBytesS = SE->getAddExpr(BECount, SE->getConstant(IntPtr, 1),
                                         true, true /*nooverflow*/);
  if (StoreSize != 1)
    NumBytesS = SE->getMulExpr(NumBytesS, SE->getConstant(IntPtr, StoreSize),
                               true, true /*nooverflow*/);
  
  Value *NumBytes =
    Expander.expandCodeFor(NumBytesS, IntPtr, Preheader->getTerminator());
  
  Value *NewCall =
    Builder.CreateMemCpy(StoreBasePtr, LoadBasePtr, NumBytes,
                         std::min(SI->getAlignment(), LI->getAlignment()));
  
  DEBUG(dbgs() << "  Formed memcpy: " << *NewCall << "\n"
               << "    from load ptr=" << *LoadEv << " at: " << *LI << "\n"
               << "    from store ptr=" << *StoreEv << " at: " << *SI << "\n");
  (void)NewCall;
  
  // Okay, the memset has been formed.  Zap the original store and anything that
  // feeds into it.
  DeleteDeadInstruction(SI, *SE);
  ++NumMemCpy;
  return true;
}
