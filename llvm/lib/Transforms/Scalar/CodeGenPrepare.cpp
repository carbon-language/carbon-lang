//===- CodeGenPrepare.cpp - Prepare a function for code generation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass munges the code in the input function to better prepare it for
// SelectionDAG-based code generation. This works around limitations in it's
// basic-block-at-a-time approach. It should eventually be removed.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "codegenprepare"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Transforms/Utils/AddrModeMatcher.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace llvm::PatternMatch;

static cl::opt<bool> FactorCommonPreds("split-critical-paths-tweak",
                                       cl::init(false), cl::Hidden);

namespace {
  class CodeGenPrepare : public FunctionPass {
    /// TLI - Keep a pointer of a TargetLowering to consult for determining
    /// transformation profitability.
    const TargetLowering *TLI;
    ProfileInfo *PFI;

    /// BackEdges - Keep a set of all the loop back edges.
    ///
    SmallSet<std::pair<const BasicBlock*, const BasicBlock*>, 8> BackEdges;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit CodeGenPrepare(const TargetLowering *tli = 0)
      : FunctionPass(&ID), TLI(tli) {}
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<ProfileInfo>();
    }

  private:
    bool EliminateMostlyEmptyBlocks(Function &F);
    bool CanMergeBlocks(const BasicBlock *BB, const BasicBlock *DestBB) const;
    void EliminateMostlyEmptyBlock(BasicBlock *BB);
    bool OptimizeBlock(BasicBlock &BB);
    bool OptimizeMemoryInst(Instruction *I, Value *Addr, const Type *AccessTy,
                            DenseMap<Value*,Value*> &SunkAddrs);
    bool OptimizeInlineAsmInst(Instruction *I, CallSite CS,
                               DenseMap<Value*,Value*> &SunkAddrs);
    bool MoveExtToFormExtLoad(Instruction *I);
    bool OptimizeExtUses(Instruction *I);
    void findLoopBackEdges(const Function &F);
  };
}

char CodeGenPrepare::ID = 0;
static RegisterPass<CodeGenPrepare> X("codegenprepare",
                                      "Optimize for code generation");

FunctionPass *llvm::createCodeGenPreparePass(const TargetLowering *TLI) {
  return new CodeGenPrepare(TLI);
}

/// findLoopBackEdges - Do a DFS walk to find loop back edges.
///
void CodeGenPrepare::findLoopBackEdges(const Function &F) {
  SmallVector<std::pair<const BasicBlock*,const BasicBlock*>, 32> Edges;
  FindFunctionBackedges(F, Edges);
  
  BackEdges.insert(Edges.begin(), Edges.end());
}


bool CodeGenPrepare::runOnFunction(Function &F) {
  bool EverMadeChange = false;

  PFI = getAnalysisIfAvailable<ProfileInfo>();
  // First pass, eliminate blocks that contain only PHI nodes and an
  // unconditional branch.
  EverMadeChange |= EliminateMostlyEmptyBlocks(F);

  // Now find loop back edges.
  findLoopBackEdges(F);

  bool MadeChange = true;
  while (MadeChange) {
    MadeChange = false;
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      MadeChange |= OptimizeBlock(*BB);
    EverMadeChange |= MadeChange;
  }
  return EverMadeChange;
}

/// EliminateMostlyEmptyBlocks - eliminate blocks that contain only PHI nodes,
/// debug info directives, and an unconditional branch.  Passes before isel
/// (e.g. LSR/loopsimplify) often split edges in ways that are non-optimal for
/// isel.  Start by eliminating these blocks so we can split them the way we
/// want them.
bool CodeGenPrepare::EliminateMostlyEmptyBlocks(Function &F) {
  bool MadeChange = false;
  // Note that this intentionally skips the entry block.
  for (Function::iterator I = ++F.begin(), E = F.end(); I != E; ) {
    BasicBlock *BB = I++;

    // If this block doesn't end with an uncond branch, ignore it.
    BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BI || !BI->isUnconditional())
      continue;

    // If the instruction before the branch (skipping debug info) isn't a phi
    // node, then other stuff is happening here.
    BasicBlock::iterator BBI = BI;
    if (BBI != BB->begin()) {
      --BBI;
      while (isa<DbgInfoIntrinsic>(BBI)) {
        if (BBI == BB->begin())
          break;
        --BBI;
      }
      if (!isa<DbgInfoIntrinsic>(BBI) && !isa<PHINode>(BBI))
        continue;
    }

    // Do not break infinite loops.
    BasicBlock *DestBB = BI->getSuccessor(0);
    if (DestBB == BB)
      continue;

    if (!CanMergeBlocks(BB, DestBB))
      continue;

    EliminateMostlyEmptyBlock(BB);
    MadeChange = true;
  }
  return MadeChange;
}

/// CanMergeBlocks - Return true if we can merge BB into DestBB if there is a
/// single uncond branch between them, and BB contains no other non-phi
/// instructions.
bool CodeGenPrepare::CanMergeBlocks(const BasicBlock *BB,
                                    const BasicBlock *DestBB) const {
  // We only want to eliminate blocks whose phi nodes are used by phi nodes in
  // the successor.  If there are more complex condition (e.g. preheaders),
  // don't mess around with them.
  BasicBlock::const_iterator BBI = BB->begin();
  while (const PHINode *PN = dyn_cast<PHINode>(BBI++)) {
    for (Value::use_const_iterator UI = PN->use_begin(), E = PN->use_end();
         UI != E; ++UI) {
      const Instruction *User = cast<Instruction>(*UI);
      if (User->getParent() != DestBB || !isa<PHINode>(User))
        return false;
      // If User is inside DestBB block and it is a PHINode then check
      // incoming value. If incoming value is not from BB then this is
      // a complex condition (e.g. preheaders) we want to avoid here.
      if (User->getParent() == DestBB) {
        if (const PHINode *UPN = dyn_cast<PHINode>(User))
          for (unsigned I = 0, E = UPN->getNumIncomingValues(); I != E; ++I) {
            Instruction *Insn = dyn_cast<Instruction>(UPN->getIncomingValue(I));
            if (Insn && Insn->getParent() == BB &&
                Insn->getParent() != UPN->getIncomingBlock(I))
              return false;
          }
      }
    }
  }

  // If BB and DestBB contain any common predecessors, then the phi nodes in BB
  // and DestBB may have conflicting incoming values for the block.  If so, we
  // can't merge the block.
  const PHINode *DestBBPN = dyn_cast<PHINode>(DestBB->begin());
  if (!DestBBPN) return true;  // no conflict.

  // Collect the preds of BB.
  SmallPtrSet<const BasicBlock*, 16> BBPreds;
  if (const PHINode *BBPN = dyn_cast<PHINode>(BB->begin())) {
    // It is faster to get preds from a PHI than with pred_iterator.
    for (unsigned i = 0, e = BBPN->getNumIncomingValues(); i != e; ++i)
      BBPreds.insert(BBPN->getIncomingBlock(i));
  } else {
    BBPreds.insert(pred_begin(BB), pred_end(BB));
  }

  // Walk the preds of DestBB.
  for (unsigned i = 0, e = DestBBPN->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *Pred = DestBBPN->getIncomingBlock(i);
    if (BBPreds.count(Pred)) {   // Common predecessor?
      BBI = DestBB->begin();
      while (const PHINode *PN = dyn_cast<PHINode>(BBI++)) {
        const Value *V1 = PN->getIncomingValueForBlock(Pred);
        const Value *V2 = PN->getIncomingValueForBlock(BB);

        // If V2 is a phi node in BB, look up what the mapped value will be.
        if (const PHINode *V2PN = dyn_cast<PHINode>(V2))
          if (V2PN->getParent() == BB)
            V2 = V2PN->getIncomingValueForBlock(Pred);

        // If there is a conflict, bail out.
        if (V1 != V2) return false;
      }
    }
  }

  return true;
}


/// EliminateMostlyEmptyBlock - Eliminate a basic block that have only phi's and
/// an unconditional branch in it.
void CodeGenPrepare::EliminateMostlyEmptyBlock(BasicBlock *BB) {
  BranchInst *BI = cast<BranchInst>(BB->getTerminator());
  BasicBlock *DestBB = BI->getSuccessor(0);

  DEBUG(errs() << "MERGING MOSTLY EMPTY BLOCKS - BEFORE:\n" << *BB << *DestBB);

  // If the destination block has a single pred, then this is a trivial edge,
  // just collapse it.
  if (BasicBlock *SinglePred = DestBB->getSinglePredecessor()) {
    if (SinglePred != DestBB) {
      // Remember if SinglePred was the entry block of the function.  If so, we
      // will need to move BB back to the entry position.
      bool isEntry = SinglePred == &SinglePred->getParent()->getEntryBlock();
      MergeBasicBlockIntoOnlyPred(DestBB, this);

      if (isEntry && BB != &BB->getParent()->getEntryBlock())
        BB->moveBefore(&BB->getParent()->getEntryBlock());
      
      DEBUG(errs() << "AFTER:\n" << *DestBB << "\n\n\n");
      return;
    }
  }

  // Otherwise, we have multiple predecessors of BB.  Update the PHIs in DestBB
  // to handle the new incoming edges it is about to have.
  PHINode *PN;
  for (BasicBlock::iterator BBI = DestBB->begin();
       (PN = dyn_cast<PHINode>(BBI)); ++BBI) {
    // Remove the incoming value for BB, and remember it.
    Value *InVal = PN->removeIncomingValue(BB, false);

    // Two options: either the InVal is a phi node defined in BB or it is some
    // value that dominates BB.
    PHINode *InValPhi = dyn_cast<PHINode>(InVal);
    if (InValPhi && InValPhi->getParent() == BB) {
      // Add all of the input values of the input PHI as inputs of this phi.
      for (unsigned i = 0, e = InValPhi->getNumIncomingValues(); i != e; ++i)
        PN->addIncoming(InValPhi->getIncomingValue(i),
                        InValPhi->getIncomingBlock(i));
    } else {
      // Otherwise, add one instance of the dominating value for each edge that
      // we will be adding.
      if (PHINode *BBPN = dyn_cast<PHINode>(BB->begin())) {
        for (unsigned i = 0, e = BBPN->getNumIncomingValues(); i != e; ++i)
          PN->addIncoming(InVal, BBPN->getIncomingBlock(i));
      } else {
        for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
          PN->addIncoming(InVal, *PI);
      }
    }
  }

  // The PHIs are now updated, change everything that refers to BB to use
  // DestBB and remove BB.
  BB->replaceAllUsesWith(DestBB);
  if (PFI) {
    PFI->replaceAllUses(BB, DestBB);
    PFI->removeEdge(ProfileInfo::getEdge(BB, DestBB));
  }
  BB->eraseFromParent();

  DEBUG(errs() << "AFTER:\n" << *DestBB << "\n\n\n");
}


/// SplitEdgeNicely - Split the critical edge from TI to its specified
/// successor if it will improve codegen.  We only do this if the successor has
/// phi nodes (otherwise critical edges are ok).  If there is already another
/// predecessor of the succ that is empty (and thus has no phi nodes), use it
/// instead of introducing a new block.
static void SplitEdgeNicely(TerminatorInst *TI, unsigned SuccNum,
                     SmallSet<std::pair<const BasicBlock*,
                                        const BasicBlock*>, 8> &BackEdges,
                             Pass *P) {
  BasicBlock *TIBB = TI->getParent();
  BasicBlock *Dest = TI->getSuccessor(SuccNum);
  assert(isa<PHINode>(Dest->begin()) &&
         "This should only be called if Dest has a PHI!");

  // Do not split edges to EH landing pads.
  if (InvokeInst *Invoke = dyn_cast<InvokeInst>(TI)) {
    if (Invoke->getSuccessor(1) == Dest)
      return;
  }
  

  // As a hack, never split backedges of loops.  Even though the copy for any
  // PHIs inserted on the backedge would be dead for exits from the loop, we
  // assume that the cost of *splitting* the backedge would be too high.
  if (BackEdges.count(std::make_pair(TIBB, Dest)))
    return;

  if (!FactorCommonPreds) {
    /// TIPHIValues - This array is lazily computed to determine the values of
    /// PHIs in Dest that TI would provide.
    SmallVector<Value*, 32> TIPHIValues;

    // Check to see if Dest has any blocks that can be used as a split edge for
    // this terminator.
    for (pred_iterator PI = pred_begin(Dest), E = pred_end(Dest); PI != E; ++PI) {
      BasicBlock *Pred = *PI;
      // To be usable, the pred has to end with an uncond branch to the dest.
      BranchInst *PredBr = dyn_cast<BranchInst>(Pred->getTerminator());
      if (!PredBr || !PredBr->isUnconditional())
        continue;
      // Must be empty other than the branch and debug info.
      BasicBlock::iterator I = Pred->begin();
      while (isa<DbgInfoIntrinsic>(I))
        I++;
      if (dyn_cast<Instruction>(I) != PredBr)
        continue;
      // Cannot be the entry block; its label does not get emitted.
      if (Pred == &(Dest->getParent()->getEntryBlock()))
        continue;

      // Finally, since we know that Dest has phi nodes in it, we have to make
      // sure that jumping to Pred will have the same effect as going to Dest in
      // terms of PHI values.
      PHINode *PN;
      unsigned PHINo = 0;
      bool FoundMatch = true;
      for (BasicBlock::iterator I = Dest->begin();
           (PN = dyn_cast<PHINode>(I)); ++I, ++PHINo) {
        if (PHINo == TIPHIValues.size())
          TIPHIValues.push_back(PN->getIncomingValueForBlock(TIBB));

        // If the PHI entry doesn't work, we can't use this pred.
        if (TIPHIValues[PHINo] != PN->getIncomingValueForBlock(Pred)) {
          FoundMatch = false;
          break;
        }
      }

      // If we found a workable predecessor, change TI to branch to Succ.
      if (FoundMatch) {
        ProfileInfo *PFI = P->getAnalysisIfAvailable<ProfileInfo>();
        if (PFI)
          PFI->splitEdge(TIBB, Dest, Pred);
        Dest->removePredecessor(TIBB);
        TI->setSuccessor(SuccNum, Pred);
        return;
      }
    }

    SplitCriticalEdge(TI, SuccNum, P, true);
    return;
  }

  PHINode *PN;
  SmallVector<Value*, 8> TIPHIValues;
  for (BasicBlock::iterator I = Dest->begin();
       (PN = dyn_cast<PHINode>(I)); ++I)
    TIPHIValues.push_back(PN->getIncomingValueForBlock(TIBB));

  SmallVector<BasicBlock*, 8> IdenticalPreds;
  for (pred_iterator PI = pred_begin(Dest), E = pred_end(Dest); PI != E; ++PI) {
    BasicBlock *Pred = *PI;
    if (BackEdges.count(std::make_pair(Pred, Dest)))
      continue;
    if (PI == TIBB)
      IdenticalPreds.push_back(Pred);
    else {
      bool Identical = true;
      unsigned PHINo = 0;
      for (BasicBlock::iterator I = Dest->begin();
           (PN = dyn_cast<PHINode>(I)); ++I, ++PHINo)
        if (TIPHIValues[PHINo] != PN->getIncomingValueForBlock(Pred)) {
          Identical = false;
          break;
        }
      if (Identical)
        IdenticalPreds.push_back(Pred);
    }
  }

  assert(!IdenticalPreds.empty());
  SplitBlockPredecessors(Dest, &IdenticalPreds[0], IdenticalPreds.size(),
                         ".critedge", P);
}


/// OptimizeNoopCopyExpression - If the specified cast instruction is a noop
/// copy (e.g. it's casting from one pointer type to another, i32->i8 on PPC),
/// sink it into user blocks to reduce the number of virtual
/// registers that must be created and coalesced.
///
/// Return true if any changes are made.
///
static bool OptimizeNoopCopyExpression(CastInst *CI, const TargetLowering &TLI){
  // If this is a noop copy,
  EVT SrcVT = TLI.getValueType(CI->getOperand(0)->getType());
  EVT DstVT = TLI.getValueType(CI->getType());

  // This is an fp<->int conversion?
  if (SrcVT.isInteger() != DstVT.isInteger())
    return false;

  // If this is an extension, it will be a zero or sign extension, which
  // isn't a noop.
  if (SrcVT.bitsLT(DstVT)) return false;

  // If these values will be promoted, find out what they will be promoted
  // to.  This helps us consider truncates on PPC as noop copies when they
  // are.
  if (TLI.getTypeAction(CI->getContext(), SrcVT) == TargetLowering::Promote)
    SrcVT = TLI.getTypeToTransformTo(CI->getContext(), SrcVT);
  if (TLI.getTypeAction(CI->getContext(), DstVT) == TargetLowering::Promote)
    DstVT = TLI.getTypeToTransformTo(CI->getContext(), DstVT);

  // If, after promotion, these are the same types, this is a noop copy.
  if (SrcVT != DstVT)
    return false;

  BasicBlock *DefBB = CI->getParent();

  /// InsertedCasts - Only insert a cast in each block once.
  DenseMap<BasicBlock*, CastInst*> InsertedCasts;

  bool MadeChange = false;
  for (Value::use_iterator UI = CI->use_begin(), E = CI->use_end();
       UI != E; ) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Figure out which BB this cast is used in.  For PHI's this is the
    // appropriate predecessor block.
    BasicBlock *UserBB = User->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(User)) {
      UserBB = PN->getIncomingBlock(UI);
    }

    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    // If this user is in the same block as the cast, don't change the cast.
    if (UserBB == DefBB) continue;

    // If we have already inserted a cast into this block, use it.
    CastInst *&InsertedCast = InsertedCasts[UserBB];

    if (!InsertedCast) {
      BasicBlock::iterator InsertPt = UserBB->getFirstNonPHI();

      InsertedCast =
        CastInst::Create(CI->getOpcode(), CI->getOperand(0), CI->getType(), "",
                         InsertPt);
      MadeChange = true;
    }

    // Replace a use of the cast with a use of the new cast.
    TheUse = InsertedCast;
  }

  // If we removed all uses, nuke the cast.
  if (CI->use_empty()) {
    CI->eraseFromParent();
    MadeChange = true;
  }

  return MadeChange;
}

/// OptimizeCmpExpression - sink the given CmpInst into user blocks to reduce
/// the number of virtual registers that must be created and coalesced.  This is
/// a clear win except on targets with multiple condition code registers
///  (PowerPC), where it might lose; some adjustment may be wanted there.
///
/// Return true if any changes are made.
static bool OptimizeCmpExpression(CmpInst *CI) {
  BasicBlock *DefBB = CI->getParent();

  /// InsertedCmp - Only insert a cmp in each block once.
  DenseMap<BasicBlock*, CmpInst*> InsertedCmps;

  bool MadeChange = false;
  for (Value::use_iterator UI = CI->use_begin(), E = CI->use_end();
       UI != E; ) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    // Don't bother for PHI nodes.
    if (isa<PHINode>(User))
      continue;

    // Figure out which BB this cmp is used in.
    BasicBlock *UserBB = User->getParent();

    // If this user is in the same block as the cmp, don't change the cmp.
    if (UserBB == DefBB) continue;

    // If we have already inserted a cmp into this block, use it.
    CmpInst *&InsertedCmp = InsertedCmps[UserBB];

    if (!InsertedCmp) {
      BasicBlock::iterator InsertPt = UserBB->getFirstNonPHI();

      InsertedCmp =
        CmpInst::Create(CI->getOpcode(),
                        CI->getPredicate(),  CI->getOperand(0),
                        CI->getOperand(1), "", InsertPt);
      MadeChange = true;
    }

    // Replace a use of the cmp with a use of the new cmp.
    TheUse = InsertedCmp;
  }

  // If we removed all uses, nuke the cmp.
  if (CI->use_empty())
    CI->eraseFromParent();

  return MadeChange;
}

//===----------------------------------------------------------------------===//
// Memory Optimization
//===----------------------------------------------------------------------===//

/// IsNonLocalValue - Return true if the specified values are defined in a
/// different basic block than BB.
static bool IsNonLocalValue(Value *V, BasicBlock *BB) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent() != BB;
  return false;
}

/// OptimizeMemoryInst - Load and Store Instructions often have
/// addressing modes that can do significant amounts of computation.  As such,
/// instruction selection will try to get the load or store to do as much
/// computation as possible for the program.  The problem is that isel can only
/// see within a single block.  As such, we sink as much legal addressing mode
/// stuff into the block as possible.
///
/// This method is used to optimize both load/store and inline asms with memory
/// operands.
bool CodeGenPrepare::OptimizeMemoryInst(Instruction *MemoryInst, Value *Addr,
                                        const Type *AccessTy,
                                        DenseMap<Value*,Value*> &SunkAddrs) {
  // Figure out what addressing mode will be built up for this operation.
  SmallVector<Instruction*, 16> AddrModeInsts;
  ExtAddrMode AddrMode = AddressingModeMatcher::Match(Addr, AccessTy,MemoryInst,
                                                      AddrModeInsts, *TLI);

  // Check to see if any of the instructions supersumed by this addr mode are
  // non-local to I's BB.
  bool AnyNonLocal = false;
  for (unsigned i = 0, e = AddrModeInsts.size(); i != e; ++i) {
    if (IsNonLocalValue(AddrModeInsts[i], MemoryInst->getParent())) {
      AnyNonLocal = true;
      break;
    }
  }

  // If all the instructions matched are already in this BB, don't do anything.
  if (!AnyNonLocal) {
    DEBUG(errs() << "CGP: Found      local addrmode: " << AddrMode << "\n");
    return false;
  }

  // Insert this computation right after this user.  Since our caller is
  // scanning from the top of the BB to the bottom, reuse of the expr are
  // guaranteed to happen later.
  BasicBlock::iterator InsertPt = MemoryInst;

  // Now that we determined the addressing expression we want to use and know
  // that we have to sink it into this block.  Check to see if we have already
  // done this for some other load/store instr in this block.  If so, reuse the
  // computation.
  Value *&SunkAddr = SunkAddrs[Addr];
  if (SunkAddr) {
    DEBUG(errs() << "CGP: Reusing nonlocal addrmode: " << AddrMode << " for "
                 << *MemoryInst);
    if (SunkAddr->getType() != Addr->getType())
      SunkAddr = new BitCastInst(SunkAddr, Addr->getType(), "tmp", InsertPt);
  } else {
    DEBUG(errs() << "CGP: SINKING nonlocal addrmode: " << AddrMode << " for "
                 << *MemoryInst);
    const Type *IntPtrTy =
          TLI->getTargetData()->getIntPtrType(AccessTy->getContext());

    Value *Result = 0;
    // Start with the scale value.
    if (AddrMode.Scale) {
      Value *V = AddrMode.ScaledReg;
      if (V->getType() == IntPtrTy) {
        // done.
      } else if (isa<PointerType>(V->getType())) {
        V = new PtrToIntInst(V, IntPtrTy, "sunkaddr", InsertPt);
      } else if (cast<IntegerType>(IntPtrTy)->getBitWidth() <
                 cast<IntegerType>(V->getType())->getBitWidth()) {
        V = new TruncInst(V, IntPtrTy, "sunkaddr", InsertPt);
      } else {
        V = new SExtInst(V, IntPtrTy, "sunkaddr", InsertPt);
      }
      if (AddrMode.Scale != 1)
        V = BinaryOperator::CreateMul(V, ConstantInt::get(IntPtrTy,
                                                                AddrMode.Scale),
                                      "sunkaddr", InsertPt);
      Result = V;
    }

    // Add in the base register.
    if (AddrMode.BaseReg) {
      Value *V = AddrMode.BaseReg;
      if (isa<PointerType>(V->getType()))
        V = new PtrToIntInst(V, IntPtrTy, "sunkaddr", InsertPt);
      if (V->getType() != IntPtrTy)
        V = CastInst::CreateIntegerCast(V, IntPtrTy, /*isSigned=*/true,
                                        "sunkaddr", InsertPt);
      if (Result)
        Result = BinaryOperator::CreateAdd(Result, V, "sunkaddr", InsertPt);
      else
        Result = V;
    }

    // Add in the BaseGV if present.
    if (AddrMode.BaseGV) {
      Value *V = new PtrToIntInst(AddrMode.BaseGV, IntPtrTy, "sunkaddr",
                                  InsertPt);
      if (Result)
        Result = BinaryOperator::CreateAdd(Result, V, "sunkaddr", InsertPt);
      else
        Result = V;
    }

    // Add in the Base Offset if present.
    if (AddrMode.BaseOffs) {
      Value *V = ConstantInt::get(IntPtrTy, AddrMode.BaseOffs);
      if (Result)
        Result = BinaryOperator::CreateAdd(Result, V, "sunkaddr", InsertPt);
      else
        Result = V;
    }

    if (Result == 0)
      SunkAddr = Constant::getNullValue(Addr->getType());
    else
      SunkAddr = new IntToPtrInst(Result, Addr->getType(), "sunkaddr",InsertPt);
  }

  MemoryInst->replaceUsesOfWith(Addr, SunkAddr);

  if (Addr->use_empty())
    RecursivelyDeleteTriviallyDeadInstructions(Addr);
  return true;
}

/// OptimizeInlineAsmInst - If there are any memory operands, use
/// OptimizeMemoryInst to sink their address computing into the block when
/// possible / profitable.
bool CodeGenPrepare::OptimizeInlineAsmInst(Instruction *I, CallSite CS,
                                           DenseMap<Value*,Value*> &SunkAddrs) {
  bool MadeChange = false;
  InlineAsm *IA = cast<InlineAsm>(CS.getCalledValue());

  // Do a prepass over the constraints, canonicalizing them, and building up the
  // ConstraintOperands list.
  std::vector<InlineAsm::ConstraintInfo>
    ConstraintInfos = IA->ParseConstraints();

  /// ConstraintOperands - Information about all of the constraints.
  std::vector<TargetLowering::AsmOperandInfo> ConstraintOperands;
  unsigned ArgNo = 0;   // ArgNo - The argument of the CallInst.
  for (unsigned i = 0, e = ConstraintInfos.size(); i != e; ++i) {
    ConstraintOperands.
      push_back(TargetLowering::AsmOperandInfo(ConstraintInfos[i]));
    TargetLowering::AsmOperandInfo &OpInfo = ConstraintOperands.back();

    // Compute the value type for each operand.
    switch (OpInfo.Type) {
    case InlineAsm::isOutput:
      if (OpInfo.isIndirect)
        OpInfo.CallOperandVal = CS.getArgument(ArgNo++);
      break;
    case InlineAsm::isInput:
      OpInfo.CallOperandVal = CS.getArgument(ArgNo++);
      break;
    case InlineAsm::isClobber:
      // Nothing to do.
      break;
    }

    // Compute the constraint code and ConstraintType to use.
    TLI->ComputeConstraintToUse(OpInfo, SDValue(),
                             OpInfo.ConstraintType == TargetLowering::C_Memory);

    if (OpInfo.ConstraintType == TargetLowering::C_Memory &&
        OpInfo.isIndirect) {
      Value *OpVal = OpInfo.CallOperandVal;
      MadeChange |= OptimizeMemoryInst(I, OpVal, OpVal->getType(), SunkAddrs);
    }
  }

  return MadeChange;
}

/// MoveExtToFormExtLoad - Move a zext or sext fed by a load into the same
/// basic block as the load, unless conditions are unfavorable. This allows
/// SelectionDAG to fold the extend into the load.
///
bool CodeGenPrepare::MoveExtToFormExtLoad(Instruction *I) {
  // Look for a load being extended.
  LoadInst *LI = dyn_cast<LoadInst>(I->getOperand(0));
  if (!LI) return false;

  // If they're already in the same block, there's nothing to do.
  if (LI->getParent() == I->getParent())
    return false;

  // If the load has other users and the truncate is not free, this probably
  // isn't worthwhile.
  if (!LI->hasOneUse() &&
      TLI && !TLI->isTruncateFree(I->getType(), LI->getType()))
    return false;

  // Check whether the target supports casts folded into loads.
  unsigned LType;
  if (isa<ZExtInst>(I))
    LType = ISD::ZEXTLOAD;
  else {
    assert(isa<SExtInst>(I) && "Unexpected ext type!");
    LType = ISD::SEXTLOAD;
  }
  if (TLI && !TLI->isLoadExtLegal(LType, TLI->getValueType(LI->getType())))
    return false;

  // Move the extend into the same block as the load, so that SelectionDAG
  // can fold it.
  I->removeFromParent();
  I->insertAfter(LI);
  return true;
}

bool CodeGenPrepare::OptimizeExtUses(Instruction *I) {
  BasicBlock *DefBB = I->getParent();

  // If both result of the {s|z}xt and its source are live out, rewrite all
  // other uses of the source with result of extension.
  Value *Src = I->getOperand(0);
  if (Src->hasOneUse())
    return false;

  // Only do this xform if truncating is free.
  if (TLI && !TLI->isTruncateFree(I->getType(), Src->getType()))
    return false;

  // Only safe to perform the optimization if the source is also defined in
  // this block.
  if (!isa<Instruction>(Src) || DefBB != cast<Instruction>(Src)->getParent())
    return false;

  bool DefIsLiveOut = false;
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);

    // Figure out which BB this ext is used in.
    BasicBlock *UserBB = User->getParent();
    if (UserBB == DefBB) continue;
    DefIsLiveOut = true;
    break;
  }
  if (!DefIsLiveOut)
    return false;

  // Make sure non of the uses are PHI nodes.
  for (Value::use_iterator UI = Src->use_begin(), E = Src->use_end();
       UI != E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    BasicBlock *UserBB = User->getParent();
    if (UserBB == DefBB) continue;
    // Be conservative. We don't want this xform to end up introducing
    // reloads just before load / store instructions.
    if (isa<PHINode>(User) || isa<LoadInst>(User) || isa<StoreInst>(User))
      return false;
  }

  // InsertedTruncs - Only insert one trunc in each block once.
  DenseMap<BasicBlock*, Instruction*> InsertedTruncs;

  bool MadeChange = false;
  for (Value::use_iterator UI = Src->use_begin(), E = Src->use_end();
       UI != E; ++UI) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Figure out which BB this ext is used in.
    BasicBlock *UserBB = User->getParent();
    if (UserBB == DefBB) continue;

    // Both src and def are live in this block. Rewrite the use.
    Instruction *&InsertedTrunc = InsertedTruncs[UserBB];

    if (!InsertedTrunc) {
      BasicBlock::iterator InsertPt = UserBB->getFirstNonPHI();

      InsertedTrunc = new TruncInst(I, Src->getType(), "", InsertPt);
    }

    // Replace a use of the {s|z}ext source with a use of the result.
    TheUse = InsertedTrunc;

    MadeChange = true;
  }

  return MadeChange;
}

// In this pass we look for GEP and cast instructions that are used
// across basic blocks and rewrite them to improve basic-block-at-a-time
// selection.
bool CodeGenPrepare::OptimizeBlock(BasicBlock &BB) {
  bool MadeChange = false;

  // Split all critical edges where the dest block has a PHI.
  TerminatorInst *BBTI = BB.getTerminator();
  if (BBTI->getNumSuccessors() > 1 && !isa<IndirectBrInst>(BBTI)) {
    for (unsigned i = 0, e = BBTI->getNumSuccessors(); i != e; ++i) {
      BasicBlock *SuccBB = BBTI->getSuccessor(i);
      if (isa<PHINode>(SuccBB->begin()) && isCriticalEdge(BBTI, i, true))
        SplitEdgeNicely(BBTI, i, BackEdges, this);
    }
  }

  // Keep track of non-local addresses that have been sunk into this block.
  // This allows us to avoid inserting duplicate code for blocks with multiple
  // load/stores of the same address.
  DenseMap<Value*, Value*> SunkAddrs;

  for (BasicBlock::iterator BBI = BB.begin(), E = BB.end(); BBI != E; ) {
    Instruction *I = BBI++;

    if (CastInst *CI = dyn_cast<CastInst>(I)) {
      // If the source of the cast is a constant, then this should have
      // already been constant folded.  The only reason NOT to constant fold
      // it is if something (e.g. LSR) was careful to place the constant
      // evaluation in a block other than then one that uses it (e.g. to hoist
      // the address of globals out of a loop).  If this is the case, we don't
      // want to forward-subst the cast.
      if (isa<Constant>(CI->getOperand(0)))
        continue;

      bool Change = false;
      if (TLI) {
        Change = OptimizeNoopCopyExpression(CI, *TLI);
        MadeChange |= Change;
      }

      if (!Change && (isa<ZExtInst>(I) || isa<SExtInst>(I))) {
        MadeChange |= MoveExtToFormExtLoad(I);
        MadeChange |= OptimizeExtUses(I);
      }
    } else if (CmpInst *CI = dyn_cast<CmpInst>(I)) {
      MadeChange |= OptimizeCmpExpression(CI);
    } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      if (TLI)
        MadeChange |= OptimizeMemoryInst(I, I->getOperand(0), LI->getType(),
                                         SunkAddrs);
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      if (TLI)
        MadeChange |= OptimizeMemoryInst(I, SI->getOperand(1),
                                         SI->getOperand(0)->getType(),
                                         SunkAddrs);
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
      if (GEPI->hasAllZeroIndices()) {
        /// The GEP operand must be a pointer, so must its result -> BitCast
        Instruction *NC = new BitCastInst(GEPI->getOperand(0), GEPI->getType(),
                                          GEPI->getName(), GEPI);
        GEPI->replaceAllUsesWith(NC);
        GEPI->eraseFromParent();
        MadeChange = true;
        BBI = NC;
      }
    } else if (CallInst *CI = dyn_cast<CallInst>(I)) {
      // If we found an inline asm expession, and if the target knows how to
      // lower it to normal LLVM code, do so now.
      if (TLI && isa<InlineAsm>(CI->getCalledValue())) {
        if (TLI->ExpandInlineAsm(CI)) {
          BBI = BB.begin();
          // Avoid processing instructions out of order, which could cause
          // reuse before a value is defined.
          SunkAddrs.clear();
        } else
          // Sink address computing for memory operands into the block.
          MadeChange |= OptimizeInlineAsmInst(I, &(*CI), SunkAddrs);
      }
    }
  }

  return MadeChange;
}
