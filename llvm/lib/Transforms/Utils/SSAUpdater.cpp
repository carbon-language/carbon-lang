//===- SSAUpdater.cpp - Unstructured SSA Update Tool ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SSAUpdater class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ssaupdater"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

/// BBInfo - Per-basic block information used internally by SSAUpdater.
/// The predecessors of each block are cached here since pred_iterator is
/// slow and we need to iterate over the blocks at least a few times.
class SSAUpdater::BBInfo {
public:
  BasicBlock *BB;      // Back-pointer to the corresponding block.
  Value *AvailableVal; // Value to use in this block.
  BBInfo *DefBB;       // Block that defines the available value.
  int BlkNum;          // Postorder number.
  BBInfo *IDom;        // Immediate dominator.
  unsigned NumPreds;   // Number of predecessor blocks.
  BBInfo **Preds;      // Array[NumPreds] of predecessor blocks.
  PHINode *PHITag;     // Marker for existing PHIs that match.

  BBInfo(BasicBlock *ThisBB, Value *V)
    : BB(ThisBB), AvailableVal(V), DefBB(V ? this : 0), BlkNum(0), IDom(0),
      NumPreds(0), Preds(0), PHITag(0) { }
};

typedef DenseMap<BasicBlock*, SSAUpdater::BBInfo*> BBMapTy;

typedef DenseMap<BasicBlock*, Value*> AvailableValsTy;
static AvailableValsTy &getAvailableVals(void *AV) {
  return *static_cast<AvailableValsTy*>(AV);
}

static BBMapTy *getBBMap(void *BM) {
  return static_cast<BBMapTy*>(BM);
}

SSAUpdater::SSAUpdater(SmallVectorImpl<PHINode*> *NewPHI)
  : AV(0), PrototypeValue(0), BM(0), InsertedPHIs(NewPHI) {}

SSAUpdater::~SSAUpdater() {
  delete &getAvailableVals(AV);
}

/// Initialize - Reset this object to get ready for a new set of SSA
/// updates.  ProtoValue is the value used to name PHI nodes.
void SSAUpdater::Initialize(Value *ProtoValue) {
  if (AV == 0)
    AV = new AvailableValsTy();
  else
    getAvailableVals(AV).clear();
  PrototypeValue = ProtoValue;
}

/// HasValueForBlock - Return true if the SSAUpdater already has a value for
/// the specified block.
bool SSAUpdater::HasValueForBlock(BasicBlock *BB) const {
  return getAvailableVals(AV).count(BB);
}

/// AddAvailableValue - Indicate that a rewritten value is available in the
/// specified block with the specified value.
void SSAUpdater::AddAvailableValue(BasicBlock *BB, Value *V) {
  assert(PrototypeValue != 0 && "Need to initialize SSAUpdater");
  assert(PrototypeValue->getType() == V->getType() &&
         "All rewritten values must have the same type");
  getAvailableVals(AV)[BB] = V;
}

/// IsEquivalentPHI - Check if PHI has the same incoming value as specified
/// in ValueMapping for each predecessor block.
static bool IsEquivalentPHI(PHINode *PHI,
                            DenseMap<BasicBlock*, Value*> &ValueMapping) {
  unsigned PHINumValues = PHI->getNumIncomingValues();
  if (PHINumValues != ValueMapping.size())
    return false;

  // Scan the phi to see if it matches.
  for (unsigned i = 0, e = PHINumValues; i != e; ++i)
    if (ValueMapping[PHI->getIncomingBlock(i)] !=
        PHI->getIncomingValue(i)) {
      return false;
    }

  return true;
}

/// GetValueAtEndOfBlock - Construct SSA form, materializing a value that is
/// live at the end of the specified block.
Value *SSAUpdater::GetValueAtEndOfBlock(BasicBlock *BB) {
  assert(BM == 0 && "Unexpected Internal State");
  Value *Res = GetValueAtEndOfBlockInternal(BB);
  assert(BM == 0 && "Unexpected Internal State");
  return Res;
}

/// GetValueInMiddleOfBlock - Construct SSA form, materializing a value that
/// is live in the middle of the specified block.
///
/// GetValueInMiddleOfBlock is the same as GetValueAtEndOfBlock except in one
/// important case: if there is a definition of the rewritten value after the
/// 'use' in BB.  Consider code like this:
///
///      X1 = ...
///   SomeBB:
///      use(X)
///      X2 = ...
///      br Cond, SomeBB, OutBB
///
/// In this case, there are two values (X1 and X2) added to the AvailableVals
/// set by the client of the rewriter, and those values are both live out of
/// their respective blocks.  However, the use of X happens in the *middle* of
/// a block.  Because of this, we need to insert a new PHI node in SomeBB to
/// merge the appropriate values, and this value isn't live out of the block.
///
Value *SSAUpdater::GetValueInMiddleOfBlock(BasicBlock *BB) {
  // If there is no definition of the renamed variable in this block, just use
  // GetValueAtEndOfBlock to do our work.
  if (!HasValueForBlock(BB))
    return GetValueAtEndOfBlock(BB);

  // Otherwise, we have the hard case.  Get the live-in values for each
  // predecessor.
  SmallVector<std::pair<BasicBlock*, Value*>, 8> PredValues;
  Value *SingularValue = 0;

  // We can get our predecessor info by walking the pred_iterator list, but it
  // is relatively slow.  If we already have PHI nodes in this block, walk one
  // of them to get the predecessor list instead.
  if (PHINode *SomePhi = dyn_cast<PHINode>(BB->begin())) {
    for (unsigned i = 0, e = SomePhi->getNumIncomingValues(); i != e; ++i) {
      BasicBlock *PredBB = SomePhi->getIncomingBlock(i);
      Value *PredVal = GetValueAtEndOfBlock(PredBB);
      PredValues.push_back(std::make_pair(PredBB, PredVal));

      // Compute SingularValue.
      if (i == 0)
        SingularValue = PredVal;
      else if (PredVal != SingularValue)
        SingularValue = 0;
    }
  } else {
    bool isFirstPred = true;
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      BasicBlock *PredBB = *PI;
      Value *PredVal = GetValueAtEndOfBlock(PredBB);
      PredValues.push_back(std::make_pair(PredBB, PredVal));

      // Compute SingularValue.
      if (isFirstPred) {
        SingularValue = PredVal;
        isFirstPred = false;
      } else if (PredVal != SingularValue)
        SingularValue = 0;
    }
  }

  // If there are no predecessors, just return undef.
  if (PredValues.empty())
    return UndefValue::get(PrototypeValue->getType());

  // Otherwise, if all the merged values are the same, just use it.
  if (SingularValue != 0)
    return SingularValue;

  // Otherwise, we do need a PHI: check to see if we already have one available
  // in this block that produces the right value.
  if (isa<PHINode>(BB->begin())) {
    DenseMap<BasicBlock*, Value*> ValueMapping(PredValues.begin(),
                                               PredValues.end());
    PHINode *SomePHI;
    for (BasicBlock::iterator It = BB->begin();
         (SomePHI = dyn_cast<PHINode>(It)); ++It) {
      if (IsEquivalentPHI(SomePHI, ValueMapping))
        return SomePHI;
    }
  }

  // Ok, we have no way out, insert a new one now.
  PHINode *InsertedPHI = PHINode::Create(PrototypeValue->getType(),
                                         PrototypeValue->getName(),
                                         &BB->front());
  InsertedPHI->reserveOperandSpace(PredValues.size());

  // Fill in all the predecessors of the PHI.
  for (unsigned i = 0, e = PredValues.size(); i != e; ++i)
    InsertedPHI->addIncoming(PredValues[i].second, PredValues[i].first);

  // See if the PHI node can be merged to a single value.  This can happen in
  // loop cases when we get a PHI of itself and one other value.
  if (Value *ConstVal = InsertedPHI->hasConstantValue()) {
    InsertedPHI->eraseFromParent();
    return ConstVal;
  }

  // If the client wants to know about all new instructions, tell it.
  if (InsertedPHIs) InsertedPHIs->push_back(InsertedPHI);

  DEBUG(dbgs() << "  Inserted PHI: " << *InsertedPHI << "\n");
  return InsertedPHI;
}

/// RewriteUse - Rewrite a use of the symbolic value.  This handles PHI nodes,
/// which use their value in the corresponding predecessor.
void SSAUpdater::RewriteUse(Use &U) {
  Instruction *User = cast<Instruction>(U.getUser());

  Value *V;
  if (PHINode *UserPN = dyn_cast<PHINode>(User))
    V = GetValueAtEndOfBlock(UserPN->getIncomingBlock(U));
  else
    V = GetValueInMiddleOfBlock(User->getParent());

  U.set(V);
}

/// GetValueAtEndOfBlockInternal - Check to see if AvailableVals has an entry
/// for the specified BB and if so, return it.  If not, construct SSA form by
/// first calculating the required placement of PHIs and then inserting new
/// PHIs where needed.
Value *SSAUpdater::GetValueAtEndOfBlockInternal(BasicBlock *BB) {
  AvailableValsTy &AvailableVals = getAvailableVals(AV);
  if (Value *V = AvailableVals[BB])
    return V;

  // Pool allocation used internally by GetValueAtEndOfBlock.
  BumpPtrAllocator Allocator;
  BBMapTy BBMapObj;
  BM = &BBMapObj;

  SmallVector<BBInfo*, 100> BlockList;
  BuildBlockList(BB, &BlockList, &Allocator);

  // Special case: bail out if BB is unreachable.
  if (BlockList.size() == 0) {
    BM = 0;
    return UndefValue::get(PrototypeValue->getType());
  }

  FindDominators(&BlockList);
  FindPHIPlacement(&BlockList);
  FindAvailableVals(&BlockList);

  BM = 0;
  return BBMapObj[BB]->DefBB->AvailableVal;
}

/// FindPredecessorBlocks - Put the predecessors of Info->BB into the Preds
/// vector, set Info->NumPreds, and allocate space in Info->Preds.
static void FindPredecessorBlocks(SSAUpdater::BBInfo *Info,
                                  SmallVectorImpl<BasicBlock*> *Preds,
                                  BumpPtrAllocator *Allocator) {
  // We can get our predecessor info by walking the pred_iterator list,
  // but it is relatively slow.  If we already have PHI nodes in this
  // block, walk one of them to get the predecessor list instead.
  BasicBlock *BB = Info->BB;
  if (PHINode *SomePhi = dyn_cast<PHINode>(BB->begin())) {
    for (unsigned PI = 0, E = SomePhi->getNumIncomingValues(); PI != E; ++PI)
      Preds->push_back(SomePhi->getIncomingBlock(PI));
  } else {
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
      Preds->push_back(*PI);
  }

  Info->NumPreds = Preds->size();
  Info->Preds = static_cast<SSAUpdater::BBInfo**>
    (Allocator->Allocate(Info->NumPreds * sizeof(SSAUpdater::BBInfo*),
                         AlignOf<SSAUpdater::BBInfo*>::Alignment));
}

/// BuildBlockList - Starting from the specified basic block, traverse back
/// through its predecessors until reaching blocks with known values.  Create
/// BBInfo structures for the blocks and append them to the block list.
void SSAUpdater::BuildBlockList(BasicBlock *BB, BlockListTy *BlockList,
                                BumpPtrAllocator *Allocator) {
  AvailableValsTy &AvailableVals = getAvailableVals(AV);
  BBMapTy *BBMap = getBBMap(BM);
  SmallVector<BBInfo*, 10> RootList;
  SmallVector<BBInfo*, 64> WorkList;

  BBInfo *Info = new (*Allocator) BBInfo(BB, 0);
  (*BBMap)[BB] = Info;
  WorkList.push_back(Info);

  // Search backward from BB, creating BBInfos along the way and stopping when
  // reaching blocks that define the value.  Record those defining blocks on
  // the RootList.
  SmallVector<BasicBlock*, 10> Preds;
  while (!WorkList.empty()) {
    Info = WorkList.pop_back_val();
    Preds.clear();
    FindPredecessorBlocks(Info, &Preds, Allocator);

    // Treat an unreachable predecessor as a definition with 'undef'.
    if (Info->NumPreds == 0) {
      Info->AvailableVal = UndefValue::get(PrototypeValue->getType());
      Info->DefBB = Info;
      RootList.push_back(Info);
      continue;
    }

    for (unsigned p = 0; p != Info->NumPreds; ++p) {
      BasicBlock *Pred = Preds[p];
      // Check if BBMap already has a BBInfo for the predecessor block.
      BBMapTy::value_type &BBMapBucket = BBMap->FindAndConstruct(Pred);
      if (BBMapBucket.second) {
        Info->Preds[p] = BBMapBucket.second;
        continue;
      }

      // Create a new BBInfo for the predecessor.
      Value *PredVal = AvailableVals.lookup(Pred);
      BBInfo *PredInfo = new (*Allocator) BBInfo(Pred, PredVal);
      BBMapBucket.second = PredInfo;
      Info->Preds[p] = PredInfo;

      if (PredInfo->AvailableVal) {
        RootList.push_back(PredInfo);
        continue;
      }
      WorkList.push_back(PredInfo);
    }
  }

  // Now that we know what blocks are backwards-reachable from the starting
  // block, do a forward depth-first traversal to assign postorder numbers
  // to those blocks.
  BBInfo *PseudoEntry = new (*Allocator) BBInfo(0, 0);
  unsigned BlkNum = 1;

  // Initialize the worklist with the roots from the backward traversal.
  while (!RootList.empty()) {
    Info = RootList.pop_back_val();
    Info->IDom = PseudoEntry;
    Info->BlkNum = -1;
    WorkList.push_back(Info);
  }

  while (!WorkList.empty()) {
    Info = WorkList.back();

    if (Info->BlkNum == -2) {
      // All the successors have been handled; assign the postorder number.
      Info->BlkNum = BlkNum++;
      // If not a root, put it on the BlockList.
      if (!Info->AvailableVal)
        BlockList->push_back(Info);
      WorkList.pop_back();
      continue;
    }

    // Leave this entry on the worklist, but set its BlkNum to mark that its
    // successors have been put on the worklist.  When it returns to the top
    // the list, after handling its successors, it will be assigned a number.
    Info->BlkNum = -2;

    // Add unvisited successors to the work list.
    for (succ_iterator SI = succ_begin(Info->BB), E = succ_end(Info->BB);
         SI != E; ++SI) {
      BBInfo *SuccInfo = (*BBMap)[*SI];
      if (!SuccInfo || SuccInfo->BlkNum)
        continue;
      SuccInfo->BlkNum = -1;
      WorkList.push_back(SuccInfo);
    }
  }
  PseudoEntry->BlkNum = BlkNum;
}

/// IntersectDominators - This is the dataflow lattice "meet" operation for
/// finding dominators.  Given two basic blocks, it walks up the dominator
/// tree until it finds a common dominator of both.  It uses the postorder
/// number of the blocks to determine how to do that.
static SSAUpdater::BBInfo *IntersectDominators(SSAUpdater::BBInfo *Blk1,
                                               SSAUpdater::BBInfo *Blk2) {
  while (Blk1 != Blk2) {
    while (Blk1->BlkNum < Blk2->BlkNum) {
      Blk1 = Blk1->IDom;
      if (!Blk1)
        return Blk2;
    }
    while (Blk2->BlkNum < Blk1->BlkNum) {
      Blk2 = Blk2->IDom;
      if (!Blk2)
        return Blk1;
    }
  }
  return Blk1;
}

/// FindDominators - Calculate the dominator tree for the subset of the CFG
/// corresponding to the basic blocks on the BlockList.  This uses the
/// algorithm from: "A Simple, Fast Dominance Algorithm" by Cooper, Harvey and
/// Kennedy, published in Software--Practice and Experience, 2001, 4:1-10.
/// Because the CFG subset does not include any edges leading into blocks that
/// define the value, the results are not the usual dominator tree.  The CFG
/// subset has a single pseudo-entry node with edges to a set of root nodes
/// for blocks that define the value.  The dominators for this subset CFG are
/// not the standard dominators but they are adequate for placing PHIs within
/// the subset CFG.
void SSAUpdater::FindDominators(BlockListTy *BlockList) {
  bool Changed;
  do {
    Changed = false;
    // Iterate over the list in reverse order, i.e., forward on CFG edges.
    for (BlockListTy::reverse_iterator I = BlockList->rbegin(),
           E = BlockList->rend(); I != E; ++I) {
      BBInfo *Info = *I;

      // Start with the first predecessor.
      assert(Info->NumPreds > 0 && "unreachable block");
      BBInfo *NewIDom = Info->Preds[0];

      // Iterate through the block's other predecessors.
      for (unsigned p = 1; p != Info->NumPreds; ++p) {
        BBInfo *Pred = Info->Preds[p];
        NewIDom = IntersectDominators(NewIDom, Pred);
      }

      // Check if the IDom value has changed.
      if (NewIDom != Info->IDom) {
        Info->IDom = NewIDom;
        Changed = true;
      }
    }
  } while (Changed);
}

/// IsDefInDomFrontier - Search up the dominator tree from Pred to IDom for
/// any blocks containing definitions of the value.  If one is found, then the
/// successor of Pred is in the dominance frontier for the definition, and
/// this function returns true.
static bool IsDefInDomFrontier(const SSAUpdater::BBInfo *Pred,
                               const SSAUpdater::BBInfo *IDom) {
  for (; Pred != IDom; Pred = Pred->IDom) {
    if (Pred->DefBB == Pred)
      return true;
  }
  return false;
}

/// FindPHIPlacement - PHIs are needed in the iterated dominance frontiers of
/// the known definitions.  Iteratively add PHIs in the dom frontiers until
/// nothing changes.  Along the way, keep track of the nearest dominating
/// definitions for non-PHI blocks.
void SSAUpdater::FindPHIPlacement(BlockListTy *BlockList) {
  bool Changed;
  do {
    Changed = false;
    // Iterate over the list in reverse order, i.e., forward on CFG edges.
    for (BlockListTy::reverse_iterator I = BlockList->rbegin(),
           E = BlockList->rend(); I != E; ++I) {
      BBInfo *Info = *I;

      // If this block already needs a PHI, there is nothing to do here.
      if (Info->DefBB == Info)
        continue;

      // Default to use the same def as the immediate dominator.
      BBInfo *NewDefBB = Info->IDom->DefBB;
      for (unsigned p = 0; p != Info->NumPreds; ++p) {
        if (IsDefInDomFrontier(Info->Preds[p], Info->IDom)) {
          // Need a PHI here.
          NewDefBB = Info;
          break;
        }
      }

      // Check if anything changed.
      if (NewDefBB != Info->DefBB) {
        Info->DefBB = NewDefBB;
        Changed = true;
      }
    }
  } while (Changed);
}

/// FindAvailableVal - If this block requires a PHI, first check if an existing
/// PHI matches the PHI placement and reaching definitions computed earlier,
/// and if not, create a new PHI.  Visit all the block's predecessors to
/// calculate the available value for each one and fill in the incoming values
/// for a new PHI.
void SSAUpdater::FindAvailableVals(BlockListTy *BlockList) {
  AvailableValsTy &AvailableVals = getAvailableVals(AV);

  // Go through the worklist in forward order (i.e., backward through the CFG)
  // and check if existing PHIs can be used.  If not, create empty PHIs where
  // they are needed.
  for (BlockListTy::iterator I = BlockList->begin(), E = BlockList->end();
       I != E; ++I) {
    BBInfo *Info = *I;
    // Check if there needs to be a PHI in BB.
    if (Info->DefBB != Info)
      continue;

    // Look for an existing PHI.
    FindExistingPHI(Info->BB, BlockList);
    if (Info->AvailableVal)
      continue;

    PHINode *PHI = PHINode::Create(PrototypeValue->getType(),
                                   PrototypeValue->getName(),
                                   &Info->BB->front());
    PHI->reserveOperandSpace(Info->NumPreds);
    Info->AvailableVal = PHI;
    AvailableVals[Info->BB] = PHI;
  }

  // Now go back through the worklist in reverse order to fill in the arguments
  // for any new PHIs added in the forward traversal.
  for (BlockListTy::reverse_iterator I = BlockList->rbegin(),
         E = BlockList->rend(); I != E; ++I) {
    BBInfo *Info = *I;

    if (Info->DefBB != Info) {
      // Record the available value at join nodes to speed up subsequent
      // uses of this SSAUpdater for the same value.
      if (Info->NumPreds > 1)
        AvailableVals[Info->BB] = Info->DefBB->AvailableVal;
      continue;
    }

    // Check if this block contains a newly added PHI.
    PHINode *PHI = dyn_cast<PHINode>(Info->AvailableVal);
    if (!PHI || PHI->getNumIncomingValues() == Info->NumPreds)
      continue;

    // Iterate through the block's predecessors.
    for (unsigned p = 0; p != Info->NumPreds; ++p) {
      BBInfo *PredInfo = Info->Preds[p];
      BasicBlock *Pred = PredInfo->BB;
      // Skip to the nearest preceding definition.
      if (PredInfo->DefBB != PredInfo)
        PredInfo = PredInfo->DefBB;
      PHI->addIncoming(PredInfo->AvailableVal, Pred);
    }

    DEBUG(dbgs() << "  Inserted PHI: " << *PHI << "\n");

    // If the client wants to know about all new instructions, tell it.
    if (InsertedPHIs) InsertedPHIs->push_back(PHI);
  }
}

/// FindExistingPHI - Look through the PHI nodes in a block to see if any of
/// them match what is needed.
void SSAUpdater::FindExistingPHI(BasicBlock *BB, BlockListTy *BlockList) {
  PHINode *SomePHI;
  for (BasicBlock::iterator It = BB->begin();
       (SomePHI = dyn_cast<PHINode>(It)); ++It) {
    if (CheckIfPHIMatches(SomePHI)) {
      RecordMatchingPHI(SomePHI);
      break;
    }
    // Match failed: clear all the PHITag values.
    for (BlockListTy::iterator I = BlockList->begin(), E = BlockList->end();
         I != E; ++I)
      (*I)->PHITag = 0;
  }
}

/// CheckIfPHIMatches - Check if a PHI node matches the placement and values
/// in the BBMap.
bool SSAUpdater::CheckIfPHIMatches(PHINode *PHI) {
  BBMapTy *BBMap = getBBMap(BM);
  SmallVector<PHINode*, 20> WorkList;
  WorkList.push_back(PHI);

  // Mark that the block containing this PHI has been visited.
  (*BBMap)[PHI->getParent()]->PHITag = PHI;

  while (!WorkList.empty()) {
    PHI = WorkList.pop_back_val();

    // Iterate through the PHI's incoming values.
    for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i) {
      Value *IncomingVal = PHI->getIncomingValue(i);
      BBInfo *PredInfo = (*BBMap)[PHI->getIncomingBlock(i)];
      // Skip to the nearest preceding definition.
      if (PredInfo->DefBB != PredInfo)
        PredInfo = PredInfo->DefBB;

      // Check if it matches the expected value.
      if (PredInfo->AvailableVal) {
        if (IncomingVal == PredInfo->AvailableVal)
          continue;
        return false;
      }

      // Check if the value is a PHI in the correct block.
      PHINode *IncomingPHIVal = dyn_cast<PHINode>(IncomingVal);
      if (!IncomingPHIVal || IncomingPHIVal->getParent() != PredInfo->BB)
        return false;

      // If this block has already been visited, check if this PHI matches.
      if (PredInfo->PHITag) {
        if (IncomingPHIVal == PredInfo->PHITag)
          continue;
        return false;
      }
      PredInfo->PHITag = IncomingPHIVal;

      WorkList.push_back(IncomingPHIVal);
    }
  }
  return true;
}

/// RecordMatchingPHI - For a PHI node that matches, record it and its input
/// PHIs in both the BBMap and the AvailableVals mapping.
void SSAUpdater::RecordMatchingPHI(PHINode *PHI) {
  BBMapTy *BBMap = getBBMap(BM);
  AvailableValsTy &AvailableVals = getAvailableVals(AV);
  SmallVector<PHINode*, 20> WorkList;
  WorkList.push_back(PHI);

  // Record this PHI.
  BasicBlock *BB = PHI->getParent();
  AvailableVals[BB] = PHI;
  (*BBMap)[BB]->AvailableVal = PHI;

  while (!WorkList.empty()) {
    PHI = WorkList.pop_back_val();

    // Iterate through the PHI's incoming values.
    for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i) {
      PHINode *IncomingPHIVal = dyn_cast<PHINode>(PHI->getIncomingValue(i));
      if (!IncomingPHIVal) continue;
      BB = IncomingPHIVal->getParent();
      BBInfo *Info = (*BBMap)[BB];
      if (!Info || Info->AvailableVal)
        continue;

      // Record the PHI and add it to the worklist.
      AvailableVals[BB] = IncomingPHIVal;
      Info->AvailableVal = IncomingPHIVal;
      WorkList.push_back(IncomingPHIVal);
    }
  }
}
