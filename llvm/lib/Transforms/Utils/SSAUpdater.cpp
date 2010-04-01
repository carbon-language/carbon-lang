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
  Value *AvailableVal; // Value to use in this block.
  BasicBlock *DefBB;   // Block that defines the available value.
  unsigned NumPreds;   // Number of predecessor blocks.
  BasicBlock **Preds;  // Array[NumPreds] of predecessor blocks.
  unsigned Counter;    // Marker to identify blocks already visited.
  PHINode *PHITag;     // Marker for existing PHIs that match.

  BBInfo(BasicBlock *BB, Value *V, BumpPtrAllocator *Allocator);
};
typedef DenseMap<BasicBlock*, SSAUpdater::BBInfo*> BBMapTy;

SSAUpdater::BBInfo::BBInfo(BasicBlock *BB, Value *V,
                           BumpPtrAllocator *Allocator)
  : AvailableVal(V), DefBB(0), NumPreds(0), Preds(0), Counter(0), PHITag(0) {
  // If this block has a known value, don't bother finding its predecessors.
  if (V) {
    DefBB = BB;
    return;
  }

  // We can get our predecessor info by walking the pred_iterator list, but it
  // is relatively slow.  If we already have PHI nodes in this block, walk one
  // of them to get the predecessor list instead.
  if (PHINode *SomePhi = dyn_cast<PHINode>(BB->begin())) {
    NumPreds = SomePhi->getNumIncomingValues();
    Preds = static_cast<BasicBlock**>
      (Allocator->Allocate(NumPreds * sizeof(BasicBlock*),
                           AlignOf<BasicBlock*>::Alignment));
    for (unsigned pi = 0; pi != NumPreds; ++pi)
      Preds[pi] = SomePhi->getIncomingBlock(pi);
    return;
  }

  // Stash the predecessors in a temporary vector until we know how much space
  // to allocate for them.
  SmallVector<BasicBlock*, 10> TmpPreds;
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    TmpPreds.push_back(*PI);
    ++NumPreds;
  } 
  Preds = static_cast<BasicBlock**>
    (Allocator->Allocate(NumPreds * sizeof(BasicBlock*),
                         AlignOf<BasicBlock*>::Alignment));
  memcpy(Preds, TmpPreds.data(), NumPreds * sizeof(BasicBlock*));
}

typedef DenseMap<BasicBlock*, Value*> AvailableValsTy;
static AvailableValsTy &getAvailableVals(void *AV) {
  return *static_cast<AvailableValsTy*>(AV);
}

static BBMapTy *getBBMap(void *BM) {
  return static_cast<BBMapTy*>(BM);
}

static BumpPtrAllocator *getAllocator(void *BPA) {
  return static_cast<BumpPtrAllocator*>(BPA);
}

SSAUpdater::SSAUpdater(SmallVectorImpl<PHINode*> *NewPHI)
  : AV(0), PrototypeValue(0), BM(0), BPA(0), InsertedPHIs(NewPHI) {}

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
  assert(BM == 0 && BPA == 0 && "Unexpected Internal State");
  Value *Res = GetValueAtEndOfBlockInternal(BB);
  assert(BM == 0 && BPA == 0 && "Unexpected Internal State");
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
  BumpPtrAllocator AllocatorObj;
  BBMapTy BBMapObj;
  BPA = &AllocatorObj;
  BM = &BBMapObj;

  BBInfo *Info = new (AllocatorObj) BBInfo(BB, 0, &AllocatorObj);
  BBMapObj[BB] = Info;

  bool Changed;
  unsigned Counter = 1;
  do {
    Changed = false;
    FindPHIPlacement(BB, Info, Changed, Counter);
    ++Counter;
  } while (Changed);

  FindAvailableVal(BB, Info, Counter);

  BPA = 0;
  BM = 0;
  return Info->AvailableVal;
}

/// FindPHIPlacement - Recursively visit the predecessors of a block to find
/// the reaching definition for each predecessor and then determine whether
/// a PHI is needed in this block.  
void SSAUpdater::FindPHIPlacement(BasicBlock *BB, BBInfo *Info, bool &Changed,
                                  unsigned Counter) {
  AvailableValsTy &AvailableVals = getAvailableVals(AV);
  BBMapTy *BBMap = getBBMap(BM);
  BumpPtrAllocator *Allocator = getAllocator(BPA);
  bool BBNeedsPHI = false;
  BasicBlock *SamePredDefBB = 0;

  // If there are no predecessors, then we must have found an unreachable
  // block.  Treat it as a definition with 'undef'.
  if (Info->NumPreds == 0) {
    Info->AvailableVal = UndefValue::get(PrototypeValue->getType());
    Info->DefBB = BB;
    return;
  }

  Info->Counter = Counter;
  for (unsigned pi = 0; pi != Info->NumPreds; ++pi) {
    BasicBlock *Pred = Info->Preds[pi];
    BBMapTy::value_type &BBMapBucket = BBMap->FindAndConstruct(Pred);
    if (!BBMapBucket.second) {
      Value *PredVal = AvailableVals.lookup(Pred);
      BBMapBucket.second = new (*Allocator) BBInfo(Pred, PredVal, Allocator);
    }
    BBInfo *PredInfo = BBMapBucket.second;
    BasicBlock *DefBB = 0;
    if (!PredInfo->AvailableVal) {
      if (PredInfo->Counter != Counter)
        FindPHIPlacement(Pred, PredInfo, Changed, Counter);

      // Ignore back edges where the value is not yet known.
      if (!PredInfo->DefBB)
        continue;
    }
    DefBB = PredInfo->DefBB;

    if (!SamePredDefBB)
      SamePredDefBB = DefBB;
    else if (DefBB != SamePredDefBB)
      BBNeedsPHI = true;
  }

  BasicBlock *NewDefBB = (BBNeedsPHI ? BB : SamePredDefBB);
  if (Info->DefBB != NewDefBB) {
    Changed = true;
    Info->DefBB = NewDefBB;
  }
}

/// FindAvailableVal - If this block requires a PHI, first check if an existing
/// PHI matches the PHI placement and reaching definitions computed earlier,
/// and if not, create a new PHI.  Visit all the block's predecessors to
/// calculate the available value for each one and fill in the incoming values
/// for a new PHI.
void SSAUpdater::FindAvailableVal(BasicBlock *BB, BBInfo *Info,
                                  unsigned Counter) {
  if (Info->AvailableVal || Info->Counter == Counter)
    return;

  AvailableValsTy &AvailableVals = getAvailableVals(AV);
  BBMapTy *BBMap = getBBMap(BM);

  // Check if there needs to be a PHI in BB.
  PHINode *NewPHI = 0;
  if (Info->DefBB == BB) {
    // Look for an existing PHI.
    FindExistingPHI(BB, Info);
    if (!Info->AvailableVal) {
      NewPHI = PHINode::Create(PrototypeValue->getType(),
                               PrototypeValue->getName(), &BB->front());
      NewPHI->reserveOperandSpace(Info->NumPreds);
      Info->AvailableVal = NewPHI;
      AvailableVals[BB] = NewPHI;
    }
  }

  // Iterate through the block's predecessors.
  Info->Counter = Counter;
  for (unsigned pi = 0; pi != Info->NumPreds; ++pi) {
    BasicBlock *Pred = Info->Preds[pi];
    BBInfo *PredInfo = (*BBMap)[Pred];
    FindAvailableVal(Pred, PredInfo, Counter);
    if (NewPHI) {
      // Skip to the nearest preceding definition.
      if (PredInfo->DefBB != Pred)
        PredInfo = (*BBMap)[PredInfo->DefBB];
      NewPHI->addIncoming(PredInfo->AvailableVal, Pred);
    } else if (!Info->AvailableVal)
      Info->AvailableVal = PredInfo->AvailableVal;
  }
 
  if (NewPHI) {
    DEBUG(dbgs() << "  Inserted PHI: " << *NewPHI << "\n");

    // If the client wants to know about all new instructions, tell it.
    if (InsertedPHIs) InsertedPHIs->push_back(NewPHI);
  }
}

/// FindExistingPHI - Look through the PHI nodes in a block to see if any of
/// them match what is needed.
void SSAUpdater::FindExistingPHI(BasicBlock *BB, BBInfo *Info) {
  PHINode *SomePHI;
  for (BasicBlock::iterator It = BB->begin();
       (SomePHI = dyn_cast<PHINode>(It)); ++It) {
    if (CheckIfPHIMatches(BB, Info, SomePHI)) {
      RecordMatchingPHI(BB, Info, SomePHI);
      break;
    }
    ClearPHITags(SomePHI);
  }
}

/// CheckIfPHIMatches - Check if Val is a PHI node in block BB that matches
/// the placement and values in the BBMap.
bool SSAUpdater::CheckIfPHIMatches(BasicBlock *BB, BBInfo *Info, Value *Val) {
  if (Info->AvailableVal)
    return Val == Info->AvailableVal;

  // Check if Val is a PHI in this block.
  PHINode *PHI = dyn_cast<PHINode>(Val);
  if (!PHI || PHI->getParent() != BB)
    return false;

  // If this block has already been visited, check if this PHI matches.
  if (Info->PHITag)
    return PHI == Info->PHITag;
  Info->PHITag = PHI;
  bool IsMatch = true;

  // Iterate through the predecessors.
  BBMapTy *BBMap = getBBMap(BM);
  for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *Pred = PHI->getIncomingBlock(i);
    Value *IncomingVal = PHI->getIncomingValue(i);
    BBInfo *PredInfo = (*BBMap)[Pred];
    // Skip to the nearest preceding definition.
    if (PredInfo->DefBB != Pred) {
      Pred = PredInfo->DefBB;
      PredInfo = (*BBMap)[Pred];
    }
    if (!CheckIfPHIMatches(Pred, PredInfo, IncomingVal)) {
      IsMatch = false;
      break;
    }
  }
  return IsMatch;
}

/// RecordMatchingPHI - For a PHI node that matches, record it in both the
/// BBMap and the AvailableVals mapping.  Recursively record its input PHIs
/// as well.
void SSAUpdater::RecordMatchingPHI(BasicBlock *BB, BBInfo *Info, PHINode *PHI) {
  if (!Info || Info->AvailableVal)
    return;

  // Record the PHI.
  AvailableValsTy &AvailableVals = getAvailableVals(AV);
  AvailableVals[BB] = PHI;
  Info->AvailableVal = PHI;

  // Iterate through the predecessors.
  BBMapTy *BBMap = getBBMap(BM);
  for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i) {
    PHINode *PHIVal = dyn_cast<PHINode>(PHI->getIncomingValue(i));
    if (!PHIVal) continue;
    BasicBlock *Pred = PHIVal->getParent();
    RecordMatchingPHI(Pred, (*BBMap)[Pred], PHIVal);
  }
}

/// ClearPHITags - When one of the existing PHI nodes fails to match, clear
/// the PHITag values that were stored in the BBMap when checking to see if
/// it matched.
void SSAUpdater::ClearPHITags(PHINode *PHI) {
  BBMapTy *BBMap = getBBMap(BM);
  SmallVector<PHINode*, 20> WorkList;
  WorkList.push_back(PHI);

  while (!WorkList.empty()) {
    PHI = WorkList.pop_back_val();
    BasicBlock *BB = PHI->getParent();
    BBInfo *Info = (*BBMap)[BB];
    if (!Info || Info->AvailableVal || !Info->PHITag)
      continue;

    // Clear the tag.
    Info->PHITag = 0;

    // Iterate through the PHI's incoming values.
    for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i) {
      PHINode *IncomingVal = dyn_cast<PHINode>(PHI->getIncomingValue(i));
      if (!IncomingVal) continue;
      WorkList.push_back(IncomingVal);
    }
  }
}
